import logging
import json, os
from typing import AsyncGenerator, List, Dict, Any
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from pydantic import BaseModel, Field, field_validator, ValidationError

# os.environ['GOOGLE_API_KEY'] = key

# --- Constants ---
APP_NAME = "report_app"
USER_ID = "user_decision_maker"
SESSION_ID = "session_complex_problem"
GEMINI_MODEL = "gemini-2.0-flash-lite" # Using a more capable model might be better for complex tasks
MAX_SECTIONS = 6
MAX_SECTION_REVISIONS = 2 # Set iterations for section critique/revise loop
MAX_FINAL_REVISIONS = 2   # Set iterations for final report critique/revise loop

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Agent for Assembling the Report ---
class ReportAssemblerAgent(BaseAgent):
    """
    A simple agent to assemble the report sections from state.
    """
    name: str = "ReportAssembler"

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Assembling report sections.")
        state = ctx.session.state
        report_parts = []
        report_structure = state.get("report_structure")
        section_titles = report_structure.get("report_sections", [f"Section {i+1}" for i in range(MAX_SECTIONS)]) if report_structure else [f"Section {i+1}" for i in range(MAX_SECTIONS)]


        for i in range(MAX_SECTIONS):
            section_content = state.get(f"section_{i+1}_content")
            title = section_titles[i]

            if section_content and title != "N/A" and section_content.strip().upper() != "N/A":
                report_parts.append(f"## {title}\n\n{section_content}\n")
            elif title != "N/A":
                 logger.warning(f"[{self.name}] Section {i+1} ('{title}') content not found or is N/A, skipping.")


        if not report_parts:
            logger.warning(f"[{self.name}] No valid sections found to assemble.")
            draft_report = "No report could be generated based on the available sections."
        else:
            draft_report = "\n".join(report_parts)

        state["draft_report"] = draft_report
        logger.info(f"[{self.name}] Draft report assembled.")
        # Yield an event to signal completion and potentially show the draft
        yield Event(
            author=self.name,
            content=types.Content(role="assistant", parts=[types.Part(text=draft_report)]),
            actions=EventActions(state_delta={"draft_report": draft_report})
        )


# --- Custom Orchestrator Agent ---
class ReportGenerationAgent(BaseAgent):
    """
    Orchestrates a multi-agent workflow for generating a report based on user input.
    """
    # Pydantic will manage these based on __init__ args
    planning_agent: LlmAgent
    prompt_generation_agent: LlmAgent
    report_writer_agents: List[LlmAgent]
    section_critique_agents: List[LlmAgent]
    section_revision_agents: List[LlmAgent]
    report_assembler_agent: ReportAssemblerAgent
    final_critique_agent: LlmAgent
    final_revision_agent: LlmAgent

    # Internal composite agents
    section_processor_agents: List[SequentialAgent] # Holds [Writer, Loop(Critique, Reviser)]
    sections_sequential_agent: SequentialAgent      # Runs all SectionProcessors
    final_review_loop: LoopAgent                    # Runs FinalCritique -> FinalRevision

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        planning_agent: LlmAgent,
        prompt_generation_agent: LlmAgent,
        report_writer_agents: List[LlmAgent],
        section_critique_agents: List[LlmAgent],
        section_revision_agents: List[LlmAgent],
        report_assembler_agent: ReportAssemblerAgent,
        final_critique_agent: LlmAgent,
        final_revision_agent: LlmAgent,
    ):
        # --- Create internal composite agents BEFORE super().__init__ ---
        section_processor_agents = []
        if len(report_writer_agents) != MAX_SECTIONS or \
           len(section_critique_agents) != MAX_SECTIONS or \
           len(section_revision_agents) != MAX_SECTIONS:
            raise ValueError(f"Must provide exactly 6 of each section agent type.")

        for i in range(MAX_SECTIONS):
            critique_revise_loop = LoopAgent(
                name=f"CritiqueReviseLoop_{i+1}",
                sub_agents=[section_critique_agents[i], section_revision_agents[i]],
                max_iterations=MAX_SECTION_REVISIONS
            )
            section_processor = SequentialAgent(
                name=f"SectionProcessor_{i+1}",
                sub_agents=[report_writer_agents[i], critique_revise_loop]
            )
            section_processor_agents.append(section_processor)

        sections_sequential_agent = SequentialAgent(
            name="SectionsSequentialAgent",
            sub_agents=section_processor_agents
        )

        final_review_loop = LoopAgent(
            name="FinalReviewLoop",
            sub_agents=[final_critique_agent, final_revision_agent],
            max_iterations=MAX_FINAL_REVISIONS
        )

        # --- Define the overall sequence for the orchestrator ---
        # This agent will run these steps sequentially in its _run_async_impl
        # We don't pass them as sub_agents to BaseAgent to have finer control
        # over the execution flow and logging within this orchestrator.
        # ---

        super().__init__(
            name=name,
            # Store references to all agents passed in for Pydantic validation
            # and potential direct use (though we'll primarily use the composite agents)
            planning_agent=planning_agent,
            prompt_generation_agent=prompt_generation_agent,
            report_writer_agents=report_writer_agents,
            section_critique_agents=section_critique_agents,
            section_revision_agents=section_revision_agents,
            report_assembler_agent=report_assembler_agent,
            final_critique_agent=final_critique_agent,
            final_revision_agent=final_revision_agent,
            # Store the composite agents as well
            section_processor_agents=section_processor_agents,
            sections_sequential_agent=sections_sequential_agent,
            final_review_loop=final_review_loop,
            # No sub_agents needed for BaseAgent itself in this pattern
            sub_agents=[]
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the report generation workflow.
        """
        logger.info(f"[{self.name}] Starting report generation workflow.")

        # --- Stage 1: Information Gathering (Simulated) ---
        # Assumes 'situation_summary' is in the initial state
        if "situation_summary" not in ctx.session.state:
            logger.error(f"[{self.name}] 'situation_summary' not found in initial state. Aborting.")
            yield Event(author=self.name)
            return
        logger.info(f"[{self.name}] Situation Summary: {ctx.session.state['situation_summary'][:100]}...")

        # --- Stage 2: Planning ---
        logger.info(f"[{self.name}] Running Planning Agent...")
        async for event in self.planning_agent.run_async(ctx):
            yield event
        if "plan" not in ctx.session.state:
             logger.error(f"[{self.name}] Failed to generate report structure. Aborting.")
             yield Event.final_response(author=self.name, text="Error: Failed planning stage.")
             return
        logger.info(f"[{self.name}] Planning complete. User Question: {ctx.session.state['plan']}")
        logger.info(f"[{self.name}] Report Sections: {ctx.session.state['plan']}")


        # --- Stage 3: Prompt Generation ---
        logger.info(f"[{self.name}] Running Prompt Generation Agent...")
        async for event in self.prompt_generation_agent.run_async(ctx):
            yield event

        if "writer_prompts" not in ctx.session.state:
             logger.error(f"[{self.name}] Failed to generate writer prompts. Aborting.")
             yield Event.final_response(author=self.name, text="Error: Failed prompt generation.")
             return

        try:
            writer_prompts_raw = ctx.session.state["writer_prompts"]
            if isinstance(writer_prompts_raw, str):
                logger.info(f"[{self.name}] 'writer_prompts' is a string, attempting JSON parsing.")
                cleaned_json_string = writer_prompts_raw.strip() # Remove leading/trailing whitespace

                # --- Start: Stripping Logic ---
                # Method 1: Using removeprefix/removesuffix (Python 3.9+) - Recommended
                if cleaned_json_string.startswith("```json"):
                    cleaned_json_string = cleaned_json_string.removeprefix("```json")
                elif cleaned_json_string.startswith("```"): # Handle case without 'json' identifier
                     cleaned_json_string = cleaned_json_string.removeprefix("```")

                if cleaned_json_string.endswith("```"):
                     cleaned_json_string = cleaned_json_string.removesuffix("```")

                # Optional: Strip again in case there was whitespace between content and fences
                cleaned_json_string = cleaned_json_string.strip()
                writer_prompts_data = json.loads(cleaned_json_string)
            else:
                raise TypeError(f"Unexpected type for 'writer_prompts': {type(writer_prompts_raw)}")
            
            logger.info(f"[{self.name}] Generated {writer_prompts_data} {cleaned_json_string} writer prompts.")
            prompts_list = writer_prompts_data["writer_prompts"]
            for i, p in enumerate(prompts_list):
                ctx.session.state[f"writer_system_prompt_{i+1}"] = p["system_prompt"]
                ctx.session.state[f"writer_prompt_{i+1}"] = p["user_prompt"]
            logger.info(f"[{self.name}] Generated and stored {len(prompts_list)} sets of writer prompts.")
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"[{self.name}] Error processing generated prompts: {e}. Aborting.")
            yield Event.final_response(author=self.name, text="Error: Problem processing generated prompts.")
            return

        # --- Stage 4: Section Writing & Revision ---
        logger.info(f"[{self.name}] Running Sections Sequential Agent...")
        # This agent sequentially runs each SectionProcessor (Writer -> Critique/Revise Loop)
        async for event in self.sections_sequential_agent.run_async(ctx):
            # Log events from sub-agents if needed for detailed tracing
            # logger.info(f"[{self.name}] Event from SectionsSequentialAgent: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        logger.info(f"[{self.name}] Section writing and revision complete.")
        # Optional: Add checks here to ensure all section_X_content keys exist


        # --- Stage 5: Report Assembly ---
        logger.info(f"[{self.name}] Running Report Assembler Agent...")
        async for event in self.report_assembler_agent.run_async(ctx):
             yield event
        if "draft_report" not in ctx.session.state:
             logger.error(f"[{self.name}] Failed to assemble draft report. Aborting.")
             yield Event(author=self.name, actions=EventActions(escalate=True))
             return
        logger.info(f"[{self.name}] Draft report generated.")
        # logger.debug(f"[{self.name}] Draft Report Preview:\n{ctx.session.state['draft_report'][:500]}...")


        # --- Stage 6: Final Review & Revision ---
        logger.info(f"[{self.name}] Running Final Review Loop...")
        # This loop runs FinalCritique -> FinalRevision
        async for event in self.final_review_loop.run_async(ctx):
            yield event
        if "final_report" not in ctx.session.state:
             logger.warning(f"[{self.name}] Final report key 'final_report' not found after final review loop. Using draft report.")
             # Fallback: Use the draft if final revision didn't populate the key
             ctx.session.state["final_report"] = ctx.session.state.get("draft_report", "Error: Report generation failed.")
        else:
             logger.info(f"[{self.name}] Final report revision complete.")


        # --- Stage 7: Presentation ---
        # The final event yielded should contain the report
        final_report_content = ctx.session.state.get("final_report", "Error: Could not retrieve final report.")
        logger.info(f"[{self.name}] Workflow finished. Presenting final report.")
        content=types.Content(role="assistant", parts=[types.Part(text=final_report_content)])
        yield Event(author=self.name, content=content)

# --- Define the individual LLM agents ---

# Stage 2 Agent
planning_agent = LlmAgent(
    name="PlanningAgent",
    model=GEMINI_MODEL,
    instruction="""You are a planning assistant. Analyze the user's situation provided in the situation_summary.
1. Determine the core underlying question the user needs answered or the decision they need help with and put it in the state plan['user_question'].
2. Define a report structure with exactly 6 sections, give each a title and one-sentence description, make sure the sections make sense and are arranged in the way that would best address this question. Use 'N/A' for any sections not needed. put it in plan['report_structure']
""",
    output_key="plan", # Stores the parsed Pydantic model object
)

# Stage 3 Agent
prompt_generation_agent = LlmAgent(
    name="PromptGenerationAgent",
    model=GEMINI_MODEL,
    instruction="""You are a prompt engineering specialist. Based on the 'situation_summary' and the 'report_structure' (containing 'user_question' and 'report_sections') provided in the session state:
Generate 6 sets of prompts for specialized writers. For each section title in 'report_sections':
1. Define a persona (system_prompt): A brief description of an expert suitable for writing that section (e.g., 'You are a financial analyst specializing in risk assessment.', 'You are a market researcher focusing on competitive landscapes.'). If the section title is 'N/A', use a generic persona like 'You are a helpful assistant.'
2. Define the task (user_prompt): A specific instruction for the writer telling them what content to generate for that section, referencing the overall 'situation_summary' and 'user_question' for context. If the section title is 'N/A', the task should be 'Write "N/A" for this section.'

Output only a JSON object with no escape characters like ``` for of the above into a json object with 'system_prompt_n' and 'user_prompt_n' keys. put the json object into state 'writer_prompts'""",    output_key="writer_prompts",
)

# Stage 4 Agents (Create lists for the 6 instances)
report_writer_agents = []
section_critique_agents = []
section_revision_agents = []

for i in range(1, MAX_SECTIONS + 1):
    # Writer - Uses generated prompts from state
    writer = LlmAgent(
        name=f"ReportWriterAgent_{i}",
        model=GEMINI_MODEL,
        # Instruction uses f-string formatting to pull keys from state at runtime
        instruction=f"""{{writer_system_prompt_{i}}}

Based on the overall situation described below:
--- SITUATION START ---
{{situation_summary}}
--- SITUATION END ---

Your task is to write the report section addressing the following specific request:
--- TASK START ---
{{writer_prompt_{i}}}
--- TASK END ---

Generate only the content for this specific section. If the task is to write 'N/A', output only N/A.""",
        output_key=f"section_{i}_content",
    )
    report_writer_agents.append(writer)

    # Section Critique
    critique = LlmAgent(
        name=f"SectionCritiqueAgent_{i}",
        model=GEMINI_MODEL,
        instruction=f"""You are a critical reviewer. Review the draft section provided in state key 'section_{i}_content'.
Consider the original goal (state key 'user_question'), the specific task for this section (state key 'writer_prompt_{i}'), and the overall context ('situation_summary').
Is the section clear, relevant, and does it sufficiently address its specific task? Provide 1-3 concise bullet points of constructive criticism for improvement. If the section is 'N/A' or already excellent, output 'No critique needed.'.""",
        output_key=f"section_{i}_critique",
    )
    section_critique_agents.append(critique)

    # Section Revision
    reviser = LlmAgent(
        name=f"SectionRevisionAgent_{i}",
        model=GEMINI_MODEL,
        instruction=f"""You are a section reviser. Revise the draft section provided in state key 'section_{i}_content' based *only* on the critique provided in state key 'section_{i}_critique'.
If the critique is 'No critique needed.', simply output the original section content unchanged.
If the section content is 'N/A', output 'N/A'.
Output only the revised section content.""",
        output_key=f"section_{i}_content", # Overwrites the original section
    )
    section_revision_agents.append(reviser)

# Stage 5 Agent
report_assembler_agent = ReportAssemblerAgent() # Use the custom class

# Stage 6 Agents
final_critique_agent = LlmAgent(
    name="FinalCritiqueAgent",
    model=GEMINI_MODEL,
    instruction="""You are a chief editor. Review the complete assembled report provided in state key 'draft_report'.
Consider the original user question (state key 'user_question') and the overall situation ('situation_summary').
Does the report flow logically? Is it coherent? Does it comprehensively address the user's core question? Are there any redundancies or inconsistencies between sections? Are there any template leftovers like 'insert percentage here' or other things that look unprofessional? Are there any simple errors of fact?
Provide 2-4 concise bullet points of constructive criticism for improving the *entire* report. If the report is excellent, output 'Report is final.'""",
    output_key="final_critique",
)

final_revision_agent = LlmAgent(
    name="FinalRevisionAgent",
    model=GEMINI_MODEL,
    instruction="""You are a final report editor. Revise the entire report draft (state key 'draft_report') based *only* on the final critique (state key 'final_critique').
Address the points raised in the critique to improve coherence, flow, and completeness.
If the critique is 'Report is final.', output the original report content unchanged.
Output *only* the full, revised final report content.""",
    output_key="final_report", # Writes to the final key
)


# --- Create the main orchestrator agent instance ---
report_generation_agent = ReportGenerationAgent(
    name="ReportGenerationAgent",
    planning_agent=planning_agent,
    prompt_generation_agent=prompt_generation_agent,
    report_writer_agents=report_writer_agents,
    section_critique_agents=section_critique_agents,
    section_revision_agents=section_revision_agents,
    report_assembler_agent=report_assembler_agent,
    final_critique_agent=final_critique_agent,
    final_revision_agent=final_revision_agent,
)

root_agent = LlmAgent(
    name="council_liason",
    model="gemini-2.0-flash",
    description=(
        "Agent to establish the base facts of a situation that a person is asking for advice about."
    ),
    instruction=(
        "You are the liason for a large network of agents who are all specialists or generalists and wise advisors, your goal is to understand the persons situation and their context. You will ask specific questions until you have a good understanding of the situation. You will ask if there is anything else after you have run out of specific questions. Once the user has no more questions, you will output the conversation to 'situation_summary' and then transfer the user to a report generation agent"
    ),
    output_key="situation_summary",
    sub_agents=[report_generation_agent],
)


# # --- Run the Agent ---
# run_report_generation()
