import logging
import os
import asyncio
import inspect
import json
import re # For parsing and sanitizing
from typing import AsyncGenerator, List, Dict, Optional, Any
import time

from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent # SequentialAgent not used directly by orchestrator in this version
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from google.adk.models import LlmResponse
import google.genai as genai
from google.adk.agents.callback_context import CallbackContext

from .output_schemas import OutlineExpertOutput, ConsistencyCheckOutput, SectionQualityOutput
            

# --- Configuration ---
APP_NAME = os.environ.get("REPORT_APP_NAME", "Business Report Generator")
USER_ID = os.environ.get("REPORT_USER_ID", "other_agents")
SESSION_ID_COUNTER = 0
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")

RATE_LIMIT_DELAY=3
MAX_CONSISTENCY_REVISIONS = 5
MAX_SECTION_REVISIONS = 2

logging.basicConfig(
    filename="wisecouncilrun.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# --- LLM Instruction Prompts ---
LLM_INSTRUCTIONS = {
    "OUTLINE_EXPERT_GENERATOR": """You are a business strategy consultant.
Given a business idea summary in session state ['input_summary'], your tasks are:
1.  Create a compelling title for the report.
2.  Generate a clear report outline with exactly 6 section titles. These sections should cover typical aspects of a business plan or investment opportunity analysis (e.g., 'Market Analysis', 'Product/Service Offering', 'Financial Projections', 'Management Team', 'Risk Assessment & Mitigation', 'Operational Analysis', 'Competitive Landscape', 'Implementation Strategy').
3.  For each of the 6 outline sections, create a short prompt outlining the expertise needed for a hypothetical expert to write that section.

IMPORTANT: Do NOT include "Executive Summary", "Introduction", "Conclusion", or "Summary" as section titles. These will be handled separately. Focus on analytical and substantive content sections only.

Output your response as a single JSON object with the following structure:
{
  "report_title": "Your Report Title",
  "outline_sections": ["Section 1 Title", "Section 2 Title", ..., "Section 6 Title"],
  "experts": [
    {"prompt": "Expert 1 prompt", "assigned_section_title": "Section 1 Title"},
    ...
    {"prompt": "Expert 6 prompt", "assigned_section_title": "Section 6 Title"},
  ]
}
Ensure the JSON is valid and all fields are populated correctly. There must be exactly 6 sections and 6 experts, with each expert assigned to one unique section.
""",
    "REPORT_CONSISTENCY_CHECKER": """You are a meticulous editor-in-chief.
The draft business report is provided in session state ['draft_report_for_consistency']. It's a concatenation of sections, each starting with "Section Title: ...".
Review the entire draft report for:
1.  Internal consistency: Do different sections contradict each other?
2.  Logical flow: Does the report progress logically from one section to the next?
3.  Redundancy: Is there unnecessary repetition of information across sections?
4.  Math: Are any numerical values incorrect? Are all the numerators and denominators consistent and correct ie has a daily value become a weekly value or similar?

Output your findings as a JSON object with the following structure:
{
  "consistent": true/false,
  "issues": [
    {
      "section": "Section Title or 'Overall Report'",
      "description": "Detailed description of the inconsistency"
    }
  ]
}
If consistent is true, the issues array should be empty or null.
""",
    "FINAL_REVIEWER": """You are a final proofreader and editor.
The almost-final business report is in session state ['report_for_final_review'].
The full report title is in session state ['report_title'].
The expert profiles are in session state ['expert_profiles'] (a list of dictionaries with prompts).
The report sections are in session state ['report_sections_content'] (a dictionary of section_title: content).

Your task is to:
1. Start the report with the main title (from ['report_title']) as a level 1 markdown heading (e.g., "# Report Title").
2. If "Executive Summary" exists in ['report_sections_content'], place it first as "## Executive Summary".
3. For the remaining sections in ['report_sections_content'], present their content under their respective titles as level 2 markdown headings (e.g., "## Section Title"). Use the section titles from ['outline_sections'] to determine the order of the main analytical sections (excluding Executive Summary).
4. Perform a light polish on the language for clarity, conciseness, and professionalism. Ensure a consistent tone.
Do not add new content or change the core meaning. Output only the final, polished report text.
""",
    "EXPERT_SECTION_WRITER_TEMPLATE": """{expert_prompt}
Your task is to write the '{section_title}' section of a business report.
The overall business idea/summary is in session state ['input_summary'].
The full report outline is in session state ['outline_sections'].
Focus ONLY on the '{section_title}' section.
Write a comprehensive two to four paragraphs (approx 500-1000 words) for this section. Include any relevant analytical tools that would be appropriate for this scenario.
If session state ['current_section_critique'] contains feedback (and is not 'Initial draft needed.'), address it in your revision.

IMPORTANT: Output ONLY the raw text content for the '{section_title}' section. Do not include any markdown formatting, code blocks, or conversational responses. Do not include section headers or titles. Start directly with the content. Do not make up any information not provided, instead make a note that more information is required to fully explore the idea. Do not include any [Insert Information Here] style tags.
Current draft (if any, for revision) is in session state ['current_section_content'].
""",
    "SECTION_QUALITY_CHECKER_TEMPLATE": """You are a quality assurance editor.
Review the content provided in session state ['current_section_content'] for the section titled '{section_title}'.
The overall business idea is in session state ['input_summary'].
Is the section content reasonable, well-written, on-topic for '{section_title}', and sufficiently detailed (around 500-1000 words)?
The section content strictly must not contain any [Add details here] like [Add Revenue number] or [Add source].
Output your findings as a JSON object with the following structure:
{{
  "status": "reasonable" or "needs_revision",
  "critique": "Brief, actionable feedback if revision needed, or null if reasonable"
}}
""",
    "EXPERT_SECTION_REVISER_TEMPLATE": """{expert_prompt}
You are revising the '{section_title}' section of a business report based on feedback.
The overall business idea/summary is in session state ['input_summary'].
The current draft of your section is in session state ['current_section_content'].
The feedback for revision is in session state ['current_section_critique'].
Address the feedback and provide the revised content for ONLY the '{section_title}' section.

IMPORTANT: Output ONLY the raw text content for the revised '{section_title}' section. Do not include any markdown formatting, code blocks, or conversational responses. Do not include section headers or titles. Start directly with the revised content.
""",
    "EXECUTIVE_SUMMARY_GENERATOR": """You are a senior business analyst tasked with creating an executive summary.
The complete business report content is provided in session state ['final_report_content']. This contains all the analytical sections that have been written and reviewed.
The original business idea/summary is in session state ['input_summary'].
The report title is in session state ['report_title'].

Your task is to create a comprehensive executive summary (2-3 paragraphs, approximately 100-500 words) that:
1. Briefly describes the business opportunity or situation being analyzed
2. Summarizes the key findings from each section of the completed report
3. Provides a clear, actionable recommendation based on the analysis
4. Highlights the most critical risks and opportunities identified
5. States the expected return on investment or key financial metrics if applicable

IMPORTANT: Base your summary ONLY on the content that has been written in the report sections. Do not add new analysis or speculation. This should be a true summary of the completed analysis, not a preview of what might be analyzed.

Output ONLY the raw text content for the executive summary. Do not include any markdown formatting, code blocks, conversational responses, or section headers. Start directly with the summary content.
"""
}

def _sanitize_for_agent_name(name_part: str) -> str:
    # Replace any character that is not an ASCII letter, digit, or underscore with an underscore
    s = re.sub(r'[^a-zA-Z0-9_]', '_', name_part)
    # If the result is empty (e.g., input was all special chars), provide a default
    # to ensure the agent name part is not empty.
    return s if s else "default_part"

def delay(callback_context: CallbackContext) -> None:
    time.sleep(RATE_LIMIT_DELAY)


# --- Helper Agent for Loop Control ---
class StopLoopIfConditionMetAgent(BaseAgent):
    """An agent that escalates (stops the loop) if a state condition is met."""
    condition_key: str
    desired_value: Any
    max_iterations_reached_key: Optional[str] = None

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        current_value = ctx.session.state.get(self.condition_key)
        should_stop = (current_value == self.desired_value)

        if self.max_iterations_reached_key and ctx.session.state.get(self.max_iterations_reached_key, False):
            logger.warning(f"[{self.name}] Max iterations reached for loop, forcing stop despite condition '{self.condition_key}' not being '{self.desired_value}'.")
            should_stop = True
            yield Event(author=self.name, actions=EventActions(state_delta={self.max_iterations_reached_key: None})) # Set to None or use a pop action if available

        logger.info(f"[{self.name}] Checking condition: {self.condition_key} ('{current_value}') == {self.desired_value}. Stop: {should_stop}")
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))


# --- Main Orchestrator Agent ---
class BusinessReportOrchestratorAgent(BaseAgent):
    outline_expert_generator: LlmAgent
    report_consistency_checker: LlmAgent
    executive_summary_generator: LlmAgent
    final_reviewer: LlmAgent

    model_config = {"arbitrary_types_allowed": True} 

    def __init__(
        self,
        name: str,
        outline_expert_generator: LlmAgent,
        report_consistency_checker: LlmAgent,
        executive_summary_generator: LlmAgent,
        final_reviewer: LlmAgent,
        before_agent_callback: Any = None,
    ):
        super().__init__(
            name=name,
            outline_expert_generator=outline_expert_generator,
            report_consistency_checker=report_consistency_checker,
            executive_summary_generator=executive_summary_generator,
            final_reviewer=final_reviewer,
            sub_agents=[
                outline_expert_generator,
                report_consistency_checker,
                executive_summary_generator,
                final_reviewer,
            ],
            before_agent_callback=before_agent_callback,
        )
        self.outline_expert_generator = outline_expert_generator
        self.report_consistency_checker = report_consistency_checker
        self.executive_summary_generator = executive_summary_generator
        self.final_reviewer = final_reviewer

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting Business Report Generation Workflow.")
        if inspect.iscoroutine(ctx.session):
            ctx.session = await ctx.session
       
        logger.info(f"[{self.name}] Running OutlineAndExpertGeneratorAgent...")
        yield Event(author=self.name, actions=EventActions(state_delta={"generated_report_structure_text": None})) # For callback
        async for event in self.outline_expert_generator.run_async(ctx):
            yield event
        # Data is now expected to be parsed by the callback and put directly into these keys
        report_title = ctx.session.state.get("report_title")
        outline_sections = ctx.session.state.get("outline_sections")
        expert_profiles = ctx.session.state.get("expert_profiles")

        if not (report_title and outline_sections and expert_profiles and len(outline_sections) == 6 and len(expert_profiles) == 6):
            logger.error(f"[{self.name}] Failed to generate or parse valid report structure. Aborting. "
                         f"Title: {report_title}, Sections: {outline_sections}, Experts: {expert_profiles}")
            yield Event(author=self.name, content=genai_types.Content(parts=[genai_types.Part(text="Error: Could not generate report outline and experts correctly.")]))
            return

        yield Event(author=self.name, actions=EventActions(state_delta={"report_sections_content": {}}))
        
        logger.info(f"[{self.name}] Report Title: {report_title}")
        logger.info(f"[{self.name}] Outline Sections: {outline_sections}")
        logger.info(f"[{self.name}] Expert Profiles: {json.dumps(expert_profiles, indent=2)}")

        # 2. Dynamically Invoke Agents for Each Section with Refinement Loop
        for i, section_title_from_outline in enumerate(outline_sections):
            # Find the expert assigned to this section title (case-insensitive match for robustness)
            current_expert_profile = next(
                (ep for ep in expert_profiles if ep.get("assigned_section_title", "").lower() == section_title_from_outline.lower()), 
                None
            )

            if not current_expert_profile:
                logger.warning(f"[{self.name}] No expert found for section: {section_title_from_outline}. Skipping.")
                ctx.session.state["report_sections_content"][section_title_from_outline] = f"[Content not generated: No expert assigned to this section: {section_title_from_outline}]"
                continue
            
            actual_section_title = current_expert_profile.get("assigned_section_title", section_title_from_outline) # Use title from expert profile if available
            expert_prompt = current_expert_profile.get("prompt", "You are an expert analyst.")

            logger.info(f"[{self.name}] Starting section: '{actual_section_title}' with prompt: {expert_prompt[:50]}...")
            yield Event(author=self.name, content=genai_types.Content(parts=[genai_types.Part(text=f"Writing section: {actual_section_title}...")]))

            ctx.session.state["current_section_title_being_written"] = actual_section_title
            ctx.session.state["current_expert_prompt"] = expert_prompt
            ctx.session.state["current_section_content"] = ""
            ctx.session.state["current_section_quality_status"] = "needs_revision"
            ctx.session.state["current_section_critique"] = "Initial draft needed."

            expert_section_writer = LlmAgent(
                name=f"ExpertSectionWriter_{_sanitize_for_agent_name(actual_section_title)}",
                model=GEMINI_MODEL,
                instruction=LLM_INSTRUCTIONS["EXPERT_SECTION_WRITER_TEMPLATE"].format(
                    expert_prompt=expert_prompt,
                    section_title=actual_section_title
                ),
                output_key="current_section_content",
                before_agent_callback=delay
            )

            section_quality_checker = LlmAgent(
                name=f"SectionQualityChecker_{_sanitize_for_agent_name(actual_section_title)}",
                model=GEMINI_MODEL,
                instruction=LLM_INSTRUCTIONS["SECTION_QUALITY_CHECKER_TEMPLATE"].format(
                    section_title=actual_section_title
                ),
                output_key="raw_section_quality_assessment", 
                output_schema=SectionQualityOutput,
                before_agent_callback=delay
            )
            
            def parse_quality_result_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
                text_output = ""
                if llm_response and llm_response.content and llm_response.content.parts:
                    text_parts = [part.text for part in llm_response.content.parts if hasattr(part, 'text') and part.text is not None]
                    if text_parts:
                        text_output = "".join(text_parts)
                
                if not text_output:
                    logger.warning(f"Quality checker for '{callback_context.state['current_section_title_being_written']}' produced no text output.")
                    callback_context.state["current_section_quality_status"] = "needs_revision" # Default to needs_revision
                    callback_context.state["current_section_critique"] = "Quality checker response was empty."
                    return

                # --- Start: Stripping Logic for JSON ---
                cleaned_json_string = text_output.strip()

                if cleaned_json_string.startswith("```json"):
                    cleaned_json_string = cleaned_json_string.removeprefix("```json")
                elif cleaned_json_string.startswith("```"):
                    cleaned_json_string = cleaned_json_string.removeprefix("```")

                if cleaned_json_string.endswith("```"):
                    cleaned_json_string = cleaned_json_string.removesuffix("```")

                cleaned_json_string = cleaned_json_string.strip()
                # --- End: Stripping Logic ---

                try:
                    data = json.loads(cleaned_json_string)
                    callback_context.state["current_section_quality_status"] = data.get("status", "needs_revision")
                    critique = data.get("critique")
                    callback_context.state["current_section_critique"] = critique if critique and critique.lower() != "none" else ""
                    logger.info(f"Quality Check for '{callback_context.state['current_section_title_being_written']}': {callback_context.state['current_section_quality_status']}, Critique: {callback_context.state['current_section_critique']}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from quality checker: {e}. Cleaned string (first 500 chars):\n{cleaned_json_string[:500]}...")
                    callback_context.state["current_section_quality_status"] = "needs_revision"
                    callback_context.state["current_section_critique"] = "Error parsing quality check JSON response."
                except Exception as e:
                    logger.error(f"Failed to process quality check data: {e}")
                    callback_context.state["current_section_quality_status"] = "needs_revision"
                    callback_context.state["current_section_critique"] = "Error processing quality check response."

            section_quality_checker.after_model_callback = parse_quality_result_callback

            stop_section_loop_agent = StopLoopIfConditionMetAgent(
                name=f"StopSectionLoop_{_sanitize_for_agent_name(actual_section_title)}",
                condition_key="current_section_quality_status",
                desired_value="reasonable",
                max_iterations_reached_key=f"max_iterations_section_{i}",
                before_agent_callback=delay
            )

            section_writer_loop = LoopAgent(
                name=f"SectionWriterLoop_{_sanitize_for_agent_name(actual_section_title)}",
                sub_agents=[expert_section_writer, section_quality_checker, stop_section_loop_agent],
                max_iterations=MAX_SECTION_REVISIONS,
                before_agent_callback=delay
            )

            async for event in section_writer_loop.run_async(ctx):
                yield event
            
            final_section_content = ctx.session.state.get("current_section_content", f"[Error generating content for {actual_section_title}]")
            ctx.session.state["report_sections_content"][actual_section_title] = final_section_content
            logger.info(f"[{self.name}] Finished section '{actual_section_title}'. Content length: {len(final_section_content)}")

        draft_report_for_consistency = "\n\n".join(
            f"Section Title: {title}\n{content}"
            for title, content in ctx.session.state["report_sections_content"].items()
        )
        ctx.session.state["draft_report_for_consistency"] = draft_report_for_consistency
        logger.info(f"[{self.name}] Draft report compiled for consistency check. Length: {len(draft_report_for_consistency)}")

        
        for attempt in range(MAX_CONSISTENCY_REVISIONS):
            logger.info(f"[{self.name}] Running ReportConsistencyCheckerAgent (Attempt {attempt + 1})...")
            # yield Event(author=self.name, content=genai_types.Content(parts=[genai_types.Part(text="Checking report consistency...")]))
            ctx.session.state["raw_consistency_check_output"] = None # For callback
            async for event in self.report_consistency_checker.run_async(ctx):
                yield event

            # Callback parses and puts 'is_consistent' (bool) and 'consistency_issues' (list of dicts) into state
            is_consistent = ctx.session.state.get("is_consistent", True) # Default to consistent if parsing fails
            consistency_issues = ctx.session.state.get("consistency_issues", [])
            
            if is_consistent:
                logger.info(f"[{self.name}] Report is consistent.")
                ctx.session.state["final_report_content"] = draft_report_for_consistency
                break
            else:
                logger.warning(f"[{self.name}] Report has inconsistencies: {json.dumps(consistency_issues, indent=2)}")
                if attempt < MAX_CONSISTENCY_REVISIONS - 1 and consistency_issues: # This condition will be false with MAX_CONSISTENCY_REVISIONS=1
                    yield Event(author=self.name, content=genai_types.Content(parts=[genai_types.Part(text="Revising sections for consistency...")]))
                    for issue in consistency_issues:
                        issue_section_title = issue.get("section_title")
                        issue_description = issue.get("description")
                        if issue_section_title and issue_section_title in ctx.session.state["report_sections_content"]:
                            logger.info(f"[{self.name}] Revising section '{issue_section_title}' due to consistency issue: {issue_description}")
                            
                            # Find the expert profile again
                            expert_profile_for_revision = next((ep for ep in expert_profiles if ep.get("assigned_section_title", "").lower() == issue_section_title.lower()), None)
                            if not expert_profile_for_revision: 
                                logger.warning(f"Cannot find expert for revision of {issue_section_title}")
                                continue
                            
                            ctx.session.state["current_section_title_being_written"] = issue_section_title
                            expert_prompt_for_revision = expert_profile_for_revision.get("prompt", "You are an expert analyst.")
                            ctx.session.state["current_expert_prompt"] = expert_prompt_for_revision
                            ctx.session.state["current_section_content"] = ctx.session.state["report_sections_content"][issue_section_title]
                            ctx.session.state["current_section_critique"] = (
                                f"CONSISTENCY FEEDBACK for section '{issue_section_title}': {issue_description}. "
                                "Please revise your section to address this."
                            )
                            
                            reviser_agent = LlmAgent(
                                name=f"ExpertSectionReviser_{_sanitize_for_agent_name(issue_section_title)}",
                                model=GEMINI_MODEL,
                                instruction=LLM_INSTRUCTIONS["EXPERT_SECTION_REVISER_TEMPLATE"].format(
                                    expert_prompt=expert_prompt_for_revision,
                                    section_title=issue_section_title
                                ),
                                output_key="current_section_content",
                                before_agent_callback=delay
                            )
                            async for rev_event in reviser_agent.run_async(ctx):
                                yield rev_event
                            
                            ctx.session.state["report_sections_content"][issue_section_title] = ctx.session.state.get("current_section_content", f"[Error revising {issue_section_title}]")
                            logger.info(f"[{self.name}] Revised section '{issue_section_title}'.")
                        else:
                             logger.warning(f"[{self.name}] Consistency issue for unknown or unhandled section '{issue_section_title}'. Cannot revise.")
                    
                    draft_report_for_consistency = "\n\n".join(
                        f"Section Title: {title}\n{content}"
                        for title, content in ctx.session.state["report_sections_content"].items()
                    )
                    ctx.session.state["draft_report_for_consistency"] = draft_report_for_consistency
                else: 
                    logger.warning(f"[{self.name}] Max consistency revisions reached or no actionable issues. Proceeding with current draft.")
                    ctx.session.state["final_report_content"] = draft_report_for_consistency
                    break
        else: 
            logger.warning(f"[{self.name}] Report may still have inconsistencies after {MAX_CONSISTENCY_REVISIONS} attempts.")
            ctx.session.state["final_report_content"] = draft_report_for_consistency

        # NEW: Generate Executive Summary based on completed content
        logger.info(f"[{self.name}] Running ExecutiveSummaryGeneratorAgent...")
        yield Event(author=self.name, content=genai_types.Content(parts=[genai_types.Part(text="Generating executive summary based on completed analysis...")]))
        async for event in self.executive_summary_generator.run_async(ctx):
            yield event
        
        executive_summary = ctx.session.state.get("executive_summary_content", "Error: Executive summary not generated.")
        logger.info(f"[{self.name}] Executive summary generated. Length: {len(executive_summary)}")
        
        # Add executive summary to the report sections content (it will be placed first by final reviewer)
        ctx.session.state["report_sections_content"]["Executive Summary"] = executive_summary

        logger.info(f"[{self.name}] Running FinalReviewerAgent...")
        # yield Event(author=self.name, content=genai_types.Content(parts=[genai_types.Part(text="Performing final review...")]))
        ctx.session.state["report_for_final_review"] = ctx.session.state.get("final_report_content", "Error: No content for final review.")
        async for event in self.final_reviewer.run_async(ctx):
            yield event
        
        final_report_text = ctx.session.state.get("final_polished_report", "Error: Final report not generated.")
        logger.info(f"[{self.name}] Workflow Finished. Final Report Length: {len(final_report_text)}")

        print(final_report_text)
        
        yield Event(
            author=self.name,
            content=genai_types.Content(parts=[genai_types.Part(text=final_report_text)]),
            turn_complete=True
        )

# --- Callback Functions for Parsing LLM Text Output ---

def parse_report_structure_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
    text_output = ""
    if llm_response and llm_response.content and llm_response.content.parts:
        text_parts = [part.text for part in llm_response.content.parts if hasattr(part, 'text') and part.text is not None]
        if text_parts:
            text_output = "".join(text_parts)
    
    if not text_output:
        logger.error("OutlineAndExpertGeneratorAgent produced no text output.")
        callback_context.state["report_title"] = None
        callback_context.state["outline_sections"] = None
        callback_context.state["expert_profiles"] = None
        return

    # --- Start: Stripping Logic - Assume markdown fences by default ---
    cleaned_json_string = text_output.strip() # Remove leading/trailing whitespace

    if cleaned_json_string.startswith("```json"):
        cleaned_json_string = cleaned_json_string.removeprefix("```json")
    elif cleaned_json_string.startswith("```"): # Handle case without 'json' identifier
        cleaned_json_string = cleaned_json_string.removeprefix("```")

    if cleaned_json_string.endswith("```"):
        cleaned_json_string = cleaned_json_string.removesuffix("```")

    # Optional: Strip again in case there was whitespace between content and fences
    cleaned_json_string = cleaned_json_string.strip()
    # --- End: Stripping Logic ---

    try: # Attempt to parse the cleaned string
        data = json.loads(cleaned_json_string)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON after stripping: {e}. Cleaned string (first 500 chars):\n{cleaned_json_string[:500]}...")
        logger.error(f"Original text_output (first 500 chars):\n{text_output[:500]}...")
        callback_context.state["report_title"] = None
        callback_context.state["outline_sections"] = None
        callback_context.state["expert_profiles"] = None
        return

    try: # Separate try-except for validating the *parsed* data structure
        callback_context.state["report_title"] = data.get("report_title")
        callback_context.state["outline_sections"] = data.get("outline_sections")
        callback_context.state["expert_profiles"] = data.get("experts")

        if not (isinstance(callback_context.state["report_title"], str) and
                isinstance(callback_context.state["outline_sections"], list) and
                len(callback_context.state["outline_sections"]) == 6 and
                isinstance(callback_context.state["expert_profiles"], list) and
                len(callback_context.state["expert_profiles"]) == 6 and
                all(isinstance(s, str) for s in callback_context.state["outline_sections"]) and
                all(isinstance(e, dict) and "prompt" in e and "assigned_section_title" in e
                    for e in callback_context.state["expert_profiles"])):
            raise ValueError("Parsed structure does not match expected format or count.")
        logger.info("Successfully parsed report structure from OutlineAndExpertGeneratorAgent.")

    except ValueError as e: # Catch only ValueError for structure validation now
        logger.error(f"Failed to parse report structure from LLM: {e}. Raw output:\n{text_output}")
        # Fallback or set error flags if necessary
        callback_context.state["report_title"] = None
        callback_context.state["outline_sections"] = None
        callback_context.state["expert_profiles"] = None


def parse_consistency_result_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
    text_output = ""
    if llm_response and llm_response.content and llm_response.content.parts:
        text_parts = [part.text for part in llm_response.content.parts if hasattr(part, 'text') and part.text is not None]
        if text_parts:
            text_output = "".join(text_parts)

    if not text_output:
        logger.warning("Consistency checker produced no text output.")
        callback_context.state["is_consistent"] = True 
        callback_context.state["consistency_issues"] = []
        return

    # --- Start: Stripping Logic for JSON ---
    cleaned_json_string = text_output.strip()

    if cleaned_json_string.startswith("```json"):
        cleaned_json_string = cleaned_json_string.removeprefix("```json")
    elif cleaned_json_string.startswith("```"):
        cleaned_json_string = cleaned_json_string.removeprefix("```")

    if cleaned_json_string.endswith("```"):
        cleaned_json_string = cleaned_json_string.removesuffix("```")

    cleaned_json_string = cleaned_json_string.strip()
    # --- End: Stripping Logic ---

    try:
        data = json.loads(cleaned_json_string)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from consistency checker: {e}. Cleaned string (first 500 chars):\n{cleaned_json_string[:500]}...")
        callback_context.state["is_consistent"] = True # Default to consistent
        callback_context.state["consistency_issues"] = []
        return

    try:
        callback_context.state["is_consistent"] = data.get("consistent", True)
        raw_issues = data.get("issues", [])
        
        # Convert issues to the expected format
        issues = []
        if raw_issues:
            for issue in raw_issues:
                if isinstance(issue, dict) and "section" in issue and "description" in issue:
                    issues.append({
                        "section_title": issue["section"],
                        "description": issue["description"]
                    })

        callback_context.state["consistency_issues"] = issues
        logger.info(f"Parsed Consistency Check: Consistent={callback_context.state['is_consistent']}, Issues: {len(issues)}")

    except Exception as e:
        logger.error(f"Failed to process consistency check data: {e}. Raw data: {data}")
        callback_context.state["is_consistent"] = True
        callback_context.state["consistency_issues"] = []


outline_expert_generator = LlmAgent(
    name="OutlineAndExpertGeneratorAgent",
    model=GEMINI_MODEL,
    instruction=LLM_INSTRUCTIONS["OUTLINE_EXPERT_GENERATOR"],
    output_key="generated_report_structure_text", 
    output_schema=OutlineExpertOutput,
    before_agent_callback=delay,
    after_model_callback=parse_report_structure_callback
)

report_consistency_checker = LlmAgent(
    name="ReportConsistencyCheckerAgent",
    model=GEMINI_MODEL,
    instruction=LLM_INSTRUCTIONS["REPORT_CONSISTENCY_CHECKER"],
    output_key="raw_consistency_check_output", # Raw text output
    output_schema=ConsistencyCheckOutput,
    after_model_callback=parse_consistency_result_callback,
    before_agent_callback=delay
)

executive_summary_generator = LlmAgent(
    name="ExecutiveSummaryGeneratorAgent",
    model=GEMINI_MODEL,
    instruction=LLM_INSTRUCTIONS["EXECUTIVE_SUMMARY_GENERATOR"],
    output_key="executive_summary_content",
    before_agent_callback=delay
)

final_reviewer = LlmAgent(
    name="FinalReviewerAgent",
    model=GEMINI_MODEL,
    instruction=LLM_INSTRUCTIONS["FINAL_REVIEWER"],
    output_key="final_polished_report",
    before_agent_callback=delay
)

# --- Create the Orchestrator Agent instance ---
report_orchestrator = BusinessReportOrchestratorAgent(
    name="BusinessReportOrchestrator",
    outline_expert_generator=outline_expert_generator,
    report_consistency_checker=report_consistency_checker,
    executive_summary_generator=executive_summary_generator,
    final_reviewer=final_reviewer,
    before_agent_callback=delay
)

# --- Setup Runner and Session Service ---
session_service = InMemorySessionService()

# --- Function to Interact with the Agent System ---
async def generate_report(business_summary: str) -> str:
    global SESSION_ID_COUNTER
    SESSION_ID_COUNTER += 1
    current_session_id = f"report_session_{SESSION_ID_COUNTER}"

    initial_state = {"input_summary": business_summary}
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=current_session_id,
        state=initial_state
    )
    logger.info(f"[{current_session_id}] Initial session state: {session.state}")

    runner = Runner(
        agent=report_orchestrator,
        app_name=APP_NAME,
        session_service=session_service
    )

    initial_message = genai_types.Content(role='user', parts=[genai_types.Part(text=business_summary)])
    
    final_report_text = "Report generation failed or did not complete."
    
    async for event in runner.run_async(user_id=USER_ID, session_id=current_session_id, new_message=initial_message):
        if event.is_final_response() and event.content and event.content.parts:
            final_report_text = event.content.parts[0].text
            logger.info(f"[{current_session_id}] Final Report Event from [{event.author}]:\n{final_report_text[:500]}...")

    final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id)
    if final_session:
        logger.info(f"[{current_session_id}] --- Final Session State (Brief) ---")
        outline_sections_val = final_session.state.get("outline_sections")
        expert_profiles_val = final_session.state.get("expert_profiles")
        brief_state = {
            "input_summary": final_session.state.get("input_summary"),
            "report_title": final_session.state.get("report_title"),
            "outline_sections_count": len(outline_sections_val if outline_sections_val is not None else []),
            "expert_profiles_count": len(expert_profiles_val if expert_profiles_val is not None else []),
            "report_sections_content_keys": list(final_session.state.get("report_sections_content", {}).keys()),
            "final_polished_report_exists": "final_polished_report" in final_session.state,
            "is_consistent_final": final_session.state.get("is_consistent"),
        }
        logger.info(json.dumps(brief_state, indent=2))

    return final_report_text
