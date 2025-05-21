import logging

from typing import AsyncGenerator # Removed List, Dict, Any as not directly used in this snippet
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent # Removed LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as genai_types # Using alias for clarity
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event, EventActions

from businessreportgenerator import generate_report

# --- Constants ---
APP_NAME = "Wise_Council_Lobby"
USER_ID = "user_decision_maker"
SESSION_ID = "session_complex_problem"
GEMINI_MODEL = "gemini-1.5-flash-latest" 


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportGeneratorCallerAgent(BaseAgent):
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Entered ReportGeneratorCallerAgent.")
        situation_summary = ctx.session.state.get("situation_summary")

        if not situation_summary:
            logger.error(f"[{self.name}] 'situation_summary' not found in session state.")
            yield Event(
                author=self.name,
                content=genai_types.Content(parts=[genai_types.Part(text="Error: Situation summary was not generated or found. Cannot generate report.")]),
                turn_complete=True
            )
            return

        logger.info(f"[{self.name}] Found situation summary (first 100 chars): '{str(situation_summary)[:100]}...'. Calling bulkagent.generate_report...")
        yield Event(
            author=self.name,
            content=genai_types.Content(parts=[genai_types.Part(text="Generating detailed report based on the summary...")])
            # This is an interim, non-final message
        )
        
        try:
            final_report_text = await generate_report(str(situation_summary))
            logger.info(f"[{self.name}] Report generation complete. Yielding final report.")
            yield Event(
                author=self.name,
                content=genai_types.Content(parts=[genai_types.Part(text=final_report_text)]),
                turn_complete=True
            )
        except Exception as e:
            logger.error(f"[{self.name}] Error calling generate_report: {e}", exc_info=True)
            yield Event(
                author=self.name,
                content=genai_types.Content(parts=[genai_types.Part(text=f"Error during report generation: {e}")]),
                turn_complete=True
            )




final_agent = LlmAgent(
    name="council_liason",
    model="gemini-2.0-flash",
    description=(
        "Agent to ensure the user is satisfies with their report."
    ),
    instruction=(
        "Ensure the user is satisfied with their report, and if not, gather more information"
    ),
    output_key="final output",   
    sub_agents=[],
)

waiting_room = SequentialAgent(
    name="waiting_room",
    sub_agents=[ReportGeneratorCallerAgent(name="WiseCouncilReportGenerator"), final_agent],
)


summary_agent = LlmAgent(
    name="council_liason_summary",
    model="gemini-2.0-flash",
    description=(
        "Agent to establish a summary of a situation that a person is asking for advice about."
    ),
    instruction=(
        "You are a liason for a large network of agents who are all specialists or generalists and wise advisors, your goal is to summarize the state key 'intake_data' and organize it so that it is more structured and useful for the experts. You will ask the user to confirm the data. Once the conversation is over, you will transfer the user to a waiting room. Let the user know in your last message that the report may take up to 5 minutes to generate."
    ),
    output_key="situation_summary",
    sub_agents=[waiting_room],
)


root_agent = LlmAgent(
    name="council_liason",
    model="gemini-2.0-flash",
    description=(
        "Agent to establish the base facts of a situation that a person is asking for advice about."
    ),
    instruction=(
        "You are the liason for a large network of agents who are all specialists or generalists and wise advisors, your goal is to understand the persons situation and their context. You will ask specific questions until you have a good understanding of the situation. You will ask if there is anything else after you have run out of specific questions. Once the user has no more questions, you will output the conversation to 'intake_data' and then transfer the user to a summary_agent"
    ),
    output_key="intake_data",
    sub_agents=[summary_agent],
)