# Wise Council
A repository for the research and code initially developed for AISC 2025

Can a council of LLM's replicate the structural wisdom we build into our human organizations? Like a committee or standards organization that relies upon many people interacting with the work product and certain rules of review, can we get "wiser" output from our LLM's by implementing similar layered/hierarchical organizations where responsibilities are split?

My initial attempts at generating a protocol and doing this manually seemed promising. The protocol was essentially
1. Prompt the LLM for what personas/system prompts would be best suited to answering the question asked
2. Prompt the LLM for what the response should look like (what general topics would be of interest to the user), what format it should be in, and which personas should handle that section/topic.
3. Prompt the LLM with the persona and a request for it to generate that subsection
4. Prompt the LLM with the subsections and ask it to review and make only minor edits, or reject it if it needed major edits (and if so back to step 3)

This had subjectively good results, with more in-depth and useful answers. So I moved on to trying to measure the wisdom of the output. In https://github.com/monadsandtuples/wisecouncil/blob/main/notunwise.md I explore how we measure wisdom and come to the conclusion we have no good general measures for wisdom. This was disheartening, but I have gone on and made an attempt to automate the wise council anyway (https://github.com/monadsandtuples/wisecouncil/blob/main/wisecouncil.py) which can be used with google-adk, specifically https://github.com/google/adk-python

This newest uploaded version works as follows

1. Go to adk-python linked above and follow the instructions to install the adk
2. Set your environment variable in .env
3. Run adk web from the wisecouncil directory
4. Select the council_agent and talk with it until you are satisfied (this is handled in agent.py)
5. The report generation will happen in businessreportgenerator.py, this is usually about 30-40 calls to gemini

This is set up with a very basic rate limiter so you can use it on the free tier.
