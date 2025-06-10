# Wise Council
A repository for the research and code initially developed for AISC 2025

Can a council of LLM's replicate the structural wisdom we build into our human organizations? Like a committee or standards organization that relies upon many people interacting with the work product and certain rules of review, can we get "wiser" output from our LLM's by implementing similar layered/hierarchical organizations where responsibilities are split?

How to use:

1. Have python 3.10.x
2. Run start.sh
3. Describe your business problem/idea to the initial agent and approve the summary
4. The report generation will happen in businessreportgenerator.py, this is usually about 30-40 calls to gemini and can take a few minutes
5. Your report will be shown in the web interface and you are transferred back to the original agent

Other Notes:
This is set up with a rate limiter so you can use it on the free tier (https://aistudio.google.com/app/apikey to get yours). You can change to a better model for better results but it will not necessarily work on the free tier.

My initial attempts at generating a protocol and doing this manually seemed promising. The protocol was essentially
1. Prompt the LLM for what personas/system prompts would be best suited to answering the question asked
2. Prompt the LLM for what the response should look like (what general topics would be of interest to the user), what format it should be in, and which personas should handle that section/topic.
3. Prompt the LLM with the persona and a request for it to generate that subsection
4. Prompt the LLM with the subsections and ask it to review and make only minor edits, or reject it if it needed major edits (and if so back to step 3)

This had subjectively good results, with more in-depth and useful answers. So I moved on to trying to measure the wisdom of the output. In https://github.com/monadsandtuples/wisecouncil/blob/main/notunwise.md I explore how we measure wisdom and come to the conclusion we have no good general measures for wisdom.