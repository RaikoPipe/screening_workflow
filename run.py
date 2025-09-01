from agent.graph import State, RunnableConfig, graph
import asyncio
import os

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Example usage function
async def run_screening(exclusion_criteria: str, output_path: str = "screening_results.csv"):
    """Run the literature screening process."""

    initial_state = State(
        exclusion_criteria=exclusion_criteria,
        output_path=output_path
    )

    config = RunnableConfig(
        configurable={
            "model_name": "gemini-2.5-flash",
            "temperature": 1.0,
            "max_output_tokens": 1024,
            "api_key": os.environ.get("GOOGLE_API_KEY", "")
        }
    )

    result = await graph.ainvoke(initial_state, config=config)
    return result


# Example usage
if __name__ == "__main__":
    exclusion_criteria = """Exclusion criteria:
- EC1 Non LLM-based multi-agent systems: 
	- traditional optimization methods or machine learning without generative AI elements
	- Use of monolithical LLM without multi-agent orchestration elements
- EC2 Manufacturing context: Domain irrelevant to topic of manufacturing
	- focuses solely on: social media, gaming, entertainment, education (unless manufacturing education, healthcare (unless medical device manufacturing)
- EC3: Scope limitation
	- purely theoretical work without any application context
	- general AI surves without specific focus on manufacturing applications
"""

    asyncio.run(run_screening(
        exclusion_criteria=exclusion_criteria,
        output_path="literature_screening_results.csv"
    ))