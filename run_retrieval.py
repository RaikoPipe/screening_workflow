from src.agent.graph_systematic_retrieval import State as RetrievalState
from src.agent.graph_systematic_retrieval import graph as graph_retrieval
from langchain_core.runnables import RunnableConfig
import asyncio
from pydantic import BaseModel

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

async def run_retrieval(retrieval_schema: BaseModel, literature_items=None):
    """Run the literature screening process."""

    initial_state = RetrievalState(
        retrieval_form=retrieval_schema,
    )

    config = RunnableConfig(
        configurable={
            "model_name": "claude-sonnet-4-5-20250929",
            "temperature": 0,
        }
    )

    result = await graph_retrieval.ainvoke(initial_state, config=config)
    return result

# Example usage
if __name__ == "__main__":

    from slr_data_model import AISystem

    asyncio.run(run_retrieval(
        retrieval_schema=AISystem,
    ))

