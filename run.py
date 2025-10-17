
from src.agent import graph_screening
from src.agent import graph_qa
from src.agent.graph_screening import State as ScreeningState
from src.agent.graph_screening import graph as graph_screening
from src.agent.graph_qa import State as QAState
from src.agent.graph_qa import graph as graph_qa
from langchain_core.runnables import RunnableConfig
import asyncio
import os
import pandas as pd

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Example usage function
async def run_exclusion_screening(exclusion_criteria: str, output_path: str = "screening_results.csv", literature_items=None):
    """Run the literature screening process."""

    initial_state = ScreeningState(
        exclusion_criteria=exclusion_criteria,
        output_path=output_path,
        literature_items= literature_items
    )

    config = RunnableConfig(
        configurable={
            "model_name": "gpt-oss:120b",
            "temperature": 1.0,
            "max_output_tokens": 4096,
        }
    )

    result = await graph_screening.ainvoke(initial_state, config=config)
    return result

async def run_qa_screening(qa_criteria: str, output_path: str = "qa_results.csv", literature_items=None):
    """Run the literature screening process."""

    initial_state = QAState(
        qa_criteria=qa_criteria,
        output_path=output_path,
        literature_items= literature_items
    )

    config = RunnableConfig(
        configurable={
            "model_name": "gpt-oss:120b",
            "temperature": 1.0,
            "max_output_tokens": 4096,
        }
    )

    result = await graph_qa.ainvoke(initial_state, config=config)
    return result

# Example usage
if __name__ == "__main__":

    screening_criteria = """
   **EC1:**  - Exclude articles that do not focus on Large Language Models (LLMs) application and that do not include any Generative AI components.  - Exclude articles that do not explicitly mention/discuss the orchestration of multiple LLM-based agents or interaction between in them in any cooperative scheme (collaboration, cooperation, competition or similar).  - Exclude LLM applications that are monolithic or single-agent in nature.  
**EC2:**  - Exclude articles where the primary application domain is not directly related to an industrial context  - Specifically exclude studies focused solely on domains such as social media, gaming, entertainment, general education (unless specific to manufacturing education), or general healthcare (unless specific to medical device manufacturing or pharmaceutical manufacturing).  """

    qa_criteria = """
    Using Likert Scala: 1 - No, and not considered (Score: 0), 2 - Partially (Score: 1), 3 - Yes: (Score: 2)
QA1: Clarity of research design: Is there a clear description of the goals, motivations and objectives of the research with good structure and detail of methodology? 
- Studies with clear workflow, detailed methodologies, and reproducible processes were considered well-structured receive highest score (2)
- Studies with partially structured incomplete workflows receive moderate score (1)
- Studies with vague and unstructured workflows receive lowest score (0) 
QA2: Methodological rigor: Is the research approach sound? -> Categorization into fully empirical, partially empirical or theoretical
- Fully empirical: Involving experimental or applied research with robust data analysis receive highest score (2)
- Partially empirical: Some data-driven components but lacks extensive analysis receive moderate score (1)
- theoretical studies: Solely conceptual frameworks without empirical evidence receive lowest score (0)
QA3: Validation methods: Presence and depth of performance evaluation
- Robust validation: Detailed benchmarking, performance metrics, case studies, simulations receive highest score (2)
- Limited validation: Included preliminary comparisons or informal testing receive moderate score (1)
- Without validation: Receive lowest score (0)
    """

    # load existing literature items
    #literature_items = pd.read_csv("existing_literature_items.csv")

    asyncio.run(run_exclusion_screening(
        exclusion_criteria=screening_criteria,
        output_path="screening_result_snowballed.csv"
    ))