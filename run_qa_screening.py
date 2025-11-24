from src.agent.graph_qa import State as QAState
from src.agent.graph_qa import graph as graph_qa
from langchain_core.runnables import RunnableConfig
import asyncio

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
            "temperature": 0.2,
            "max_output_tokens": 16000,
        }
    )

    result = await graph_qa.ainvoke(initial_state, config=config)
    return result

# Example usage
if __name__ == "__main__":

    topic = """
    ## **RQ1: What architectural patterns and design choices characterize LLM-based agentic AI systems for knowledge-intensive planning and decision tasks in production and logistics? (2022-2025)**

### **Sub-questions:**

- RQ1a: What multi-agent orchestration patterns are employed (collaborative vs. hierarchical)?
- RQ1b: What levels of autonomy are implemented (fixed workflows vs. autonomous agents)?
- RQ1c: What role does human oversight play in these architectures?

## **RQ2: How do context engineering techniques influence the effectiveness of LLM-based agentic systems in industrial planning and decision support?**

### **Sub-questions:**

- RQ2a: Which context engineering methods are employed (tool use, memory management, retrieval strategies)?
- RQ2b: What is the reported impact of different context engineering approaches on decision quality, response time, and system reliability?
- RQ2c: How do knowledge sources and retrieval strategies affect system performance in knowledge-intensive tasks?

## **RQ3: What are the reported successes, limitations, and failure modes of LLM-based agentic AI systems in industrial production and logistics applications?**

### **Sub-questions:**

- RQ3a: In which types of planning and decision tasks do these systems demonstrate measurable benefits?
- RQ3b: What technical, operational, and organizational barriers limit their effectiveness?
- RQ3c: What failure modes are reported, and how are they attributed (context engineering vs. model limitations)?

## **RQ4: What are the open research challenges and gaps in deploying LLM-based agentic AI for knowledge-intensive industrial tasks?**

### **Sub-questions:**

- RQ4a: What aspects of human-AI collaboration (particularly expert-in-the-loop) remain underexplored?
- RQ4b: Which context engineering techniques lack empirical validation in industrial settings?
- RQ4c: What methodological gaps exist in evaluating agentic AI systems for production and logistics?

    """

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


    asyncio.run(run_qa_screening(
        qa_criteria=qa_criteria,
        output_path="qa_result.csv",
    ))

