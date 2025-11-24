from src.agent.graph_screening import State as ScreeningState
from src.agent.graph_screening import graph as graph_screening
from src.agent.graph_qa import State as QAState
from src.agent.graph_qa import graph as graph_qa
from langchain_core.runnables import RunnableConfig
import asyncio


# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Example usage function
async def run_exclusion_screening(exclusion_criteria: dict, topic:str, output_path: str = "screening_results.csv", literature_items=None):
    """Run the literature screening process."""

    initial_state = ScreeningState(
        exclusion_criteria=exclusion_criteria,
        topic=topic,
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

    result = await graph_screening.ainvoke(initial_state, config=config)
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

    exclusion_criteria = {
        "ec1_application_domain" : """EXCLUDE if the paper's primary application area is NOT production or logistics.
_For example, DO NOT EXCLUDE production and logistics domains such as: manufacturing operations, supply chain management, warehouse operations, production planning and scheduling, inventory management, quality control, maintenance, and transportation/distribution systems._

_For example, EXCLUDE applications such as: healthcare diagnostics, medical treatment planning, financial trading, legal document analysis, customer service chatbots, educational tutoring, creative content generation, or scientific research support (unless specifically for production/logistics domains). Finally, EXCLUDE all papers that exclusively review existing applications and do not introduce their own novel solution._"""
        "",
        "ec2_task_and_knowledge_complexity": """EXCLUDE if the LLM-based system performs ONLY simple tasks without requiring domain expertise.

_For example, DO NOT EXCLUDE papers where the system supports at least ONE of the following knowledge-intensive tasks:_

- Planning (production schedules, resource allocation)
    
- Decision-making (operational decisions requiring trade-off analysis)
    
- Scheduling and optimization (complex constraint satisfaction)
    
- Diagnostic reasoning (root cause analysis, troubleshooting)
    
- Tasks requiring domain knowledge typically acquired through formal engineering, operations management, or logistics training""",
        "ec3_system_architecture" : """
        EXCLUDE if the system is NOT based on generative AI and does not include at least any of the following aspects presented in the following examples:

_For example, DO NOT EXCLUDE papers with evidence of:_

- **Tool use**: LLM actively invokes external tools, APIs, simulations, or computational modules (beyond simple retrieval)
    
- **Autonomous decision-making**: System makes decisions or takes actions with either minimal per-step human intervention or in an expert in the loop manner, including multi-step reasoning or workflow orchestration.
    
- **Context engineering**: Advanced knowledge extraction and processing techniques including retrieval-augmented generation (RAG), dynamic memory management, multi-source knowledge integration, or adaptive prompt engineering

- **Agent orchestration: Orchestrating multiple LLM-based agents either utilizing collaborative mechanisms or through hierarchical structures (manager agents with sub-agents for specialized tasks)

_For example, EXCLUDE systems that are:_

- Systems without any generative AI components
    
- Conversational or reporting systems targeting information retrieval from a knowledge database WITHOUT any further deduction or reasoning over the retrieved information for solving downstream tasks.
    
- Systems without any adaptive context management or knowledge extraction mechanisms (such as efficient chunking strategies, pre-filtering using metadata, knowledge graphs or retrieval optimization methods)

        """
    }

    # load existing literature items
    #literature_items = pd.read_csv("existing_literature_items.csv")

    asyncio.run(run_exclusion_screening(
        exclusion_criteria=exclusion_criteria,
        output_path="screening_result.csv",
        topic=topic
    ))