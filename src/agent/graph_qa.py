"""LangGraph literature screening agent.

Screens literature titles and abstracts based on exclusion criteria.
"""

from __future__ import annotations

import os
import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph
from src.utils import get_paper_collection
from pydantic import BaseModel, Field

class Configuration(TypedDict):
    """Configurable parameters for the agent."""

    model_name: str
    temperature: float


@dataclass
class LiteratureItem:
    """Represents a literature item with title and abstract."""

    title: str
    doi: str
    abstract: str
    fulltext: str = ""  # Placeholder for full text if needed

@dataclass
class QAResult:
    """Structured output for literature quality assessment."""

    title: str
    doi: str
    methodological_rigor: int
    clarity_research_design: int
    validation_methods: int
    reasoning: str

@dataclass
class QAResultGptOss:
    """Structured output for literature quality assessment."""
    title: str
    doi: str
    response: str

class QADecision(BaseModel):
    """Structured output for literature quality assessment decision."""

    methodological_rigor: float = Field(description="Score for criterion methodological rigor (3: Highest, 2: Moderate, 1: Lowest)")
    clarity_research_design: float = Field(description="Score for criterion clarity of research design (3: Highest, 2: Moderate, 1: Lowest)")
    validation_methods: float = Field(description="Score for criterion validation methods (3: Highest, 2: Moderate, 1: Lowest)")
    reasoning: str = Field(description="Brief reasoning for the scores")

@dataclass
class State:
    """State for the literature screening agent."""

    qa_criteria: str = ""
    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: List[LiteratureItem] = field(default_factory=list)
    results: List[QAResult | QAResultGptOss] = field(default_factory=list)
    output_path: str = "qa_results.csv"


async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from the literature folder."""
    literature_items = []
    if not state.literature_items:
        paper_collection = get_paper_collection(collection_key=state.collection_key, get_fulltext=True)
    else:
        # assume literature items in correct format
        paper_collection = state.literature_items

    for idx, paper in paper_collection.iterrows():
        literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote, doi =paper.DOI, fulltext=paper.fulltext))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items}

async def qa_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """QA literature item based on quality assessment criteria, with up to 3 retries on error."""

    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "gpt-oss:120b")

    llm_agent = ChatOllama(model=model_name, **configuration)
    llm_agent = llm_agent.with_structured_output(QADecision)
    parser = JsonOutputParser()

    system_prompt = """You are a professional literature reviewer for high ranking scientific journals in the field of computer science and engineering. Perform a quality assessment of the given paper against the quality criteria given by the user.
    ALWAYS return your answer as a JSON with attributes 'reasoning' (brief explanation, addressing **ALL** the quality criteria) and a choice for scoring of each quality criteria according to the following Likert Scale: [Highest: 2, Moderate: 1, Lowest: 0]."""

    results = []

    for item in state.literature_items:
        attempt = 0
        while attempt < 3:
            try:
                human_prompt = f"""
    Quality Assessment Criteria: {state.qa_criteria}

    Title: {item.title}

    Fulltext: {item.fulltext}

    Which scoring should this paper receive in regards to each quality criterion?
    """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]

                response = await llm_agent.ainvoke(messages)

                result = QAResult(
                    title=item.title,
                    doi=item.doi,
                    reasoning=response.reasoning,
                    methodological_rigor=response.methodological_rigor,
                    clarity_research_design=response.clarity_research_design,
                    validation_methods=response.validation_methods
                )
                results.append(result)
                break  # Success, exit retry loop

            except Exception as e:
                attempt += 1
                if attempt == 3:
                    print(f"Error screening {item.title} after 3 attempts: {e}")
                    # Default to exclusion on repeated error
                    result = QAResult(
                        title=item.title,
                        doi=item.doi,
                        methodological_rigor=0,
                        clarity_research_design=0,
                        validation_methods=0,
                        reasoning=f"Error in processing after 3 attempts: {str(e)}"
                    )
                    results.append(result)

    return {"results": results}

async def qa_literature_gpt_oss(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """QA literature item based on quality assessment criteria, with up to 3 retries on error."""

    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "gpt-oss:20b")

    llm_agent = ChatOllama(model=model_name, **configuration)

    system_prompt = """You are a professional literature reviewer for high ranking scientific journals in the field of computer science and engineering. Perform a quality assessment of the given paper against the quality criteria given by the user.
    ALWAYS return your answer as a JSON with attributes 'reasoning' (brief explanation, addressing **ALL** the quality criteria) and a choice for scoring of each quality criteria according to the following Likert Scale: [Highest: 2, Moderate: 1, Lowest: 0]."""

    results = []

    for item in state.literature_items:
        attempt = 0
        while attempt < 3:
            try:
                human_prompt = f"""
    Quality Assessment Criteria: {state.qa_criteria}

    Title: {item.title}

    Fulltext: {item.fulltext}

    Which scoring should this paper receive in regards to each quality criterion? Start by addressing each quality criterion in your reasoning, then provide the scores.
    """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]

                response = await llm_agent.ainvoke(messages)

                result = QAResultGptOss(
                    title=item.title,
                    doi=item.doi,
                    response=response.content
                )

                results.append(result)
                break  # Success, exit retry loop

            except Exception as e:
                attempt += 1
                if attempt == 3:
                    print(f"Error screening {item.title} after 3 attempts: {e}")
                    result = QAResultGptOss(
                        title=item.title,
                        doi=item.doi,
                        response = f"Error in processing after 3 attempts: {str(e)}")
                    results.append(result)

    return {"results": results}

async def generate_csv(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate CSV file with screening results."""

    try:
        with open(state.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['title', 'doi', 'clarity_research_design', 'methodological_rigor', 'validation_methods','reasoning'])

            # Write results
            for result in state.results:
                writer.writerow([
                    result.title,
                    result.doi,
                    result.clarity_research_design,
                    result.methodological_rigor,
                    result.validation_methods,
                    result.reasoning])

        print(f"Results saved to {state.output_path}")

        # Calculate average score
        score_clarity = sum(r.clarity_research_design for r in state.results) / len(state.results)
        score_methodological = sum(r.methodological_rigor for r in state.results) / len(state.results)
        score_validation = sum(r.validation_methods for r in state.results) / len(state.results)
        total_average = (score_clarity + score_methodological + score_validation) / 3
        print(f"Average score for clarity of research design: {score_clarity:.2f}")
        print(f"Average score for methodological rigor: {score_methodological:.2f}")
        print(f"Average score for validation methods: {score_validation:.2f}")
        print(f"Overall average score: {total_average:.2f}")


    except Exception as e:
        print(f"Error generating CSV: {e}")

    return {}

async def generate_csv_gpt_oss(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate CSV file with screening results."""

    try:
        with open(state.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['title', 'doi', 'qa_assessment'])

            # Write results
            for result in state.results:
                writer.writerow([result.title, result.doi, result.response])

        print(f"Results saved to {state.output_path}")

    except Exception as e:
        print(f"Error generating CSV: {e}")

    return {}


# Define the graph

graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("screen_literature", qa_literature_gpt_oss)
    .add_node("generate_csv", generate_csv_gpt_oss)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "screen_literature")
    .add_edge("screen_literature", "generate_csv")
    .compile(name="Literature Screening Agent")
)
