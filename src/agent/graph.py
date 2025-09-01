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
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph
from src.utils import get_paper_collection
from pydantic import BaseModel, Field

class Configuration(TypedDict):
    """Configurable parameters for the agent."""

    model_name: str
    temperature: float
    api_key: str


@dataclass
class LiteratureItem:
    """Represents a literature item with title and abstract."""

    title: str
    abstract: str


@dataclass
class ScreeningResult:
    """Represents screening result for a literature item."""

    title: str
    inclusion: int  # 0 or 1
    reasoning: str

class ScreeningDecision(BaseModel):
    """Structured output for literature screening decision."""

    inclusion: int = Field(description="0 for exclude, 1 for include")
    reasoning: str = Field(description="Brief reasoning for the decision")

@dataclass
class State:
    """State for the literature screening agent."""

    exclusion_criteria: str = ""
    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: List[LiteratureItem] = field(default_factory=list)
    results: List[ScreeningResult] = field(default_factory=list)
    output_path: str = "screening_results.csv"



async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from the literature folder."""

    literature_items = []

    paper_collection = get_paper_collection(collection_key=state.collection_key)

    for idx, paper in paper_collection.iterrows():
        literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items}


async def screen_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Screen each literature item based on exclusion criteria, with up to 3 retries on error."""

    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "gemini-2.5-flash")

    llm_agent = ChatVertexAI(model=model_name, **configuration).with_structured_output(ScreeningDecision)
    parser = JsonOutputParser()

    system_prompt = """You are a literature screening expert. Evaluate the title and abstract against exclusion criteria.
Return JSON with 'inclusion' (0=exclude, 1=include) and 'reasoning' (brief explanation). If there is no abstract given, evaluate based on title only."""

    results = []

    for item in state.literature_items:
        attempt = 0
        while attempt < 3:
            try:
                human_prompt = f"""
Exclusion Criteria: {state.exclusion_criteria}

Title: {item.title}

Abstract: {item.abstract}

Should this paper be INCLUDED (1) or EXCLUDED (0) based on the exclusion criteria?
"""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]

                response = await llm_agent.ainvoke(messages)

                result = ScreeningResult(
                    title=item.title,
                    inclusion=response.inclusion,
                    reasoning=response.reasoning
                )
                results.append(result)
                break  # Success, exit retry loop

            except Exception as e:
                attempt += 1
                if attempt == 3:
                    print(f"Error screening {item.title} after 3 attempts: {e}")
                    # Default to exclusion on repeated error
                    result = ScreeningResult(
                        title=item.title,
                        inclusion=0,
                        reasoning=f"Error in processing after 3 attempts: {str(e)}"
                    )
                    results.append(result)

    return {"results": results}


async def generate_csv(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate CSV file with screening results."""

    try:
        with open(state.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['title', 'inclusion', 'reasoning'])

            # Write results
            for result in state.results:
                writer.writerow([result.title, result.inclusion, result.reasoning])

        print(f"Results saved to {state.output_path}")

        # Print summary
        included = sum(1 for r in state.results if r.inclusion == 1)
        excluded = len(state.results) - included
        print(f"Summary: {included} included, {excluded} excluded out of {len(state.results)} papers")

    except Exception as e:
        print(f"Error generating CSV: {e}")

    return {}


# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("screen_literature", screen_literature)
    .add_node("generate_csv", generate_csv)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "screen_literature")
    .add_edge("screen_literature", "generate_csv")
    .compile(name="Literature Screening Agent")
)