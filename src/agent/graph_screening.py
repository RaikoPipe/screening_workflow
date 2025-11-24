"""LangGraph literature screening agent.

Screens literature titles and abstracts based on exclusion criteria.
"""

from __future__ import annotations

import math
import os
import pandas
from dataclasses import dataclass, field
from pydoc import describe
from typing import Any, Dict, List, TypedDict

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
import abc
from langgraph.graph import StateGraph
from src.utils import get_paper_collection, flatten_pydantic, remove_references_section
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from typing_extensions import Optional
from tqdm.asyncio import tqdm


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
    extra: str = ""


@dataclass
class ScreeningResult:
    """Represents screening result for a literature item."""

    title: str
    doi: str
    screening_decision: Optional[ScreeningDecision]
    exclusion: bool

class ScreeningDecision(BaseModel):
    """Structured output for literature screening decision."""

    @classmethod
    def add_exclusion_criteria(cls, **field_definitions: dict[str, str]):
        """Add fields to the model at runtime."""
        new_fields: Dict[str, FieldInfo] = {}
        new_annotations: Dict[str, Optional[type]] = {}

        for f_name, f_description in field_definitions.items():
            new_fields[f"{f_name}_reasoning"] = FieldInfo(annotation=str, description=f"Single sentence on whether the following exclusion criterion applies: {f_description}")
            new_fields[f"{f_name}_decision"] = FieldInfo(annotation=bool, description=f"True if the exclusion criterion applies, else False.")


        cls.__annotations__.update(new_annotations)
        cls.model_fields.update(new_fields)
        cls.model_rebuild(force=True)


@dataclass
class State:
    """State for the literature screening agent."""

    exclusion_criteria: dict[str,str] = None
    topic: str = ""
    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: List[LiteratureItem] = field(default_factory=list)
    results: List[ScreeningResult] = field(default_factory=list)
    output_path: str = "screening_results.csv"


async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from the literature folder."""
    literature_items = []
    if not state.literature_items:
        paper_collection = get_paper_collection(collection_key=state.collection_key, get_fulltext=True)
    else:
        # assume literature items in correct format
        paper_collection = state.literature_items

    for idx, paper in paper_collection.iterrows():
        literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote, doi =paper.DOI, fulltext=paper.fulltext, extra=paper.extra))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items}


async def screen_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Screen each literature item based on exclusion criteria, with up to 3 retries on error."""

    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "gpt-oss:120b")
    max_fulltext_words = configuration.get("max_fulltext_words", 12000)

    # add exclusion criteria fields to ScreeningDecision model
    ScreeningDecision.add_exclusion_criteria(**state.exclusion_criteria)

    llm_agent = ChatOllama(model=model_name, **configuration)
    llm_agent = llm_agent.with_structured_output(ScreeningDecision)

    system_prompt = """You are a literature screening expert. For each paper, evaluate the title and fulltext against each exclusion criterion independently. Provide clear, brief reasoning for each criterion, then make a definitive True/False decision on whether that specific criterion applies (True = exclude). Base decisions solely on the provided text."""

    results = []

    for item in tqdm(state.literature_items, desc="Screening literature", unit="item"):
        if type(item.fulltext) is str:
            text_to_screen = remove_references_section(item.fulltext)
            text_to_screen = text_to_screen.split()[:max_fulltext_words]
            text_label = "Fulltext"
        else:
            text_to_screen = item.abstract
            text_label = "Abstract"

        attempt = 0
        while attempt < 3:
            try:
                criteria_text = "\n".join([f"- {name}: {description}" for name, description in state.exclusion_criteria.items()])
                human_prompt = f"""
# Title: {item.title}

# {text_label}: {text_to_screen}

---
Here are my exclusion criteria:
{criteria_text}

Evaluate this paper against all exclusion criteria. For each criterion, provide reasoning and a decision.
"""
                if item.extra == "skip":
                    attempt = 3
                    raise Exception("Skipped paper due to 'skip' flag set in the 'extra' field.")

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]

                response = await llm_agent.ainvoke(messages)

                exclusion = [value for name, value in response if "_decision" in name]

                result = ScreeningResult(
                    title=item.title,
                    doi=item.doi,
                    screening_decision=response,
                    exclusion = any(exclusion)
                )
                results.append(result)
                break  # Success, exit retry loop

            except Exception as e:
                attempt += 1
                if attempt >= 3:
                    empty_reject_decision = ScreeningDecision(**{k: True if k.endswith('_decision') else '' for k in ScreeningDecision.model_fields})
                    print(f"Skipped item {item.title} after 3 attempts: {e}")
                    # Default to exclusion on repeated error
                    result = ScreeningResult(
                        title=item.title,
                        doi=item.doi,
                        screening_decision=empty_reject_decision,
                        exclusion=True
                    )
                    results.append(result)

    return {"results": results}

async def generate_csv(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate CSV file with screening results."""

    try:
        df = pd.DataFrame([
            {
                'title': r.title,
                'doi': r.doi,
                'exclusion': r.exclusion,
                **(r.screening_decision.model_dump() if r.screening_decision else {})
            }
            for r in state.results
        ])

        df.to_csv(state.output_path, index=False)
        print(f"Results saved to {state.output_path}")

        # Print summary
        excluded = sum(1 for r in state.results if r.exclusion == True)
        included = len(state.results) - excluded
        print(f"Summary: {included} included, {excluded} excluded out of {len(state.results)} papers")

    except Exception as e:
        print(f"Error generating CSV: {e}")

    return {}


# Define the graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("screen_literature", screen_literature)
    .add_node("generate_csv", generate_csv)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "screen_literature")
    .add_edge("screen_literature", "generate_csv")
    .compile(name="Literature Screening Agent")
)

