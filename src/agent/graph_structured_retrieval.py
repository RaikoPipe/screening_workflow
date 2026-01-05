"""LangGraph literature extraction pipeline.

Accepts a pydantic data model defining extraction fields, and retrieves them iteratively from a collection of papers.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from typing_extensions import Optional
from src.utils.prompt_utils import load_prompt
from loguru import logger
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from src.utils import get_paper_collection, remove_section

@tool
def extract_structured_data(text: str, schema_description: str) -> BaseModel:
    """Extract structured information from academic paper text according to a schema.

    Args:
        text: The paper text to extract from
        schema_description: Description of what fields to extract

    Returns:
        Extracted data as dictionary
    """
    # This is a placeholder - the agent will use its LLM to process
    pass

# ==================== load prompt templates ====================

RETRIEVAL_PROMPT = load_prompt("prompts/retrieval_prompt.md")

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
class SIEResult:
    """Result of a literature retrieval."""

    title: str
    doi: str
    retrieval_form: Optional[BaseModel]


@dataclass
class State:
    """State for the literature screening agent."""

    retrieval_form: Optional[BaseModel]
    collection_key: str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    literature_items: List[LiteratureItem] = field(default_factory=list)
    results: List[SIEResult] = field(default_factory=list)

async def load_literature(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load literature items from the literature folder."""
    literature_items = []
    if not state.literature_items:
        paper_collection = get_paper_collection(collection_key=state.collection_key, get_fulltext=True)
    else:
        # assume literature items in correct format
        paper_collection = state.literature_items

    for idx, paper in paper_collection.iterrows():
        # check if item was already processed
        output_filename = get_doi_based_filename(paper.DOI, "retrieval")
        if os.path.exists("outputs/" + output_filename):
            print(f"Skipping already processed paper: {paper.title}")
            continue
        else:
            literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote, doi =paper.DOI, fulltext=paper.fulltext, extra=paper.extra))

    print(f"Loaded {len(literature_items)} literature items")
    return {"literature_items": literature_items}


async def retrieve(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve with automatic validation retry."""

    configuration = config.get("configurable", {})
    model_name = configuration.get("model_name", "gpt-oss:120b")
    temperature = configuration.get("temperature", 0)

    final_config = {
        "model_name": model_name,
        "temperature": temperature,
        "max_retries": 3,
    }

    llm = ChatAnthropic(**final_config)
    parser = PydanticOutputParser(pydantic_object=state.retrieval_form)

    # Chain with retry logic
    llm = llm.with_structured_output(
        state.retrieval_form,
        method="json_schema"
    )

    results = []

    for item in tqdm(state.literature_items, desc="Retrieve", unit="item"):
        text_to_screen = remove_section(item.fulltext, section_title="References")
        try:
            if item.extra == "skip":
                raise Exception("Skipped paper")

            # Build prompt with schema and previous errors
            human_prompt = f"""Please extract the given information in the following scientific paper according to the provided schema:
# Title: {item.title}

# Fulltext: {text_to_screen}

# Your Objective:
Please extract the given information based on the provided schema.
"""

            messages = [
                SystemMessage(content=RETRIEVAL_PROMPT),
                HumanMessage(content=human_prompt),
            ]

            response = await llm.ainvoke(messages)

            if response["parsing_error"]:
                raise response["parsing_error"]

            # If include_raw=True, response is dict with 'parsed' and 'raw'
            if isinstance(response, dict):
                retrieval_result = response['parsed']
            else:
                retrieval_result = response

            # Success - validation passed
            results.append(SIEResult(
                title=item.title,
                doi=item.doi,
                retrieval_form=retrieval_result,
            ))
            break

        except Exception as e:
            traceback.print_exception(e)
            # processing failed, save empty result
            results.append(SIEResult(title=item.title, doi=item.doi, retrieval_form={}))

    return {"results": results}

def get_doi_based_filename(doi: str, suffix: str) -> str:
    """Generate a filename based on DOI."""
    safe_doi = doi.replace('/', '_')
    return f"{safe_doi}_{suffix}.json"

async def dump_output(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """flatten schema and save retrieval results as JSON"""

    # encode as list when set is encountered
    import json
    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    # save output as json
    for result in state.results:
        output = {
                'title': result.title,
                'doi': result.doi,
                'retrieval': result.retrieval_form.model_dump()
            }

        with open("outputs/" + get_doi_based_filename(result.doi, "retrieval"), encoding='utf-8') as f:
            import json
            json.dump(output, f, ensure_ascii=False, indent=4, cls=SetEncoder)


    return {'status': 'output saved as JSON'}



    # try:
    #     df = pd.DataFrame([
    #         {
    #             'title': r.title,
    #             'doi': r.doi,
    #             **flatten_pydantic(r.retrieval_form) if r.retrieval_form else {}
    #         }
    #         for r in state.results
    #     ])
    #
    #     df.to_csv(state.output_path, index=False)
    #     print(f"Results saved to {state.output_path}")
    # except Exception as e:
    #     print(f"Error generating CSV: {e}")


    # try:
    #     df = pd.DataFrame([
    #         {
    #             'title': r.title,
    #             'doi': r.doi,
    #             'exclusion': r.exclusion,
    #             **(r.screening_decision.model_dump() if r.screening_decision else {})
    #         }
    #         for r in state.results
    #     ])
    #
    #     df.to_csv(state.output_path, index=False)
    #     print(f"Results saved to {state.output_path}")
    #
    #     # Print summary
    #     excluded = sum(1 for r in state.results if r.exclusion == True)
    #     included = len(state.results) - excluded
    #     print(f"Summary: {included} included, {excluded} excluded out of {len(state.results)} papers")
    #
    # except Exception as e:
    #     print(f"Error generating CSV: {e}")
    #
    # return {}


# Define the graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("load_literature", load_literature)
    .add_node("retrieve", retrieve)
    .add_node("dump_output", dump_output)
    .add_edge("__start__", "load_literature")
    .add_edge("load_literature", "retrieve")
    .add_edge("retrieve", "dump_output")
    .compile(name="Literature Screening Agent")
)

