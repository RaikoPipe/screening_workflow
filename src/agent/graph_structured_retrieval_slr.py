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

from src.utils.fulltext_manipulation import omit_sections_markdown
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
    literature_item: LiteratureItem
    result: Optional[BaseModel] = field(default_factory=dict)


async def retrieve(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve with automatic validation retry."""

    configuration = config.get("configurable", {})

    llm = ChatOllama(model=configuration["model_name"], temperature=configuration["temperature"], max_retries=3)

    # Chain with retry logic
    llm = llm.with_structured_output(
        state.retrieval_form
    )

    result = None

    omit_titles = [
    # Definitively omit
    "abstract",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "author contributions",
    "funding",
    "conflicts of interest",
    "conflict of interest",
    "appendix",
    "appendices",
    "supplementary material",
    "supplementary materials",

    # Introduction and context
    "introduction",
    "background",
    "motivation",
    "problem statement",

    # Literature and related work
    "literature review",
    "related work",
    "related works",
    "prior work",
    "previous work",
    "state of the art",
    "state-of-the-art",
    "theoretical background",
    "theoretical framework",

    # Conclusions and future directions
    "conclusion",
    "conclusions",
    "future work",
    "future directions",
    "future research",
    "outlook",
    "limitations and future work",

    # Discussion (depending on content)
    "discussion",
    "discussion and implications",
    "implications",

    # Availability statements
    "data availability",
    "code availability",
    "availability of data and materials",

    # Ethical statements
    "Declaration of competing interest",
    "Conflict of interest statement",
]

    text_to_screen = omit_sections_markdown(state.literature_item.fulltext, omit_sections=omit_titles)
    try:
        if state.literature_item.extra == "skip":
            raise Exception("Skipped paper")

        # Build prompt with schema and previous errors
        human_prompt = f"""Please extract the given information in the following scientific paper according to the provided schema:
# Title: {state.literature_item.title}

# Fulltext: {text_to_screen}

# Your Objective:
Please extract the given information based on the provided schema.
"""
    # todo: add reasoning with schema in prompt or not?
        messages = [
            SystemMessage(content=RETRIEVAL_PROMPT),
            HumanMessage(content=human_prompt),
        ]

        response = await llm.ainvoke(messages)

        if hasattr(response, "parsing_error") and response["parsing_error"] is not None:
            raise response["parsing_error"]

        # If include_raw=True, response is dict with 'parsed' and 'raw'
        if isinstance(response, dict):
            result = response['parsed']
        else:
            result = response

    except Exception as e:
        traceback.print_exception(e)

    return {"result": result}


# Define the graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("retrieve", retrieve)
    .add_edge("__start__", "retrieve")
    .compile(name="Literature Screening Agent SLR")
)

