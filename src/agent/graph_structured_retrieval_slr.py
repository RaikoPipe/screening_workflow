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
from pydantic import BaseModel, ValidationError
from tqdm.asyncio import tqdm
from typing_extensions import Optional
import json

from src.utils.fulltext_manipulation import omit_sections_markdown
from src.utils.prompt_utils import load_prompt
from loguru import logger

from src.utils import get_paper_collection, remove_section

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
    fulltext: str = ""
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
    reasoning: Optional[str] = ""


async def retrieve(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve with manual parsing and validation retry."""

    configuration = config.get("configurable", {})

    llm = ChatOllama(
        model=configuration["model_name"],
        temperature=configuration["temperature"],
        max_retries=3,
        reasoning=True,
        format="json"
    )

    result = None
    reasoning = ""
    max_validation_retries = 3

    omit_titles = [
        "abstract", "references", "bibliography", "acknowledgments", "acknowledgements",
        "author contributions", "funding", "conflicts of interest", "conflict of interest",
        "appendix", "appendices", "supplementary material", "supplementary materials",
        "introduction", "background", "motivation", "problem statement",
        "literature review", "related work", "related works", "prior work", "previous work",
        "state of the art", "state-of-the-art", "theoretical background", "theoretical framework",
        "conclusion", "conclusions", "future work", "future directions", "future research",
        "outlook", "limitations and future work", "discussion", "discussion and implications",
        "implications", "data availability", "code availability",
        "availability of data and materials", "Declaration of competing interest",
        "Conflict of interest statement",
    ]

    text_to_screen = omit_sections_markdown(state.literature_item.fulltext, omit_sections=omit_titles)

    if state.literature_item.extra == "skip":
        logger.info(f"Skipping paper: {state.literature_item.title}")
        return {"result": None}

    # Get schema description
    schema_json = state.retrieval_form.model_json_schema()

    for attempt in range(max_validation_retries):
        try:
            # Build prompt
            human_prompt = f"""Please extract the given information in the following scientific paper according to the provided schema:

# Title: {state.literature_item.title}

# Fulltext: {text_to_screen}

# Your Objective:
Extract information based on the provided schema. Return ONLY valid JSON matching the schema.

# Schema:
{json.dumps(schema_json, indent=2)}

{"# Previous validation errors:" if attempt > 0 else ""}
{f"Attempt {attempt + 1}/{max_validation_retries} - Fix the following validation issues in your response." if attempt > 0 else ""}
"""

            messages = [
                SystemMessage(content=RETRIEVAL_PROMPT),
                HumanMessage(content=human_prompt),
            ]

            response = await llm.ainvoke(messages)

            # Extract JSON content
            content = response.content
            reasoning = response.additional_kwargs["reasoning_content"]

            # Parse and validate
            result = state.retrieval_form.model_validate_json(content)

            logger.success(f"Successfully extracted data for: {state.literature_item.title}")
            break

        except ValidationError as e:
            logger.warning(f"Validation error (attempt {attempt + 1}/{max_validation_retries}): {e}")
            if attempt == max_validation_retries - 1:
                logger.error(f"Failed to extract valid data after {max_validation_retries} attempts")
                result = None
            else:
                # Add error details to next prompt
                human_prompt += f"\n\nValidation errors from previous attempt:\n{str(e)}"

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error (attempt {attempt + 1}/{max_validation_retries}): {e}")
            if attempt == max_validation_retries - 1:
                logger.error(f"Failed to get valid JSON after {max_validation_retries} attempts")
                result = None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            traceback.print_exception(e)
            result = None
            break

    return {"result": result, "reasoning": reasoning}


# Define the graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("retrieve", retrieve)
    .add_edge("__start__", "retrieve")
    .compile(name="Literature Screening Agent SLR")
)