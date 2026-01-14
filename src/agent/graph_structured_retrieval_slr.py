from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ValidationError
from typing_extensions import Optional
from loguru import logger

from src.utils.fulltext_manipulation import omit_sections_markdown
from src.utils.prompt_utils import load_prompt

RETRIEVAL_PROMPT = load_prompt("prompts/retrieval_prompt.md")

VALIDATION_PROMPT = """You are a JSON repair assistant. Fix ONLY the specific validation errors in the JSON.

Original schema:
{schema}

Reasoning about JSON content:
{reasoning}

Current JSON with errors:
{json_content}

Validation errors:
{errors}

Return ONLY the corrected JSON with the problematic fields fixed."""

class Configuration(TypedDict):
    model_name: str
    temperature: float

@dataclass
class LiteratureItem:
    title: str
    doi: str
    abstract: str
    fulltext: str = ""
    extra: str = ""

@dataclass
class State:
    retrieval_form: Optional[BaseModel]
    literature_item: LiteratureItem
    result: Optional[BaseModel] = field(default_factory=dict)
    reasoning: Optional[str] = ""
    raw_json: Optional[str] = None
    validation_errors: Optional[str] = None
    validation_attempts: int = 0
    max_validation_attempts: int = 3
    schema_instructions: Optional[str] = None

async def generate(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate initial extraction."""
    logger.info(f"Generating extraction for paper: {state.literature_item.title}")

    if state.literature_item.extra == "skip":
        logger.info(f"Skipping paper: {state.literature_item.title}")
        return {"result": None}

    configuration = config.get("configurable", {})
    llm = ChatOllama(
        model=configuration["model_name"],
        temperature=configuration["temperature"],
        max_retries=3,
        reasoning=True,
        format="json"
    )

    parser = PydanticOutputParser(pydantic_object=state.retrieval_form)
    schema = state.retrieval_form.model_json_schema()
    schema_json = json.dumps(schema, indent=2)

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

    text_to_screen = omit_sections_markdown(
        state.literature_item.fulltext,
        omit_sections=omit_titles
    )

    if len(text_to_screen.split()) > 12000:
        logger.warning('Fulltext exceeds 12000 words')

    messages = [
        SystemMessage(content=RETRIEVAL_PROMPT.format(
            title=state.literature_item.title,
            fulltext=text_to_screen,
            schema=schema_json))
    ]

    try:
        response = await llm.ainvoke(messages)
        return {
            "raw_json": response.content,
            "reasoning": response.additional_kwargs.get("reasoning_content", ""),
            "validation_attempts": 0,
            "schema_instructions": schema_json
        }
    except ConnectionError as e:
        logger.error(f"Generation error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected generation error: {traceback.format_exc()}")
        return {"result": None}

async def validate(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Validate and parse JSON."""

    try:
        result = state.retrieval_form.model_validate_json(state.raw_json)
        return {"result": result, "validation_errors": None}

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "validation_errors": e.json(indent=2),  # or str(e.errors())
            "validation_attempts": state.validation_attempts + 1
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return {
            "validation_errors": f"JSON decode error: {str(e)}",
            "validation_attempts": state.validation_attempts + 1
        }

async def repair(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Repair validation errors in JSON."""

    configuration = config.get("configurable", {})
    llm = ChatOllama(
        model=configuration["model_name"],
        temperature=0.0,
        max_retries=2,
        format="json"
    )

    messages = [
        SystemMessage(content=VALIDATION_PROMPT.format(
            schema=state.schema_instructions,
            json_content=state.raw_json,
            errors=state.validation_errors,
            reasoning=state.reasoning
        ))
    ]

    try:
        response = await llm.ainvoke(messages)
        return {"raw_json": response.content}
    except Exception as e:
        logger.error(f"Repair error: {e}")
        return {}

async def repair_edit(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Repair validation errors by applying targeted JSON edits."""

    configuration = config.get("configurable", {})
    llm = ChatOllama(
        model=configuration["model_name"],
        temperature=0.0,
        max_retries=2,
        format="json"
    )

    # Define the edit instruction schema
    class JsonEdit(BaseModel):
        old_str: str
        new_str: str
        reason: str

    class JsonEditPlan(BaseModel):
        edits: list[JsonEdit]

    EDIT_PROMPT = """You are a JSON repair assistant. Create targeted string replacements to fix validation errors.

Reasoning about JSON content:
{reasoning}

Current JSON with errors:
{json_content}

Validation errors:
{errors}

Return a JSON with a list of 'edits', where each edit has:
- old_str: exact string to find (must appear exactly once)
- new_str: replacement string
- reason: explanation of the fix

Make minimal, precise edits targeting only the problematic fields."""

    messages = [
        SystemMessage(content=EDIT_PROMPT.format(
            json_content=state.raw_json,
            errors=state.validation_errors,
            reasoning=state.reasoning
        ))
    ]

    max_edit_iterations = 3
    broken_edits = []
    edited_json = state.raw_json

    for edit_iteration in range(max_edit_iterations):
        try:
            # Get edit plan from LLM
            response = await llm.ainvoke(messages)
            edit_plan = JsonEditPlan.model_validate_json(response.content)

            # Apply edits sequentially
            for edit in edit_plan.edits:
                try:
                    if edited_json.count(edit.old_str) == 1:
                        edited_json = edited_json.replace(edit.old_str, edit.new_str, 1)
                    elif edited_json.count(edit.old_str) > 1:
                        raise ValueError(f"Multiple occurrences: {edit.reason} for '{edit.old_str}'")
                    else:
                        raise ValueError(f"String not found: {edit.reason} for '{edit.old_str}'")
                except ValueError as ve:
                    logger.warning(str(ve))
                    broken_edits.append(edit)

            # If all edits succeeded, break
            if not broken_edits:
                break
            else:
                # Provide feedback for next iteration
                feedback = "\n".join([f"- {be.reason}: '{be.old_str}'" for be in broken_edits])
                messages.append(HumanMessage(
                    content=f"These edits failed:\n{feedback}\n\nRevise your edit plan with more specific strings."
                ))
                broken_edits = []

        except Exception as e:
            logger.error(f"Edit iteration {edit_iteration} error: {e}")
            break

    return {"raw_json": edited_json}

def should_repair(state: State) -> str:
    """Route based on validation status."""

    if state.result is not None and state.result != {}:
        return END

    if state.validation_errors is None:
        return "validate"

    if state.validation_attempts >= state.max_validation_attempts:
        logger.error(f"Max validation attempts reached: {state.literature_item.title}")
        return END

    return "repair"

# Build graph
graph = (
    StateGraph(State, context_schema=Configuration)
    .add_node("generate", generate)
    .add_node("validate", validate)
    .add_node("repair", repair_edit)
    .add_edge("__start__", "generate")
    .add_edge("generate", "validate")
    .add_conditional_edges("validate", should_repair, {
        "repair": "repair",
        "validate": "validate",
        END: END
    })
    .add_edge("repair", "validate")
    .compile(name="Literature Screening Agent SLR")
)