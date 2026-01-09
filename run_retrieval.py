import os
from typing import List, Optional

from src.agent.graph_structured_retrieval_slr import State as RetrievalState
from src.agent.graph_structured_retrieval_slr import graph as graph_retrieval
from langchain_core.runnables import RunnableConfig
import asyncio
from pydantic import BaseModel, Field, create_model
from dataclasses import dataclass
from slr_data_model import AISystem, SystemArchitecture, AIAgent, OrchestrationPattern
from tqdm.asyncio import tqdm

# load environment variables from .env file
from dotenv import load_dotenv

from src.utils import get_paper_collection

load_dotenv()

from pydantic.fields import FieldInfo

def select_fields(model: type[BaseModel], include: list[str]):
    return create_model(
        f"{model.__name__}Partial",
        **{
            name: (field.annotation, FieldInfo.from_annotated_attribute(field.annotation, field))
            for name, field in model.model_fields.items()
            if name in include
        }
    )

async def run_retrieval(retrieval_schema: BaseModel, literature_item) -> BaseModel:
    """Run the literature screening process."""

    initial_state = RetrievalState(
        retrieval_form=retrieval_schema,
        literature_item=literature_item
    )

    config = RunnableConfig(
        configurable={
            "model_name": "gpt-oss:120b",
            "temperature": 0,
        }
    )

    result = await graph_retrieval.ainvoke(initial_state, config=config)
    return result

def get_doi_based_filename(doi: str, suffix: str) -> str:
    """Generate a filename based on DOI."""
    safe_doi = doi.replace('/', '_')
    return f"{safe_doi}_{suffix}.json"

@dataclass
class LiteratureItem:
    """Represents a literature item with title and abstract."""

    title: str
    doi: str
    abstract: str
    fulltext: str = ""  # Placeholder for full text if needed
    extra: str = ""

def load_literature(collection_key) -> List[LiteratureItem]:
    """Load literature items from the literature folder."""
    literature_items = []

    paper_collection = get_paper_collection(collection_key=collection_key, get_fulltext="parsed")

    for idx, paper in paper_collection.iterrows():
        # check if item was already processed
        output_filename = get_doi_based_filename(paper.DOI, "retrieval")
        if os.path.exists("outputs/" + output_filename):
            print(f"Skipping already processed paper: {paper.title}")
            continue
        else:
            literature_items.append(LiteratureItem(title=paper.title, abstract=paper.abstractNote, doi =paper.DOI, fulltext=paper.fulltext, extra=paper.extra))

    print(f"Loaded {len(literature_items)} literature items")
    return literature_items

def dump_output(title, doi, output, reasoning):
    """flatten schema and save retrieval results as JSON"""

    # encode as list when set is encountered
    import json
    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    # save output as json
    output = {
            'title': title,
            'doi': doi,
            'retrieval': output.model_dump(),
            'reasoning': reasoning
        }

    with open("outputs/" + get_doi_based_filename(doi, "retrieval"), encoding='utf-8', mode='w') as f:
        import json
        json.dump(output, f, ensure_ascii=False, indent=4, cls=SetEncoder)


    return {'status': 'output saved as JSON'}

def orchestrate_retrieval(literature_item):
    # decompose the schema for distributed retrieval
    system_architecture_part1 = select_fields(
        SystemArchitecture,
        include=["agents"]
    )

    system_architecture_part2 = select_fields(
        SystemArchitecture,
        include=["orchestration_pattern", "trigger"]
    )

    system_architecture_part3 = select_fields(
        SystemArchitecture,
        include=["human_integration"]
    )

    ai_system_part1 = select_fields(
        AISystem,
        include=["application_domain", "paradigm"]
    )

    ai_system_part2 = select_fields(
        AISystem,
        include=["validation_methods"]
    )

    ai_system_part3 = select_fields(
        AISystem,
        include=["reported_outcomes"]
    )

    # run retrieval for each part
    result = {}
    reasoning = {}
    loop = asyncio.get_event_loop()
    for part_name, part_schema in [
        ("system_architecture_part1", system_architecture_part1),
        ("system_architecture_part2", system_architecture_part2),
        ("system_architecture_part3", system_architecture_part3),
        ("ai_system_part1", ai_system_part1),
        ("ai_system_part2", ai_system_part2),
        ("ai_system_part3", ai_system_part3),
    ]:
        print(f"Running retrieval for {part_name}...")
        part_result = loop.run_until_complete(run_retrieval(part_schema, literature_item))
        result[part_name] = part_result["result"]
        reasoning[part_name] = part_result["reasoning"]
        print(f"Completed retrieval for {part_name}.")

    system_architecture = SystemArchitecture(
        agents=result["system_architecture_part1"].agents,
        orchestration_pattern=result["system_architecture_part2"].orchestration_pattern,
        trigger=result["system_architecture_part2"].trigger,
        human_integration=result["system_architecture_part3"].human_integration
    )

    ai_system = AISystem(
        system_architecture=system_architecture,
        application_domain=result["ai_system_part1"].application_domain,
        paradigm=result["ai_system_part1"].paradigm,
        validation_methods=result["ai_system_part2"].validation_methods,
        reported_outcomes=result["ai_system_part3"].reported_outcomes
    )

    dump_output(
        title=literature_item.title,
        doi=literature_item.doi,
        output=ai_system,
        reasoning=reasoning
    )


# Example usage
if __name__ == "__main__":

    literature = load_literature(collection_key="5HE7P89C")

    for item in tqdm(literature, desc="Retrieve", unit="item"):
        print(f"Processing paper: {item.title}")
        orchestrate_retrieval(item)




