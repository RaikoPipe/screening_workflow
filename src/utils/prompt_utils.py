from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

def load_prompt(path:str) -> str:
    """Load and validate prompt from markdown file."""
    # load prompt
    prompt_loader = UnstructuredMarkdownLoader(path)

    # validate prompts as ingestible
    prompt_doc = prompt_loader.load()

    assert len(prompt_doc) == 1
    assert isinstance(prompt_doc[0], Document)

    prompt_content = prompt_doc[0].page_content

    return prompt_content