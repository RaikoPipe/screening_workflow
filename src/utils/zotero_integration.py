from syslira_tools import ZoteroClient, OpenAlexClient, PaperLibrary
from loguru import logger
import os

def get_paper_collection(collection_key):
    """Initialize the Zotero client."""

    zotero_api_key = os.environ.get("ZOTERO_API_KEY")
    zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID")
    zotero_library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", "user")

    logger.info("Initializing Zotero client...")
    zotero_client = ZoteroClient(zotero_api_key, zotero_library_id,
                                 library_type=zotero_library_type)
    zotero_client.init()

    logger.info("Initializing OpenAlex client...")
    openalex_client = OpenAlexClient()
    openalex_client.init()

    # Set up paper library
    paper_library = PaperLibrary(
        zotero_client=zotero_client,
        openalex_client=openalex_client,
        collection_key=collection_key,
    )

    # get papers
    result = paper_library.update_from_zotero(get_fulltext=False)
    logger.info(result)

    return paper_library.get_library_df()

