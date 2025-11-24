from typing import List, Dict, Any, TypedDict
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
import chromadb
from langchain_community.embeddings import OllamaEmbeddings

class UnstructuredRetriever:
    def __init__(self, collection_name: str = "scientific_papers",
                 embedding_model: str = "nomic-embed-text"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    def ingest_document(self, text: str, doc_id: str,
                        max_characters: int = 1000,
                        combine_text_under_n_chars: int = 200):
        """Partition and chunk text hierarchically"""

        # Partition text into elements
        elements = partition_text(text=text)

        # Chunk by title (maintains hierarchy)
        chunks = chunk_by_title(
            elements,
            max_characters=max_characters,
            combine_text_under_n_chars=combine_text_under_n_chars
        )

        # Extract texts and metadata
        texts = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            chunk_text = str(chunk)

            # Extract metadata from element
            metadata = {
                'doc_id': doc_id,
                'chunk_idx': idx,
                'element_type': chunk.category if hasattr(chunk, 'category') else 'text',
            }

            # Add hierarchical metadata if available
            if hasattr(chunk, 'metadata'):
                elem_meta = chunk.metadata.to_dict()
                metadata.update({
                    'page_number': elem_meta.get('page_number', 0),
                    'filename': elem_meta.get('filename', doc_id),
                })

                # Hierarchy indicators
                if 'emphasized_text_contents' in elem_meta:
                    metadata['section_title'] = elem_meta['emphasized_text_contents']
                if 'parent_id' in elem_meta:
                    metadata['parent_id'] = elem_meta['parent_id']

            texts.append(chunk_text)
            metadatas.append(metadata)

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)

        # Store in ChromaDB
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )

        return len(chunks)

    def retrieve(self, query: str, n_results: int = 5,
                 filter_dict: Dict = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks"""
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )

        return [{
            'text': doc,
            'metadata': meta,
            'distance': dist
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]

# LangGraph Integration
class RetrieverState(TypedDict):
    query: str
    documents: List[Dict[str, Any]]
    paper_text: str
    doc_id: str

def retrieval_node(state: RetrieverState, retriever: UnstructuredRetriever) -> RetrieverState:
    """LangGraph node for retrieval"""
    docs = retriever.retrieve(state['query'], n_results=5)
    return {**state, 'documents': docs}