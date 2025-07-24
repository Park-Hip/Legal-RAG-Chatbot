from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
import logging
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
)
logger = logging.getLogger(__name__)

def get_vectorstore() -> QdrantVectorStore:
    base_dir = Path(__file__).resolve().parent.parent
    doc_path = base_dir / 'data' / 'processed_data' / 'criminal_code_of_vietnam.pkl'

    with open(doc_path, 'rb') as f:
        doc_list = pickle.load(f)

    url = 'http://localhost:6333'
    collection_name = 'legal_db'
    client = QdrantClient(url=url)

    model_name = 'BAAI/bge-large-en'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    logger.info('Embedding created.')
    dummy_embedding = embeddings.embed_query('A dummy to test embedding dimension')
    vector_dim = len(dummy_embedding)
    logger.info(f'Detected embedding dimension: {vector_dim}')

    vectors_config = models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)

    if collection_name in [c.name for c in client.get_collections().collections]:
        logger.info('Collection exists. Connecting...')
        collection_info = client.get_collection(collection_name)
        existing_dim = None
        if hasattr(collection_info.config, 'vectors') and hasattr(collection_info.config.vectors, 'size'):
            existing_dim = collection_info.config.vectors.size
        elif hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'vectors') and hasattr(collection_info.config.params.vectors, 'size'):
            existing_dim = collection_info.config.params.vectors.size

        if existing_dim != vector_dim:
            logger.warning(
                f'Existing collection {collection_name} has dimension {existing_dim} but model expects dimension'
                f' {vector_dim}'
            )

        db = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            prefer_grpc=False,
            url=url,
        )
    else:
        logger.info(f'Collection "{collection_name}" does not exist. Creating new collection...')
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,

        )
        db = QdrantVectorStore.from_documents(
            documents=doc_list,
            embedding=embeddings,
            url=url,
            prefer_grpc=False,
            collection_name=collection_name,
        )
        logger.info('Qdrant Index created.')

    return db

def retriever_fn(queries, db=None):
    if db is None:
        db = get_vectorstore()
    return [db.similarity_search(q) for q in queries]

def test_query():
    db = get_vectorstore()
    query = "What happens if an old man commits a crime?"
    docs = db.similarity_search_with_score(query=query, k=3)

    logger.info('-' * 60)
    for doc, score in docs:
        logger.info(f"Score: {score}")
        logger.info(f"Content: {doc.page_content}")
        logger.info(f"Metadata: {doc.metadata}")
    logger.info('-' * 60)

    metadata_filtered_docs = db.similarity_search_with_score(
        query=query,
        k=1,
        filter=models.Filter(
            should=[
                models.FieldCondition(
                    key='metadata.chapter',
                    match=models.MatchValue(
                        value='Chapter VIII DECISION ON SENTENCES'
                    )
                )
            ]
        )
    )
    if not metadata_filtered_docs:
        logger.warning('No document matched the filter')
    else:
        for doc, score in metadata_filtered_docs:
            logger.info(f'*{doc.page_content}[{doc.metadata}]')

if __name__ == "__main__":
    test_query()
