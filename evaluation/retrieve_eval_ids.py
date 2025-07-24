import mlflow
import pandas as pd
from embeddings.embedder import retriever_fn
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('evaluate_retriever')

base_path = Path(__file__).resolve().parent.parent
data_path = base_path / 'evaluation' / 'datasets' / 'dataframe' / 'eval_dataset.csv'

def get_retrieved_ids():
    logger.info('Getting evaluation dataset...')
    eval_df = pd.read_csv(data_path)
    eval_df.head(3)


    logger.info('Starting retrieval for all questions...')
    retrieved_ids = []
    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc='Retrieving'):
        question = row['questions']
        try:
            docs = retriever_fn([question])
            retrieved_docs = docs[0]
            context_ids = [doc.metadata.get('id') for doc in retrieved_docs if doc.metadata.get('id')]
            logger.info(f'Row {i}: Retrieved {len(context_ids)} IDs')
        except Exception as e:
            logger.error(f"Retriever failed at row {i} - {e}")
            context_ids = []
        retrieved_ids.append(context_ids)

    eval_df['retrieved_id'] = retrieved_ids
    logger.info("Finished adding 'retrieved_id' column.")
    eval_ds_with_retrieved_ids_path = base_path/'evaluation'/'datasets'/'dataframe'/'eval_dataset_with_retrieved_ids.csv'
    eval_df.to_csv(eval_ds_with_retrieved_ids_path, index=False)

if __name__ == '__main__':
    get_retrieved_ids()


