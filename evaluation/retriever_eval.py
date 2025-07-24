import mlflow
import logging
from pathlib import Path
from embeddings.embedder import retriever_fn
import pandas as pd
import ast
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("retriever_eval")

base_path = Path(__file__).resolve().parent.parent
ds_path = base_path / 'evaluation' / 'datasets' / 'dataframe' / 'eval_dataset_with_retrieved_ids.csv'
eval_df = pd.read_csv(ds_path)

# When save to CSV, Python converts all columns to strings, even if some looked like lists. To convert them back to list:
eval_df["ground_truth_doc_ids"] = eval_df['ground_truth_doc_ids'].apply(ast.literal_eval)
eval_df['retrieved_id'] = eval_df['retrieved_id'].apply(ast.literal_eval)

with mlflow.start_run():
    mlflow.log_param('retriever', retriever_fn.__name__)
    mlflow.log_artifact(str(ds_path))

    evaluate_results = mlflow.evaluate(
        model = None,
        data=eval_df,
        model_type='retriever',
        predictions='retrieved_id',
        targets='ground_truth_doc_ids',
        evaluators='default',
        evaluator_config={'retriever_k': 4},
        extra_metrics = [
            mlflow.metrics.precision_at_k(1),
            mlflow.metrics.precision_at_k(2),
            mlflow.metrics.precision_at_k(3),
            mlflow.metrics.recall_at_k(1),
            mlflow.metrics.recall_at_k(2),
            mlflow.metrics.recall_at_k(3),
            mlflow.metrics.ndcg_at_k(1),
            mlflow.metrics.ndcg_at_k(2),
            mlflow.metrics.ndcg_at_k(3),

        ]
    )

    logger.info("Evaluation Metrics:")
    for k, v in evaluate_results.metrics.items():
        logger.info(f"{k}: {v}")

    for metric_name in ['precision', 'recall', 'ndcg']:
        y = [evaluate_results.metrics[f'{metric_name}_at_{k}/mean'] for k in range(1,5)]
        plt.plot([1,2,3,4], y, label=f'{metric_name}@k')
    plt.title('bge-large-en Performance')
    plt.xlabel('k')
    plt.ylabel('Metric Value')
    plt.xticks([1,2,3,4])
    plt.legend()
    plt.savefig(str(base_path / 'evaluation' / 'visualizations' / 'bge_large_en_performance.png'))
    mlflow.log_artifact(str(base_path / 'evaluation' / 'visualizations' / 'bge_large_en_performance.png'))
