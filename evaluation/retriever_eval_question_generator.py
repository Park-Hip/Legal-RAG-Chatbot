import mlflow
import pandas as pd
import time
import pickle
import numpy as np
import json
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pathlib import Path


# Mlflow set up
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('generate_eval_questions_for_retriever')
print('Setting up Mlflow')

# Creating evaluation data
llm = ChatOllama(model='deepseek-r1',
                 reasoning=False,
                 temperature=0.1,
                 format= 'json'
)
print('LLM loaded sucessfully.')
prompt = ChatPromptTemplate.from_template(
    '''
You are a legal question generator. Based on the legal article below, generate three questions in valid JSON format.

### Instruction:
You must generate the following three types of questions:

1. **Factual Question**: Ask about objective facts clearly stated in the article.  
    Example: *What responsibilities do police authorities have in preventing crimes?*

2. **Scenario-Based Question**: Pose a real-life situation and ask how the law applies.  
    Example: *If a company doesn’t educate employees about the law, could it be held responsible?*

3. **Advice-Seeking Question**: Ask for guidance someone might seek based on the law.  
    Example: *As a citizen, what can I do to help prevent crimes?*

### Response Format:
Return your output in **valid JSON**, like this:
{{
  "factual": "FACTUAL QUESTION HERE",
  "scenario_based": "SCENARIO-BASED QUESTION HERE",
  "advice_seeking": "ADVICE-SEEKING QUESTION HERE"
}}

---

Important Constraints:
- Do **NOT** mention or refer to article numbers (e.g., "Article 4").
- Do **NOT** include phrases like “according to the article” or “as stated above”.
- Your questions must sound natural and standalone.

Incorrect: "According to Article 4, what should citizens do?"
Correct: "What should citizens do to prevent crimes?"

If you include such references, your response will be considered **invalid**.

<|EOT|>

### Instruction:
Generate the three questions as described above using the following context. Your output must follow the format and constraints.

Context:
{context}

### Response:
'''
)

parser = JsonOutputParser()

chain = prompt | llm | parser

first_run_context = '''
Article 4. Responsibility for prevention and fight against crimes
1. Police authorities, the People’s Procuracies, People’s Courts, and other authorities concerned
shall perform their functions, duties and authority; provide guidance and assistance for other
state agencies, organizations and individuals in prevention and fight against crimes, supervision
and education of criminals in the community.
2. Organizations are responsible for raising the awareness of people under their management of
protection and compliance with the law, respect for socialism rules; promptly take measures for
eliminate causes and conditions of crimes within their organizations.
3. Every citizen has the duty to participate in prevention and fight against crimes.
'''

def token_count(result: dict) -> int:
    return sum(len(v) for v in result.values())


def create_retriever_dataset(first_run=False):
    if first_run: # Creating first run to warm up the model for following invokes
        print('First run...')
        with mlflow.start_run():
            mlflow.log_params({
                'model': 'deepseek-r1',
                'provider': 'ollama',
                'context': first_run_context,
                'temperature': 0.1,
            })
            start_time = time.time()
            result = chain.invoke({'context': first_run_context})
            latency = np.round(time.time() - start_time, decimals=2)
            try:
                json_str = json.dumps(result)
                json.loads(json_str)
                print('Output is valid JSON')
                mlflow.log_dict(result, 'generated_question.json')
                mlflow.log_metric('token_count', token_count(result))
                mlflow.log_metric('latency', latency)
                print('Latency:', latency)
                print(result)
            except json.JSONDecodeError as e:
                print('Output is not valid JSON')
                print('Error', e)
                mlflow.log_metric('token_count', token_count(result))
                mlflow.log_metric('latency', latency)
                print('Latency:', latency)
                print(result)
                print('Raw output:', result)

    else:
        question_dict= {}
        error_dict = {}
        factual_questions = []
        scenario_based_questions = []
        advice_seeking_questions = []
        doc_ids = []
        ground_truth_docs = []
        error_encounter = 0

        base_path = Path(__file__).resolve().parent.parent
        doc_path = base_path / 'data' / 'processed_data' / 'criminal_code_of_vietnam.pkl'
        with open(doc_path, 'rb') as f:
            doc_list = pickle.load(f)

        question_path = base_path / 'evaluation' / 'datasets' / 'json' / 'generated_questions.json'
        if not question_path.exists():
            with open(question_path, 'w') as f:
                json.dump(question_dict, f)

        error_path = base_path / 'evaluation' / 'datasets' / 'json' / 'errors.json'
        if not error_path.exists():
            with open(error_path, 'w') as f:
                json.dump(error_dict, f)

        for i, doc in enumerate(doc_list):
            print(f'Creating questions for article {i + 1}/{len(doc_list)}')
            context = doc.page_content
            article_id = doc.metadata['id']
            with mlflow.start_run(run_name=f"article_{article_id}"):
                mlflow.log_params({
                    'model': 'deepseek-r1',
                    'provider': 'ollama',
                    'context': context,
                    'temperature': 0.1,
                    'article_id': article_id,
                })

                start_time = time.time()
                result = chain.invoke({'context': context})
                latency = np.round(time.time() - start_time, decimals=2)

                try:
                    json_str = json.dumps(result)
                    json.loads(json_str)
                    mlflow.log_dict(result, 'generated_question.json')

                    factual_questions.append(result['factual'])
                    scenario_based_questions.append(result['scenario_based'])
                    advice_seeking_questions.append(result['advice_seeking'])
                    doc_ids.append(article_id)
                    ground_truth_docs.append(context)
                    new_questions = {
                        'factual': result['factual'],
                        'scenario_based': result['scenario_based'],
                        'advice_seeking': result['advice_seeking'],
                    }
                    with open(question_path, 'r') as f:
                        data = json.load(f)

                    data[f'{article_id}'] = new_questions

                    with open(question_path, 'w') as f:
                        json.dump(data, f ,indent=2)

                    mlflow.set_tag("status", "success")

                except json.JSONDecodeError as e:
                    print(f'Output {i} is not valid JSON')
                    print('Error', e)
                    error_encounter += 1
                    mlflow.set_tag('status', 'invalid_json')

                    new_errors = {
                        'error': str(e),
                        'id': article_id,
                        'article': doc.metadata['article']
                    }
                    with open(error_path, 'r') as f:
                        errors= json.load(f)

                    errors[f'{article_id}'] = new_errors
                    with open(error_path, 'w') as f:
                        json.dump(errors, f, indent=2)

                    print('Raw output:', result)
                    continue

                except KeyError as e:
                    error_encounter += 1
                    print(f'Article {article_id} - Missing key.')
                    mlflow.set_tag('status', 'missing_keys')
                    mlflow.log_param('error', str(e))

                    new_errors = {
                        'error': str(e),
                        'id': article_id,
                        'article': doc.metadata['article']
                    }
                    with open(error_path, 'r') as f:
                        errors = json.load(f)

                    errors[f'{article_id}'] = new_errors
                    with open(error_path, 'w') as f:
                        json.dump(errors, f, indent=2)

                    print('Error:', e)
                    continue

                except Exception as e:
                    error_encounter += 1
                    print(f'Article {article_id} - Unexpected error.')
                    mlflow.set_tag('status', 'unknown_error')
                    mlflow.log_param('error', str(e))

                    new_errors = {
                        'error': str(e),
                        'id': article_id,
                        'article': doc.metadata['article']
                    }
                    with open(error_path, 'r') as f:
                        errors = json.load(f)

                    errors[f'{article_id}'] = new_errors
                    with open(error_path, 'w') as f:
                        json.dump(errors, f, indent=2)

                    print('Error:', e)
                    continue

                finally:
                    try:
                        mlflow.log_metric('token_count', token_count(result))
                    except Exception as e:
                        print("Token count logging failed:", e)

                    mlflow.log_metric('latency', latency)
                    print('Latency:', latency)
                    print('-------------------------------------------------')
        print(f'Invalid JSON: {error_encounter}')

    combined_questions = []
    combined_doc_ids = []

    for i in range(len(factual_questions)):
        combined_questions.append(factual_questions[i])
        combined_questions.append(scenario_based_questions[i])
        combined_questions.append(advice_seeking_questions[i])

        combined_doc_ids.append(doc_ids[i])
        combined_doc_ids.append(doc_ids[i])
        combined_doc_ids.append(doc_ids[i])

    full_dataset = pd.DataFrame({
        'factual_questions': factual_questions,
        'scenario_based_questions': scenario_based_questions,
        'advice_seeking_questions': advice_seeking_questions,
        'ground_truth_doc_ids': doc_ids,
        'ground_truth_docs': [doc.page_content for doc in doc_list],
    })

    eval_dataset = pd.DataFrame({
        'questions': combined_questions,
        'ground_truth_doc_ids': combined_doc_ids,
    })
    eval_dataset['ground_truth_doc_ids'] = eval_dataset['ground_truth_doc_ids'].apply(lambda x: f"['{str(x)}']")

    dataset_dir = base_path / 'evaluation' / 'datasets' / 'dataframe'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    full_dataset.to_csv(dataset_dir / 'full_dataset.csv', index=False)
    eval_dataset.to_csv(dataset_dir / 'eval_dataset.csv', index=False)

    return full_dataset, eval_dataset


full_dataset, eval_dataset = create_retriever_dataset(first_run=False)













