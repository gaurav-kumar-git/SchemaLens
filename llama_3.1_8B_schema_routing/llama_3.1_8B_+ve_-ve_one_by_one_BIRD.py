import os
import json
import re
import random
import time
import gc
import traceback
from collections import defaultdict
import pdb

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" 
BIRD_DIR = '/scratch/gaurav/data/BIRD'
EXPERIMENT_RUN_NAME = "llama-3.1-8B-instruct_BIRD_one_by_one_no_examples"
EXPERIMENT_PROJECT_DIR = os.path.join('/scratch/gaurav/report_experiments', EXPERIMENT_RUN_NAME)
os.makedirs(EXPERIMENT_PROJECT_DIR, exist_ok=True)
RESULTS_FILE_PATH = os.path.join(EXPERIMENT_PROJECT_DIR, f"{EXPERIMENT_RUN_NAME}_results.json")
# PROMPT_LOG_DIR = os.path.join(EXPERIMENT_PROJECT_DIR, "prompt_logs")
EVAL_SUMMARY_PATH = os.path.join(EXPERIMENT_PROJECT_DIR, f"{EXPERIMENT_RUN_NAME}_evaluation_summary.json")
# os.makedirs(PROMPT_LOG_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an expert system that determines if a natural language question can be answered using ONLY the provided database schema.
Your task is to respond with a single character: '1' if the question is answerable, or '0' if it is not.
Do not provide any explanations or other text. Just '1' or '0'."""

FEW_SHOT_EXAMPLE_CLASSIFICATION = """# --- Example ---
[Schema:
CREATE TABLE singer (
  Singer_ID INTEGER PRIMARY KEY,
  Name TEXT,
  Country TEXT,
  Song_Name TEXT,
  Launch_Date DATETIME,
  Is_Male BOOLEAN
);
]

# Task: Can the question below be answered using the schema provided above? Respond with 1 (Yes) or 0 (No).
Q: How many singers are from France?
A: 1
# --- End of Example ---"""

def load_model_and_tokenizer(model_path):
    """Loads the model and tokenizer from the specified path."""
    print(f"--- Loading Model and Tokenizer from {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"\nModel loaded successfully on device: {model.device}\n")
    return model, tokenizer

def load_bird_data(bird_dir):
    """Loads all necessary BIRD dataset files."""
    print("--- Loading BIRD Dataset ---")
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    dev_data = load_json(os.path.join(bird_dir, 'dev', 'dev.json'))
    train_data = load_json(os.path.join(bird_dir, 'train', 'train.json'))
    tables_data = load_json(os.path.join(bird_dir, 'dev', 'dev_tables.json')) + \
                  load_json(os.path.join(bird_dir, 'train', 'train_tables.json'))
    all_schemas = {db['db_id']: db for db in tables_data}
    print(f"Loaded {len(dev_data)} dev queries, {len(train_data)} train queries, and {len(all_schemas)} schemas.\n")
    return dev_data, train_data, all_schemas

def create_examples_and_test_set(train_data, dev_data):
    """Creates a clean test set and maps of positive/negative examples."""
    print("--- Curating Positive/Negative Examples and Test Set ---")
    all_data = train_data + dev_data
    all_questions_by_db = defaultdict(list)
    for item in all_data:
        all_questions_by_db[item['db_id']].append(item)

    positive_pool_by_db = {db_id: sorted(items, key=lambda x: x['question'])[:5] for db_id, items in all_questions_by_db.items()}

    random.seed(42)
    negative_pool_by_db = defaultdict(list)
    all_db_ids = list(positive_pool_by_db.keys())
    for target_db_id in all_db_ids:
        other_db_ids = [db for db in all_db_ids if db != target_db_id]
        if not other_db_ids: continue
        sampled_db_ids = random.sample(other_db_ids, min(len(other_db_ids), 5))
        for neg_db_id in sampled_db_ids:
            if positive_pool_by_db.get(neg_db_id):
                negative_pool_by_db[target_db_id].append(random.choice(positive_pool_by_db[neg_db_id]))
    
    positive_example_map = {db_id: [ex['question'] for ex in exs] for db_id, exs in positive_pool_by_db.items()}
    negative_example_map = {db_id: [ex['question'] for ex in exs] for db_id, exs in negative_pool_by_db.items()}

    example_questions_text = {ex['question'] for examples in positive_pool_by_db.values() for ex in examples}
    test_set_queries = [item for item in dev_data if item['question'] not in example_questions_text]
    
    print(f"Curated positive examples for {len(positive_example_map)} databases.")
    print(f"Curated negative examples for {len(negative_example_map)} databases.")
    print(f"Created a clean test set with {len(test_set_queries)} queries.\n")
    return positive_example_map, negative_example_map, test_set_queries

def construct_create_table_schemas(all_db_schemas):
    """Generates CREATE TABLE strings for all schemas."""
    print("--- Generating CREATE TABLE Schema Strings ---")
    all_schemas_sql = {}
    for db_id, db_schema in tqdm(all_db_schemas.items(), desc="Generating Schemas"):
        flat_pk_list = [pk for item in db_schema.get('primary_keys', []) for pk in (item if isinstance(item, list) else [item])]
        sql_statements = []
        column_info_by_index = {i: {"name": c_name, "table_index": t_idx, "type": db_schema['column_types'][i], "table_name": db_schema['table_names_original'][t_idx]} for i, (t_idx, c_name) in enumerate(db_schema['column_names_original']) if c_name != "*"}
        for table_idx, table_name in enumerate(db_schema['table_names_original']):
            column_definitions, table_constraints = [], []
            current_table_columns = [(c_idx, c_info) for c_idx, c_info in column_info_by_index.items() if c_info['table_index'] == table_idx]
            pk_column_indices = [pk_idx for pk_idx in flat_pk_list if column_info_by_index.get(pk_idx) and column_info_by_index[pk_idx]['table_index'] == table_idx]
            for col_idx, col_info in current_table_columns:
                is_pk = col_idx in pk_column_indices
                sql_type = "INTEGER" if col_info['type'] == "number" and is_pk else "TEXT" if col_info['type'] == "text" else "REAL" if col_info['type'] == "number" else "DATETIME" if col_info['type'] == "time" else "BOOLEAN"
                col_def_str = f"  {col_info['name']} {sql_type}" + (" PRIMARY KEY" if is_pk and len(pk_column_indices) == 1 else "")
                column_definitions.append(col_def_str)
            if len(pk_column_indices) > 1:
                pk_col_names = [column_info_by_index[idx]['name'] for idx in pk_column_indices]
                table_constraints.append(f"  PRIMARY KEY ({', '.join(pk_col_names)})")
            for fk_col_idx, ref_col_idx in db_schema.get('foreign_keys', []):
                if column_info_by_index.get(fk_col_idx) and column_info_by_index[fk_col_idx]['table_index'] == table_idx:
                    fk_info, ref_info = column_info_by_index[fk_col_idx], column_info_by_index[ref_col_idx]
                    table_constraints.append(f"  FOREIGN KEY ({fk_info['name']}) REFERENCES {ref_info['table_name']}({ref_info['name']})")
            sql_statements.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(column_definitions + table_constraints) + "\n);")
        all_schemas_sql[db_id] = "\n\n".join(sql_statements)
    print(f"Generated {len(all_schemas_sql)} schema strings.\n")
    return all_schemas_sql

def build_dynamic_prompt(schema_sql, current_query_info, positive_examples, negative_examples, few_shot_example):
    """Builds the user prompt including few-shot, schema, pos/neg examples, and the target question."""
    prompt_parts = []
    
    schema_block = f"[Schema:\n{schema_sql}\n]"
    prompt_parts.append(schema_block)
    
    # if positive_examples:
    #     pos_block = "# Here are some example questions that CAN be answered by the schema above:\n" + "\n".join([f"-- {ex}" for ex in positive_examples])
    #     prompt_parts.append(f"\n{pos_block}")

    # if negative_examples:
    #     neg_block = "# Here are some example questions that CANNOT be answered by the schema above:\n" + "\n".join([f"-- {ex}" for ex in negative_examples])
    #     prompt_parts.append(f"\n{neg_block}")

    prompt_parts.append("\n\n---\n")
    # prompt_parts.append(f"{few_shot_example}")
    task_block = f"\n\n# Task: Can the question below be answered using the schema provided above? Respond with 1 or 0.\nQ: {current_query_info['question']}\nA:"
    prompt_parts.append(task_block)
    
    return "".join(prompt_parts)

def get_one_zero_token_ids(tokenizer):
    """Finds the token IDs for '1' and '0'."""
    one_token_id = tokenizer.encode(" 1", add_special_tokens=False)
    zero_token_id = tokenizer.encode(" 0", add_special_tokens=False)
    if len(one_token_id) == 1 and len(zero_token_id) == 1: return one_token_id[0], zero_token_id[0]
    one_token_id_no_space = tokenizer.encode("1", add_special_tokens=False)
    zero_token_id_no_space = tokenizer.encode("0", add_special_tokens=False)
    if len(one_token_id_no_space) == 1 and len(zero_token_id_no_space) == 1: return one_token_id_no_space[0], zero_token_id_no_space[0]
    raise ValueError("Unstable tokenization for '1'/'0'.")

def get_schema_match_prediction(model, tokenizer, system_prompt, user_prompt_content, one_token_id, zero_token_id, query_id_for_log, db_id_for_log):#, log_save_dir):
    """Gets the model's P(Yes) score for a single question-schema pair."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}]
    prompt_for_model = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt_for_model, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - 10).to(model.device)
    # pdb.set_trace()
    with torch.no_grad():
        outputs = model(**inputs)
        # pdb.set_trace()
        logits = outputs.logits[:, -1, :]
        logit_one = logits[:, one_token_id].item()
        logit_zero = logits[:, zero_token_id].item()

    max_logit = max(logit_one, logit_zero)
    exp_one = torch.exp(torch.tensor(logit_one - max_logit))
    exp_zero = torch.exp(torch.tensor(logit_zero - max_logit))
    prob_one = exp_one / (exp_one + exp_zero)
    return prob_one.item()

def run_evaluation(results_file):
    """Calculates and prints recall and timing metrics."""
    print(f"\n--- Evaluating results from: {results_file} ---")
    with open(results_file, 'r') as f: results = json.load(f)
    k_values = [1, 3, 5, 10]
    recall_counts = {k: 0 for k in k_values}
    num_queries = len(results)
    if num_queries == 0: return
    for res in results:
        true_db_id = res['true_db_id']
        ranked_predicted_dbs = [item['candidate_db_id'] for item in res['ranked_databases_with_predictions']]
        for k in k_values:
            if true_db_id in ranked_predicted_dbs[:k]: recall_counts[k] += 1
    recall_scores = {k: (c / num_queries) * 100 for k, c in recall_counts.items()}
    runtimes = [res['time_taken_seconds'] for res in results if 'time_taken_seconds' in res]
    avg_runtime = np.mean(runtimes) if runtimes else 0
    print("\n" + "="*40 + "\n      EVALUATION SUMMARY\n" + "="*40)
    print(f"\nEvaluated on {num_queries} queries.")
    for k, r in recall_scores.items(): print(f"  Recall@{k}: {r:.2f}%")
    print(f"\nAverage time per query: {avg_runtime:.2f} seconds")
    summary_data = {"num_queries_evaluated": num_queries, "recall_scores_percent": recall_scores, "average_runtime_seconds": avg_runtime}
    with open(EVAL_SUMMARY_PATH, 'w') as f: json.dump(summary_data, f, indent=2)
    print(f"\nEvaluation summary saved to: {EVAL_SUMMARY_PATH}\n" + "="*40)

def main():
    """Orchestrates the entire BIRD retrieval experiment."""
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    dev_data, train_data, all_schemas = load_bird_data(BIRD_DIR)
    
    positive_map, negative_map, test_set_queries = create_examples_and_test_set(train_data, dev_data)
    all_schemas_sql = construct_create_table_schemas(all_schemas)
    
    one_token_id, zero_token_id = get_one_zero_token_ids(tokenizer)
    print(f"Using ONE_TOKEN_ID: {one_token_id} and ZERO_TOKEN_ID: {zero_token_id}\n")
    
    experiment_results = []
    if os.path.exists(RESULTS_FILE_PATH):
        with open(RESULTS_FILE_PATH, 'r') as f: experiment_results = json.load(f)
    completed_question_ids = {res['question_id'] for res in experiment_results}
    
    print(f"--- Starting Inference on {len(test_set_queries)} BIRD queries ({len(completed_question_ids)} completed) ---")
    
    for query_info in tqdm(test_set_queries, desc="Processing Test Queries"):
        question_id = query_info['question_id']
        if question_id in completed_question_ids: continue
        
        true_db_id = query_info['db_id']
        query_start_time = time.time()
        predictions_for_current_query = []

        try:
            for candidate_db_id, candidate_schema_sql in tqdm(all_schemas_sql.items(), desc=f"  DBs for Q on {true_db_id}", leave=False):
                pos_examples = positive_map.get(candidate_db_id, [])
                neg_examples = negative_map.get(candidate_db_id, [])
                
                user_prompt = build_dynamic_prompt(
                    candidate_schema_sql, query_info, pos_examples, neg_examples, FEW_SHOT_EXAMPLE_CLASSIFICATION
                )

                p_one_score = get_schema_match_prediction(
                    model, tokenizer, SYSTEM_PROMPT, user_prompt,
                    one_token_id, zero_token_id, f"BIRD_q{question_id}",
                    candidate_db_id) #, PROMPT_LOG_DIR
                
                predictions_for_current_query.append({'candidate_db_id': candidate_db_id, 'p_one_score': p_one_score})

            query_duration_s = time.time() - query_start_time
            ranked_dbs = sorted(predictions_for_current_query, key=lambda x: x['p_one_score'], reverse=True)
            
            result = {
                'question_id': question_id,
                'nl_query_text': query_info['question'],
                'true_db_id': true_db_id,
                'ranked_databases_with_predictions': ranked_dbs,
                'time_taken_seconds': round(query_duration_s, 3)
            }
            experiment_results.append(result)

        except Exception as e:
            print(f"FATAL ERROR processing query ID {question_id}: '{query_info['question']}'. Error: {e}")
            traceback.print_exc()
        finally:
            with open(RESULTS_FILE_PATH, 'w') as f_out: json.dump(experiment_results, f_out, indent=2)
            gc.collect(); torch.cuda.empty_cache()

    print(f"\n--- Inference Complete. Results saved to {RESULTS_FILE_PATH} ---")
    run_evaluation(RESULTS_FILE_PATH)

if __name__ == "__main__":
    main()