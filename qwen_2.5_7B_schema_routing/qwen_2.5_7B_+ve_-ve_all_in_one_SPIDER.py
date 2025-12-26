import os
import json
import re
import random
import time
import gc
import copy
import traceback
import zipfile
import pdb
from collections import defaultdict

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = "/scratch/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28" 
SPIDER_DIR = '/scratch/gaurav/data/spider_subset_data' 
EXPERIMENT_RUN_NAME = "qwen_2.5_7B_instruct_SPIDER_retrieval_all_in_one_pos_neg_examples" 
EXPERIMENT_PROJECT_DIR = os.path.join('/scratch/gaurav/in_context_+ve_-ve/report_experiments', EXPERIMENT_RUN_NAME)
os.makedirs(EXPERIMENT_PROJECT_DIR, exist_ok=True)
RESULTS_FILE_PATH = os.path.join(EXPERIMENT_PROJECT_DIR, f"{EXPERIMENT_RUN_NAME}_results.json")
# LOG_FILE_PATH = os.path.join(EXPERIMENT_PROJECT_DIR, f"{EXPERIMENT_RUN_NAME}_prompt_logs.txt")
KV_CACHE_FILE = os.path.join(EXPERIMENT_PROJECT_DIR, f"{EXPERIMENT_RUN_NAME}_static_kv_cache.pt")
EVAL_SUMMARY_PATH = os.path.join(EXPERIMENT_PROJECT_DIR, f"{EXPERIMENT_RUN_NAME}_evaluation_summary.json")

LLAMA3_SYSTEM_PROMPT = """You are an expert database routing system. Your task is to analyze a user's question and a list of available database schemas. You must identify the 10 most relevant database_ids that could answer the question.
Your output MUST be a numbered list, starting from 1, with each line containing only one database_id. Do not add any other text, explanation, or formatting."""
FEW_SHOT_EXAMPLE = """# --- Example ---
# Task: Examine all the database schemas provided above and return a ranked list of the 10 most relevant database_ids for answering the following question.
# Q: How many singers are from France?
#
# The 10 most relevant database_ids are:
1. singer
2. orchestra
3. musical
4. concert_singer
5. reality_shows
6. tvshow
7. party_host
8. hit_tracks
9. music
10. party_host
# --- End of Example ---"""

def load_model_and_tokenizer(model_path):
    """Loads the model and tokenizer from the specified path."""
    print(f"--- Loading Model and Tokenizer from {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"\nModel loaded successfully on device: {model.device}\n")
    return model, tokenizer

def load_spider_data(spider_dir):
    """Loads all necessary Spider dataset files."""
    print("--- Loading Spider Dataset ---")
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    
    dev_data = load_json(os.path.join(spider_dir, 'dev.json'))
    train_data = load_json(os.path.join(spider_dir, 'train_spider.json'))
    tables_data = load_json(os.path.join(spider_dir, 'tables.json'))
    
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

def construct_create_table_schemas_for_spider(all_db_schemas):
    """Generates CREATE TABLE strings for all schemas using the Spider format logic."""
    print("--- Generating CREATE TABLE Schema Strings for Spider ---")
    all_schemas_sql = {}
    
    for db_id, db_schema in tqdm(all_db_schemas.items(), desc="Generating Schemas"):
        sql_statements = []
        
        column_info = {}
        for i, (table_idx, col_name) in enumerate(db_schema['column_names_original']):
            if table_idx != -1:
                table_name = db_schema['table_names_original'][table_idx]
                col_type = db_schema['column_types'][i]
                column_info[i] = {"table_name": table_name, "col_name": col_name, "col_type": col_type.upper()}

        for table_idx, table_name in enumerate(db_schema['table_names_original']):
            column_definitions, table_constraints = [], []
            
            current_table_columns = [(col_idx, info['col_name'], info['col_type']) for col_idx, info in column_info.items() if info['table_name'] == table_name]
            pk_col_indices = [pk_idx for pk_idx in db_schema['primary_keys'] if column_info[pk_idx]['table_name'] == table_name]
            pk_col_names = [column_info[pk_idx]['col_name'] for pk_idx in pk_col_indices]

            for col_idx, col_name, col_type in current_table_columns:
                col_def_str = f"  `{col_name}` {col_type}"
                if len(pk_col_names) == 1 and col_name in pk_col_names:
                    col_def_str += " PRIMARY KEY"
                column_definitions.append(col_def_str)

            if len(pk_col_names) > 1:
                pk_cols_str = ", ".join([f"`{name}`" for name in pk_col_names])
                table_constraints.append(f"  PRIMARY KEY ({pk_cols_str})")

            for fk_col_idx, ref_col_idx in db_schema['foreign_keys']:
                if column_info[fk_col_idx]['table_name'] == table_name:
                    fk_info, ref_info = column_info[fk_col_idx], column_info[ref_col_idx]
                    table_constraints.append(f"  FOREIGN KEY (`{fk_info['col_name']}`) REFERENCES `{ref_info['table_name']}`(`{ref_info['col_name']}`)")
            
            sql_statements.append(f"CREATE TABLE `{table_name}` (\n" + ",\n".join(column_definitions + table_constraints) + "\n);")
        all_schemas_sql[db_id] = "\n\n".join(sql_statements)
    
    print(f"Generated {len(all_schemas_sql)} schema strings.\n")
    return all_schemas_sql

def generate_or_load_kv_cache(model, tokenizer, all_schemas_sql, positive_map, negative_map):
    """Generates or loads the static KV cache atomically to prevent corruption."""
    print("--- Handling Static KV Cache ---")
    tmp_kv_cache_file = KV_CACHE_FILE + ".tmp"

    all_db_blocks = []
    for db_id, sql_schema in all_schemas_sql.items():
        db_block = f"database_id: {db_id}\n{sql_schema}"
        pos_examples = positive_map.get(db_id, [])
        if pos_examples:
            db_block += f"\n# Here are some example questions that CAN be answered by the schema above:\n" + "\n".join([f"-- {ex}" for ex in pos_examples])
        neg_examples = negative_map.get(db_id, [])
        if neg_examples:
            db_block += f"\n# Here are some example questions that CANNOT be answered by the schema above:\n" + "\n".join([f"-- {ex}" for ex in neg_examples])
        all_db_blocks.append(db_block)
    
    all_schemas_str = "\n\n------------------------------------------------------------------------------------------\n\n".join(all_db_blocks)
    static_user_content = f"Here are all the available databases:\n{all_schemas_str}\n\n---\nNow, follow this example to understand the task and format.\n{FEW_SHOT_EXAMPLE}"
    static_messages = [{"role": "system", "content": LLAMA3_SYSTEM_PROMPT}, {"role": "user", "content": static_user_content}]

    if os.path.exists(KV_CACHE_FILE):
        try:
            print(f"Loading existing KV cache from {KV_CACHE_FILE}...")
            static_past_key_values = torch.load(KV_CACHE_FILE, map_location=model.device)
            print("KV cache loaded successfully.\n")
            return static_past_key_values, static_messages
        except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
            print(f"Warning: Could not load KV cache file ('{e}'). The file is likely corrupted.")
            print("Deleting it and regenerating...")
            os.remove(KV_CACHE_FILE)

    print("Generating new static KV cache. This is a one-time process.")
    static_inputs = tokenizer.apply_chat_template(static_messages, add_generation_prompt=False, tokenize=True, return_tensors="pt").to(model.device)
    print(f"Static prompt token length: {static_inputs.shape[1]}")

    with torch.no_grad():
        outputs = model(input_ids=static_inputs, use_cache=True)
        static_past_key_values = outputs.past_key_values
    
    torch.save(static_past_key_values, tmp_kv_cache_file)
    os.rename(tmp_kv_cache_file, KV_CACHE_FILE)
    print(f"Static KV cache generated and saved to {KV_CACHE_FILE}.\n")
    
    del static_inputs, outputs
    gc.collect(); torch.cuda.empty_cache()
    
    return static_past_key_values, static_messages

def get_prediction_with_kv_cache(model, tokenizer, static_cache, nl_query, all_db_ids_list, max_steps=150):
    """Generates a prediction using the KV cache and an explicit list of DB IDs."""
    eot_token_id = tokenizer.eos_token_id
    
    db_list_str = ", ".join(all_db_ids_list)
    dynamic_user_content = (f"\n\n---\n# Task: Examine all the database schemas provided above and return a ranked list of the 10 most relevant database_ids for answering the following question.\n"
                            f"# Your selection MUST be from the following list of valid database_ids: {db_list_str}\n"
                            f"# Q: {nl_query}"
                            f"# A: The 10 most relevant database_ids are:")
   
    manual_dynamic_string = (
        f"{dynamic_user_content}"
        f"<|im_end|>\n<|im_start|>assistant\n\n"
    )
    
    dynamic_inputs = tokenizer(
        manual_dynamic_string, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).input_ids.to(model.device)
    
    current_cache = copy.deepcopy(static_cache)
    
    with torch.no_grad():
        outputs = model(input_ids=dynamic_inputs, past_key_values=current_cache, use_cache=True)
    # pdb.set_trace()
    current_cache, next_token_logits = outputs.past_key_values, outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = [next_token_id]

    for _ in range(1, max_steps):
        with torch.no_grad():
            outputs = model(input_ids=next_token_id, past_key_values=current_cache, use_cache=True)
        current_cache, next_token_logits = outputs.past_key_values, outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # pdb.set_trace()
        generated_ids.append(next_token_id)
        if next_token_id.item() == eot_token_id:
            break

    generated_tokens = torch.cat(generated_ids, dim=-1)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    # pdb.set_trace()
    dynamic_messages = [{"role": "user", "content": dynamic_user_content}]

    return generated_text, dynamic_messages

def parse_top_k_response(raw_text, all_db_ids):
    """Extracts an ordered list of up to 10 database IDs from the model's raw text output."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(db_id) for db_id in all_db_ids) + r')\b')
    found_dbs = pattern.findall(raw_text)
    return list(dict.fromkeys(found_dbs))[:10]

def run_evaluation(results_file):
    print(f"\n--- Evaluating results from: {results_file} ---")
    with open(results_file, 'r') as f: results = json.load(f)

    k_values = [1, 3, 5, 10]
    recall_counts = {k: 0 for k in k_values}
    num_queries = len(results)
    if num_queries == 0: return

    for res in results:
        for k in k_values:
            if res['true_db_id'] in res['ranked_predicted_dbs'][:k]: recall_counts[k] += 1
    recall_scores = {k: (c / num_queries) * 100 for k, c in recall_counts.items()}
    runtimes = [res['runtime_seconds'] for res in results if 'runtime_seconds' in res]
    avg_runtime = np.mean(runtimes) if runtimes else 0

    print("\n" + "="*40 + "\n      EVALUATION SUMMARY\n" + "="*40)
    print("\n--- Schema Retrieval Performance (Recall@K) ---")
    print(f"Evaluated on {num_queries} queries.")
    for k, r in recall_scores.items(): print(f"  Recall@{k}: {r:.2f}%")
    print("\n--- Timing Performance ---")
    print(f"  Average time per query: {avg_runtime:.2f} seconds")
    
    summary_data = {"num_queries_evaluated": num_queries, "recall_scores_percent": recall_scores, "average_runtime_seconds": avg_runtime}
    with open(EVAL_SUMMARY_PATH, 'w') as f: json.dump(summary_data, f, indent=2)
    print(f"\nEvaluation summary saved to: {EVAL_SUMMARY_PATH}\n" + "="*40)

def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    dev_data, train_data, all_schemas = load_spider_data(SPIDER_DIR)
    
    positive_map, negative_map, test_set_queries = create_examples_and_test_set(train_data, dev_data)
    all_schemas_sql = construct_create_table_schemas_for_spider(all_schemas)
    static_kv_cache, static_messages = generate_or_load_kv_cache(model, tokenizer, all_schemas_sql, positive_map, negative_map)
    experiment_results = []
    if os.path.exists(RESULTS_FILE_PATH):
        with open(RESULTS_FILE_PATH, 'r') as f: experiment_results = json.load(f)
    completed_queries = {res['nl_query_text'] for res in experiment_results}
    
    print(f"--- Starting Inference on {len(test_set_queries)} Spider queries ({len(completed_queries)} completed) ---")
    all_db_ids_list = list(all_schemas.keys())
    
    # if len(completed_queries) == 0:
    #     open(LOG_FILE_PATH, 'w').close()

    for idx, item in enumerate(tqdm(test_set_queries, desc="Processing Spider Test Queries")):
        if item['question'] in completed_queries:
            continue
            
        start_time = time.time()
        try:
            raw_output, dynamic_messages = get_prediction_with_kv_cache(model, tokenizer, static_kv_cache, item['question'], all_db_ids_list)
            predicted_dbs = parse_top_k_response(raw_output, all_db_ids_list)
            
            result = {
                'query_number': idx + 1,
                'nl_query_text': item['question'],
                'true_db_id': item['db_id'],
                'ranked_predicted_dbs': predicted_dbs,
                'raw_model_output': raw_output,
                'runtime_seconds': time.time() - start_time
            }
            experiment_results.append(result)
            completed_queries.add(item['question'])


            # if len(completed_queries) <= 5:
            #     full_messages_for_log = static_messages + dynamic_messages
            #     full_prompt_text = tokenizer.apply_chat_template(full_messages_for_log, tokenize=False, add_generation_prompt=True)
                
            #     with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_f:
            #         log_f.write(f"\n\n{'='*80}\n           FULL PROMPT FOR QUERY #{len(completed_queries)}\n{'='*80}\n\n")
            #         log_f.write(full_prompt_text.rsplit("\n\n", 1)[0]) # Remove the final assistant prompt for cleaner logging
            #         log_f.write(f"\n\n--- RAW MODEL OUTPUT ---\n{raw_output}\n")

        except Exception as e:
            print(f"ERROR processing query: '{item['question']}'. Error: {e}")
            traceback.print_exc()
        finally:
            gc.collect(); torch.cuda.empty_cache()
            with open(RESULTS_FILE_PATH, 'w') as f_out:
                json.dump(experiment_results, f_out, indent=2)

    print(f"\n--- Inference Complete. Results saved to {RESULTS_FILE_PATH} ---")
    run_evaluation(RESULTS_FILE_PATH)

if __name__ == "__main__":
    main()