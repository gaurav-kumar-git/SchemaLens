import os
import json
import re
import random
import time
import gc
from collections import defaultdict
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers.models.llama.modeling_llama import repeat_kv
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pdb
from matplotlib.patches import Patch
import pandas as pd
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

TOP_20_HEADS = [
    (17, 26), (14, 20), (18, 8), (25, 7), (17, 29), 
    (17, 28), (22, 14), (16, 8), (17, 21), (22, 0),
    (26, 12), (16, 9), (24, 25), (17, 31), (26, 31),
    (14, 3), (24, 27), (14, 29), (24, 23), (14, 14)
]

QUERY_SELECTION_METHOD = "id" 
QUERY_ID =  1245 #367 # done 1054
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_DATABASES_IN_PROMPT = 30
RANDOM_SEED = 42
MODEL_PATH = "/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BIRD_DIR = '/scratch/gaurav/data/BIRD'
EXPERIMENT_PROJECT_DIR = f'/scratch/gaurav/in_context_+ve_-ve/attention_analysis_positional_sweep_TOP_20_HEADS_FOCUS'
LLAMA3_SYSTEM_PROMPT = """You are an expert database routing system. Your task is to analyze a user's question and a list of available database schemas. You must identify the 10 most relevant database_ids that could answer the question.
Your output MUST be a numbered list, starting from 1, with each line containing only one database_id. Do not add any other text, explanation, or formatting."""
FEW_SHOT_EXAMPLE = """# --- Example ---
# Task: Examine all the database schemas provided above and return a ranked list of the 10 most relevant database_ids for answering the following question.
# Q: How many singers are from France?

# A: The 10 most relevant database_ids are:
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
    print(f"--- Loading Model and Tokenizer from {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"\nModel loaded successfully on device: {model.device}\n")
    return model, tokenizer

def load_bird_data(bird_dir):
    print("--- Loading BIRD Dataset ---")
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    dev_data = load_json(os.path.join(bird_dir, 'dev', 'dev.json'))
    train_data = load_json(os.path.join(bird_dir, 'train', 'train.json'))
    tables_data = load_json(os.path.join(bird_dir, 'dev', 'dev_tables.json')) + \
                  load_json(os.path.join(bird_dir, 'train', 'train_tables.json'))
    all_schemas = {db['db_id']: db for db in tables_data}
    return dev_data, train_data, all_schemas

def create_examples(train_data, dev_data):
    all_data = train_data + dev_data; all_questions_by_db = defaultdict(list)
    for item in all_data: all_questions_by_db[item['db_id']].append(item['question'])
    random.seed(42); positive_map = {db_id: random.sample(qs, min(len(qs), 5)) for db_id, qs in all_questions_by_db.items()}
    negative_map = defaultdict(list); all_db_ids = list(all_questions_by_db.keys())
    for target_db_id in all_db_ids:
        other_db_ids = [db for db in all_db_ids if db != target_db_id]
        sampled_db_ids = random.sample(other_db_ids, min(len(other_db_ids), 5))
        for neg_db_id in sampled_db_ids:
            if all_questions_by_db[neg_db_id]: negative_map[target_db_id].append(random.choice(all_questions_by_db[neg_db_id]))
    return positive_map, negative_map

def construct_create_table_schemas(all_db_schemas):
    # This function is long and correct, so I'll omit its body for brevity
    all_schemas_sql = {};
    for db_id, db_schema in tqdm(all_db_schemas.items(), desc="Generating Schemas"):
        flat_pk_list = [pk for item in db_schema.get('primary_keys', []) for pk in (item if isinstance(item, list) else [item])]
        sql_statements = []; column_info_by_index = {i: {"name": c_name, "table_index": t_idx, "type": db_schema['column_types'][i], "table_name": db_schema['table_names_original'][t_idx]} for i, (t_idx, c_name) in enumerate(db_schema['column_names_original']) if c_name != "*"}
        for table_idx, table_name in enumerate(db_schema['table_names_original']):
            column_definitions, table_constraints = [], []; current_table_columns = [(c_idx, c_info) for c_idx, c_info in column_info_by_index.items() if c_info['table_index'] == table_idx]
            pk_column_indices = [pk_idx for pk_idx in flat_pk_list if column_info_by_index.get(pk_idx) and column_info_by_index[pk_idx]['table_index'] == table_idx]
            for col_idx, col_info in current_table_columns:
                is_pk = col_idx in pk_column_indices; sql_type = "INTEGER" if col_info['type'] == "number" and is_pk else "TEXT" if col_info['type'] == "text" else "REAL" if col_info['type'] == "number" else "DATETIME" if col_info['type'] == "time" else "BOOLEAN"; col_def_str = f"  {col_info['name']} {sql_type}" + (" PRIMARY KEY" if is_pk and len(pk_column_indices) == 1 else ""); column_definitions.append(col_def_str)
            if len(pk_column_indices) > 1: table_constraints.append(f"  PRIMARY KEY ({', '.join([column_info_by_index[idx]['name'] for idx in pk_column_indices])})")
            for fk_col_idx, ref_col_idx in db_schema.get('foreign_keys', []):
                if column_info_by_index.get(fk_col_idx) and column_info_by_index[fk_col_idx]['table_index'] == table_idx:
                    fk_info, ref_info = column_info_by_index[fk_col_idx], column_info_by_index[ref_col_idx]; table_constraints.append(f"  FOREIGN KEY ({fk_info['name']}) REFERENCES {ref_info['table_name']}({ref_info['name']})")
            sql_statements.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(column_definitions + table_constraints) + "\n);")
        all_schemas_sql[db_id] = "\n\n".join(sql_statements)
    return all_schemas_sql

def build_prompt_with_boundaries(all_schemas_sql, positive_map, negative_map, ordered_db_ids, nl_query):
    # This function is long and correct, so I'll omit its body for brevity
    all_db_blocks = []
    for db_id in ordered_db_ids:
        sql_schema = all_schemas_sql[db_id]
        db_block = f"database_id: {db_id}\n{sql_schema}"
        pos_examples = positive_map.get(db_id, [])
        if pos_examples:
            db_block += f"\n# Here are some example questions that CAN be answered by the schema above:\n" + "\n".join([f"-- {ex}" for ex in pos_examples])
        neg_examples = negative_map.get(db_id, [])
        if neg_examples:
            db_block += f"\n# Here are some example questions that CANNOT be answered by the schema above:\n" + "\n".join([f"-- {ex}" for ex in neg_examples])
        all_db_blocks.append(db_block)
    all_schemas_str = "\n\n------------------------------------------------------------------------------------------\n\n".join(all_db_blocks)
    db_list_str = ", ".join(ordered_db_ids)
    user_content = (
        f"Here are all the available databases:\n{all_schemas_str}\n\n---\n"
        f"Now, follow this example to understand the task and format.\n{FEW_SHOT_EXAMPLE}\n\n---\n"
        f"# Task: Examine all the database schemas provided above and return a ranked list of the 10 most relevant database_ids for answering the following question.\n"
        f"# Your selection MUST be from the following list of valid database_ids: {db_list_str}\n"
        f"# Q: {nl_query}"
        f"# A: # The 10 most relevant database_ids are:"
    )
    messages = [
        {"role": "system", "content": LLAMA3_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, ordered_db_ids

def find_token_boundaries(tokenizer, messages, all_db_ids):
    print("\n--- Finding Token Boundaries (Natural Prompt Mode) ---")
    boundaries = {}
    system_user_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_prompt_encoding = tokenizer(full_prompt_str, return_offsets_mapping=True)
    full_prompt_tokens = full_prompt_encoding['input_ids']
    offset_mapping = full_prompt_encoding['offset_mapping']
    system_prompt_only_str = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=False)
    system_end_char = len(system_prompt_only_str)
    system_end_token = 0
    for i, (start_char, end_char) in enumerate(offset_mapping):
        if start_char < system_end_char:
            system_end_token = i + 1
    boundaries['system'] = (0, system_end_token)
    db_separator = "\n\n------------------------------------------------------------------------------------------\n\n"
    for i, db_id in enumerate(all_db_ids):
        start_marker = f"database_id: {db_id}"
        try:
            start_char_idx = system_user_str.index(start_marker)
            if i < len(all_db_ids) - 1:
                end_char_idx = system_user_str.find(db_separator, start_char_idx)
            else:
                final_separator = "\n\n---\n"
                end_char_idx = system_user_str.find(final_separator, start_char_idx)
            start_token_idx, end_token_idx = -1, -1
            for token_i, (start_char, end_char) in enumerate(offset_mapping):
                if start_token_idx == -1 and start_char >= start_char_idx:
                    start_token_idx = token_i
                if end_token_idx == -1 and end_char >= end_char_idx:
                    end_token_idx = token_i
                    break
            if start_token_idx != -1 and end_token_idx != -1:
                boundaries[db_id] = (start_token_idx, end_token_idx)
        except ValueError:
            print(f"Warning: Could not find markers for db_id: {db_id} in the prompt string.")
            continue
    query_marker = "# Task: Examine all the database schemas"
    try:
        query_start_char_idx = system_user_str.index(query_marker)
        query_start_token = -1
        for token_i, (start_char, end_char) in enumerate(offset_mapping):
            if start_char >= query_start_char_idx:
                query_start_token = token_i
                break
        user_content_end_char = len(system_user_str)
        query_end_token = len(full_prompt_tokens)
        for token_i, (start_char, end_char) in enumerate(offset_mapping):
            if end_char >= user_content_end_char:
                query_end_token = token_i
                break
        if query_start_token != -1:
            boundaries['query'] = (query_start_token, query_end_token)
    except ValueError:
        print("Warning: Could not find the query block marker.")
    print(f"Identified boundaries for {len(boundaries)} blocks.")
    return boundaries, torch.tensor([full_prompt_tokens]).to('cuda')


def get_full_model_attention_flow(model, input_ids, boundaries):
    print("\n--- Starting Final Attention Flow Analysis (Re-computing Q) ---")
    config = model.config
    num_layers, num_heads = config.num_hidden_layers, config.num_attention_heads
    num_key_value_heads, num_key_value_groups = config.num_key_value_heads, num_heads // config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    softmax_scaling = 1.0 / (head_dim**0.5)
    seq_len = input_ids.shape[1]
    target_blocks = ['system'] + sorted([k for k in boundaries.keys() if k not in ['system', 'query']])
    attention_flow_matrices = {block: np.zeros((num_layers, num_heads)) for block in target_blocks}
    query_start, query_end = boundaries['query']
    num_query_tokens = query_end - query_start
    with torch.no_grad():
        print("Performing a single forward pass to get KV Cache and all hidden states...")
        outputs = model(input_ids, use_cache=True, output_hidden_states=True)
        kvcache, all_hidden_states = outputs.past_key_values, outputs.hidden_states
        print("Creating causal mask with torch.tril()...")
        full_causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device))
        causal_mask_for_queries = full_causal_mask[query_start:query_end, :]
        for layer_idx in tqdm(range(num_layers), desc="Analyzing Layers"):
            layer = model.model.layers[layer_idx]
            hidden_state_input = all_hidden_states[layer_idx]
            layernorm_output = layer.input_layernorm(hidden_state_input)
            q_proj_output = layer.self_attn.q_proj(layernorm_output)
            full_query_states = q_proj_output.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
            query_tokens_q = full_query_states[:, :, query_start:query_end, :]
            key_states_from_cache = kvcache[layer_idx][0]
            repeated_key_states = repeat_kv(key_states_from_cache, num_key_value_groups)
            attn_scores = torch.matmul(query_tokens_q, repeated_key_states.transpose(-2, -1)) * softmax_scaling
            attn_scores = attn_scores.masked_fill(causal_mask_for_queries == 0, torch.finfo(attn_scores.dtype).min)
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_tokens_q.dtype)
            for block_name in target_blocks:
                start, end = boundaries[block_name]
                block_attention = attn_weights[:, :, :, start:end]
                total_flow_per_head = block_attention.sum(dim=(-1, -2)).squeeze(0)
                avg_flow_per_head = total_flow_per_head / num_query_tokens
                block_length = end - start
                normalized_avg_flow = avg_flow_per_head / block_length if block_length > 0 else avg_flow_per_head
                attention_flow_matrices[block_name][layer_idx, :] = normalized_avg_flow.float().cpu().numpy()
    return attention_flow_matrices

def calculate_filtered_attention(matrix, heads_to_sum):
    """Sums attention values only for the specified (layer, head) pairs."""
    total_flow = 0.0
    for layer_idx, head_idx in heads_to_sum:
        if layer_idx < matrix.shape[0] and head_idx < matrix.shape[1]:
            total_flow += matrix[layer_idx, head_idx]
    return total_flow

def main():
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    dev_data, train_data, all_schemas = load_bird_data(BIRD_DIR)
    all_question_data = dev_data + train_data
    
    target_query_item = next((item for item in all_question_data if str(item.get('question_id')) == str(QUERY_ID)), None)
    if not target_query_item:
        raise ValueError(f"Could not find query with ID: {QUERY_ID}")

    nl_query, gold_db_id = target_query_item['question'], target_query_item['db_id']
    positive_map, negative_map = create_examples(train_data, dev_data)
    all_schemas_sql = construct_create_table_schemas(all_schemas)

    all_db_ids = sorted(list(all_schemas.keys()))
    other_db_ids = [db for db in all_db_ids if db != gold_db_id]
    num_distractors_to_sample = NUM_DATABASES_IN_PROMPT - 1
    random.seed(RANDOM_SEED)
    sampled_distractor_dbs = random.sample(other_db_ids, num_distractors_to_sample)

    safe_qid = re.sub(r'[^a-zA-Z0-9_-]', '_', str(QUERY_ID))
    os.makedirs(EXPERIMENT_PROJECT_DIR, exist_ok=True)
    # Update filenames for clarity
    plot_path = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_attention_vs_position_TOP20.png")
    log_path = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_positional_summary_log_TOP20.txt")

    positions_for_plot, gold_db_scores_for_plot, max_db_scores_for_plot = [], [], []

    for position_index in range(NUM_DATABASES_IN_PROMPT):
        schemas_for_prompt = list(sampled_distractor_dbs)
        schemas_for_prompt.insert(position_index, gold_db_id)
        human_readable_position = position_index + 1
        
        print(f"\n{'='*25}\n  Starting Run: Gold DB '{gold_db_id}' at Position {human_readable_position}/{NUM_DATABASES_IN_PROMPT}\n{'='*25}\n")
        
        messages, ordered_db_ids = build_prompt_with_boundaries(all_schemas_sql, positive_map, negative_map, schemas_for_prompt, nl_query)
        boundaries, input_ids = find_token_boundaries(tokenizer, messages, ordered_db_ids)
        attention_flow_matrices = get_full_model_attention_flow(model, input_ids, boundaries)
        
        total_attention_to_gold = calculate_filtered_attention(attention_flow_matrices[gold_db_id], TOP_20_HEADS)
        all_db_scores = {
            db_id: calculate_filtered_attention(matrix, TOP_20_HEADS)
            for db_id, matrix in attention_flow_matrices.items() if db_id in ordered_db_ids
        }
        
        positions_for_plot.append(human_readable_position)
        gold_db_scores_for_plot.append(total_attention_to_gold)
        max_attention_in_run = max(all_db_scores.values()) if all_db_scores else 0.0
        max_db_scores_for_plot.append(max_attention_in_run)
        
        del messages, ordered_db_ids, boundaries, input_ids, attention_flow_matrices
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n--- All runs complete. Generating final plot... ---")
    plt.figure(figsize=(12, 7))
    
    plt.plot(positions_for_plot, gold_db_scores_for_plot, marker='o', linestyle='-', color='b', label=f'Attention to Gold DB ({gold_db_id})')
    plt.plot(positions_for_plot, max_db_scores_for_plot, marker='x', linestyle='--', color='r', label='Max Attention to any DB')
    
    plt.title(f'Attention from Top 20 Heads to Gold DB ({gold_db_id}) vs. Position\n(Query: {safe_qid}, Total DBs: {NUM_DATABASES_IN_PROMPT})')
    plt.xlabel('Position of Gold Database in Prompt')
    plt.ylabel('Attention Flow (Sum over Top 20 Heads)')
    
    plt.xticks(np.arange(1, NUM_DATABASES_IN_PROMPT + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to: {plot_path}")
    print(f"\n--- Experiment finished in {time.time() - start_time:.2f} seconds. ---")
       
if __name__ == "__main__":
    main()