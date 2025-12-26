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
QUERY_SELECTION_METHOD = "text" 
QUERY_ID =  1321 #896, 746,954, 974, 1517
QUERY_TEXT = "How many events were held at coordinate 97,40?"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_ALL_DATABASES = True
NUM_DATABASES_IN_PROMPT = 30
RANDOM_SEED = 42
MODEL_PATH = "/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BIRD_DIR = '/scratch/gaurav/data/BIRD'
EXPERIMENT_PROJECT_DIR = f'/scratch/gaurav/in_context_+ve_-ve/attention_analysis_full_prompt_positional_sweep'
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
    # pdb.set_trace()
    return all_schemas_sql

def build_prompt_with_boundaries(all_schemas_sql, positive_map, negative_map, ordered_db_ids, nl_query):
    """
    Constructs the prompt WITHOUT special [DB_START/END] markers and fixes
    the assistant message formatting.
    """
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
        # {"role": "assistant"} # Correctly formatted
    ]
    
    return messages, ordered_db_ids

def find_token_boundaries(tokenizer, messages, all_db_ids):
    """
    Finds token boundaries based on the 'database_id:' strings and the
    separators between schemas, without relying on special markers.
    """
    print("\n--- Finding Token Boundaries (Natural Prompt Mode) ---")
    boundaries = {}
    # system_user_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False)
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
        
    # pdb.set_trace()
            
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
    # pdb.set_trace()
    return boundaries, torch.tensor([full_prompt_tokens]).to('cuda')

def generate_model_output(model, tokenizer, messages):
    print("\n--- Generating Model Output ---")
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
    # pdb.set_trace()
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs))
    generated_text = tokenizer.decode(outputs[0, inputs.shape[-1]:], skip_special_tokens=True)
    print("--- Model's Generated Text ---\n" + generated_text + "\n-----------------------------\n")
    return generated_text

def get_full_model_attention_flow(model, input_ids, boundaries):
    """
    Performs an efficient attention flow analysis. This version solves the 'query_states'
    AttributeError by manually re-computing the query states for each layer from its
    input hidden state, which is the correct and robust way to perform this analysis
    on a standard Hugging Face model.
    """
    print("\n--- Starting Final Attention Flow Analysis (Re-computing Q) ---")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads
    head_dim = config.hidden_size // num_heads
    softmax_scaling = 1.0 / (head_dim**0.5)
    seq_len = input_ids.shape[1]
    target_blocks = ['system'] + sorted([k for k in boundaries.keys() if k not in ['system', 'query']])
    attention_flow_matrices = {block: np.zeros((num_layers, num_heads)) for block in target_blocks} # block_name -> (layers, heads) for each block get a matrix of layers x heads
    query_start, query_end = boundaries['query']
    num_query_tokens = query_end - query_start
    # pdb.set_trace()
    with torch.no_grad():
        # We need both the KV Cache and the hidden states from every layer.
        print("Performing a single forward pass to get KV Cache and all hidden states...")
        outputs = model(input_ids, use_cache=True, output_hidden_states=True)
        kvcache = outputs.past_key_values # List of (key, value) tuples for each layer
        all_hidden_states = outputs.hidden_states
        print("Creating causal mask with torch.tril()...")
        full_causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device)
        )
        causal_mask_for_queries = full_causal_mask[query_start:query_end, :]
        for layer_idx in tqdm(range(num_layers), desc="Analyzing Layers"):
            layer = model.model.layers[layer_idx]
            # all_hidden_states[0] is the input to layer 0, [1] is input to layer 1, etc.
            hidden_state_input = all_hidden_states[layer_idx]
            # Replicate the first part of the attention mechanism to get Q
            layernorm_output = layer.input_layernorm(hidden_state_input)
            q_proj_output = layer.self_attn.q_proj(layernorm_output)
            # Reshape to multi-head format, exactly as the model does internally
            full_query_states = q_proj_output.view(1, seq_len, num_heads, head_dim).transpose(1, 2) # got the Q states for all tokens
            query_tokens_q = full_query_states[:, :, query_start:query_end, :] #1, num_heads, N_q, d # take only the query tokens
            key_states_from_cache = kvcache[layer_idx][0] # get the key from the cache for this layer shape: 1, num_key_value_heads, N, d
            repeated_key_states = repeat_kv(key_states_from_cache, num_key_value_groups) #1, num_heads, N, d
            attn_scores = torch.matmul(query_tokens_q, repeated_key_states.transpose(-2, -1)) * softmax_scaling # 1, num_heads, N_q, N
            attn_scores = attn_scores.masked_fill(causal_mask_for_queries == 0, torch.finfo(attn_scores.dtype).min) # fill masked positions
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_tokens_q.dtype) # softmax over last dim (N) # 1, num_heads, N_q, N you do softmax over last dim (N) so across each row(row is attention for token i of query to all tokens of input)
            for block_name in target_blocks:
                start, end = boundaries[block_name]
                block_attention = attn_weights[:, :, :, start:end] # 1, heads, N_q, N_block
                total_flow_per_head = block_attention.sum(dim=(-1, -2)).squeeze(0) # heads # sum over N_q and N_block # remove batch dim
                avg_flow_per_head = total_flow_per_head / num_query_tokens # heads # average over N_q(number of query tokens) # size: heads
                block_length = end - start
                if block_length > 0:
                    normalized_avg_flow = avg_flow_per_head / block_length
                else:
                    normalized_avg_flow = avg_flow_per_head # Should not happen, but safe
                
                attention_flow_matrices[block_name][layer_idx, :] = normalized_avg_flow.float().cpu().numpy() # store in numpy array of size heads for this layer
                
    return attention_flow_matrices # block_name -> (layers, heads) matrices this is a dict of block_name to (layers, heads) matrices

def main():
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    dev_data, train_data, all_schemas = load_bird_data(BIRD_DIR)
    all_question_data = dev_data + train_data
    target_query_item = None

    if QUERY_SELECTION_METHOD == "id":
        print(f"--- Searching for question by ID: {QUERY_ID} ---")
        target_query_item = next((item for item in all_question_data if str(item.get('question_id')) == str(QUERY_ID)), None)
        if not target_query_item: raise ValueError(f"Could not find query with ID: {QUERY_ID}")
    elif QUERY_SELECTION_METHOD == "text":
        print(f"--- Searching for question by text... ---")
        # This is the new, robust line
        target_query_item = next((item for item in all_question_data if item['question'].strip() == QUERY_TEXT.strip()), None)
        if not target_query_item: raise ValueError(f"Could not find query with the specified text. Make sure it's an exact match.")
    else:
        raise ValueError(f"Invalid QUERY_SELECTION_METHOD: '{QUERY_SELECTION_METHOD}'. Must be 'id' or 'text'.")
    
    nl_query, gold_db_id = target_query_item['question'], target_query_item['db_id']
    positive_map, negative_map = create_examples(train_data, dev_data)
    all_schemas_sql = construct_create_table_schemas(all_schemas)

    all_db_ids = sorted(list(all_schemas.keys()))
    other_db_ids = [db for db in all_db_ids if db != gold_db_id]
    num_distractors_to_sample = NUM_DATABASES_IN_PROMPT - 1
    if len(other_db_ids) < num_distractors_to_sample:
        raise ValueError(f"Not enough non-gold databases to sample {num_distractors_to_sample} distractors.")

    print(f"\n--- Sampling {num_distractors_to_sample} distractor DBs using seed {RANDOM_SEED} ---")
    random.seed(RANDOM_SEED)
    sampled_distractor_dbs = random.sample(other_db_ids, num_distractors_to_sample)

    if QUERY_SELECTION_METHOD == "id":
        safe_qid = re.sub(r'[^a-zA-Z0-9_-]', '_', str(QUERY_ID))
    else:
        safe_qid = re.sub(r'[^a-zA-Z0-9_-]', '_', nl_query[:50])

    os.makedirs(EXPERIMENT_PROJECT_DIR, exist_ok=True)
    log_path = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_positional_summary_log_{NUM_DATABASES_IN_PROMPT}.txt")
    plot_path = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_attention_vs_position_{NUM_DATABASES_IN_PROMPT}.png")
    detailed_log_path = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_max_attention_per_head_{NUM_DATABASES_IN_PROMPT}.txt")
    heatmap_dir = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_gold_focus_heatmaps_{NUM_DATABASES_IN_PROMPT}")
    os.makedirs(heatmap_dir, exist_ok=True)
    # added
    pos_focus_heatmap_dir = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_positional_focus_heatmaps_{NUM_DATABASES_IN_PROMPT}")
    os.makedirs(pos_focus_heatmap_dir, exist_ok=True)
    norm_heatmap_dir = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_normalized_focus_heatmaps_{NUM_DATABASES_IN_PROMPT}")
    os.makedirs(norm_heatmap_dir, exist_ok=True)

    positions_for_plot, gold_db_scores_for_plot = [], []
    max_db_scores_for_plot = [] # added for max_db_score for each gold position
    all_runs_correctness_matrices = []
    # exp_1 log
    with open(log_path, 'w') as log_file:
        query_id_to_log = target_query_item.get('question_id', 'N/A (from train set)')
        log_file.write(f"Positional Attention Analysis Summary Log\n")
        log_file.write(f"Query ID: {query_id_to_log}\n")
        log_file.write(f"Query: {nl_query}\n")
        log_file.write(f"Gold DB: {gold_db_id}\n")
        log_file.write(f"Total DBs in prompt: {NUM_DATABASES_IN_PROMPT}\n")
        log_file.write("="*50 + "\n")
    #exp_2 detailed log
    with open(detailed_log_path, 'w') as detailed_log_file:
        detailed_log_file.write(f"Detailed Head-by-Head Max Attention Log\n")
        detailed_log_file.write(f"For each (Layer, Head) pair, this log identifies the database block that received the maximum attention.\n")
        detailed_log_file.write(f"Query: {nl_query}\n\n")

    for position_index in range(NUM_DATABASES_IN_PROMPT):
        schemas_for_prompt = list(sampled_distractor_dbs)
        schemas_for_prompt.insert(position_index, gold_db_id)
        human_readable_position = position_index + 1
        
        print(f"\n{'='*25}\n  Starting Run: Gold DB '{gold_db_id}' at Position {human_readable_position}/{NUM_DATABASES_IN_PROMPT}\n{'='*25}\n")
        
        messages, ordered_db_ids = build_prompt_with_boundaries(all_schemas_sql, positive_map, negative_map, schemas_for_prompt, nl_query)
        boundaries, input_ids = find_token_boundaries(tokenizer, messages, ordered_db_ids)
        # pdb.set_trace() 
        attention_flow_matrices = get_full_model_attention_flow(model, input_ids, boundaries)
        
        # exp 1: total attention to gold db
        total_attention_to_gold = np.sum(attention_flow_matrices[gold_db_id])
        positions_for_plot.append(human_readable_position)
        gold_db_scores_for_plot.append(total_attention_to_gold)
        all_db_scores = {db_id: np.sum(attention_flow_matrices[db_id]) for db_id in ordered_db_ids}
        max_attention_in_run = max(all_db_scores.values()) if all_db_scores else 0.0
        max_db_scores_for_plot.append(max_attention_in_run)
        with open(log_path, 'a') as log_file:
            log_file.write(f"\n--- Position {human_readable_position} ---\n")
            log_file.write(f"Prompt Order: {', '.join(ordered_db_ids)}\n")
            scores_with_size_list = []
            for db_id in ordered_db_ids:
                if db_id in boundaries:
                    start_token, end_token = boundaries[db_id]
                    scores_with_size_list.append(f"{db_id}({end_token - start_token}): {all_db_scores.get(db_id, 0.0):.4f}")
            log_file.write(f"All DB Scores (DB_ID(token_size): score): [{', '.join(scores_with_size_list)}]\n")
            log_file.write(f"Gold DB ({gold_db_id}) Total Attention: {total_attention_to_gold:.4f}\n")
            log_file.write(f"Max Attention to any DB in this run: {max_attention_in_run:.4f}\n")
            
        # exp 2: detailed head-by-head analysis
        print(f"--- Performing detailed head-by-head analysis for position {human_readable_position}... ---")
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        with open(detailed_log_path, 'a') as detailed_log_file:
            detailed_log_file.write(f"\n{'='*25} Analysis for Gold DB at Position {human_readable_position} {'='*25}\n")
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    max_attention_value = -1.0
                    max_attention_db = None
                    blocks_to_check = ordered_db_ids
                    for block_name_to_check in blocks_to_check:
                        current_value = attention_flow_matrices[block_name_to_check][layer_idx, head_idx]
                        if current_value > max_attention_value:
                            max_attention_value = current_value
                            max_attention_db = block_name_to_check
                    if max_attention_db:
                        db_position = (ordered_db_ids.index(max_attention_db) + 1) if max_attention_db in ordered_db_ids else "N/A"
                        position_str = f"(at position {db_position}/{NUM_DATABASES_IN_PROMPT})" if db_position != "N/A" else ""
                        log_line = f"Layer {layer_idx:02d}, Head {head_idx:02d}: Max attention to {max_attention_db} {position_str}\n"
                        detailed_log_file.write(log_line)
                detailed_log_file.write("-------------------------------------------------\n")
                
        data_dir = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_gold_focus_data_arrays_{NUM_DATABASES_IN_PROMPT}")
        os.makedirs(data_dir, exist_ok=True)

        # exp 3: gold focus heatmap
        print(f"--- Generating gold focus heatmap for position {human_readable_position}... ---")
        gold_focus_map = np.zeros((num_layers, num_heads))
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                max_attention_value = -1.0
                max_attention_db = None
                blocks_to_check = ordered_db_ids
                for block_name in blocks_to_check:
                    if block_name in attention_flow_matrices:
                        current_value = attention_flow_matrices[block_name][layer_idx, head_idx]
                        if current_value > max_attention_value:
                            max_attention_value = current_value
                            max_attention_db = block_name 
                if max_attention_db == gold_db_id:
                    gold_focus_map[layer_idx, head_idx] = 1
        # gold focus map as a numpy array for future reference for plotting MVP heads           
        data_filename = f"focus_map_pos_{human_readable_position:02d}.npy"
        data_path = os.path.join(data_dir, data_filename)
        np.save(data_path, gold_focus_map)
        
        plt.figure(figsize=(12, 10))
        cmap = ListedColormap(['#A9D6E5', '#D62828']) 
        plot_data = gold_focus_map.T
        ax = sns.heatmap(plot_data, cmap=cmap, linewidths=.5, linecolor='gray', cbar=False)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, num_layers, 4) + 0.5)
        ax.set_xticklabels(np.arange(0, num_layers, 4))
        ax.set_yticks(np.arange(0, num_heads, 4) + 0.5)
        ax.set_yticklabels(np.arange(0, num_heads, 4))
        plt.title(f"Heads with Max Attention on Gold DB ('{gold_db_id}')\n(Compared ONLY against other DBs)\nGold DB at Position: {human_readable_position}/{NUM_DATABASES_IN_PROMPT}", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Head Index", fontsize=12) 
        heatmap_filename = f"gold_focus_map_pos_{human_readable_position:02d}.png"
        heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        
        
        # exp 4: positional focus heatmap
        print(f"--- Generating detailed positional focus heatmap for position {human_readable_position}... ---")
        max_pos_matrix = np.zeros((num_layers, num_heads), dtype=int)
        is_correct_matrix = np.zeros((num_layers, num_heads), dtype=int)

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                max_attention_value = -1.0
                max_attention_block = None                
                for block_name, matrix in attention_flow_matrices.items():
                    if block_name in ['query', 'system']: continue
                    current_value = matrix[layer_idx, head_idx]
                    if current_value > max_attention_value:
                        max_attention_value = current_value
                        max_attention_block = block_name
                
                position_of_max = 0
                if max_attention_block and max_attention_block in ordered_db_ids:
                    position_of_max = ordered_db_ids.index(max_attention_block) + 1
                
                max_pos_matrix[layer_idx, head_idx] = position_of_max
                
                if position_of_max == human_readable_position:
                    is_correct_matrix[layer_idx, head_idx] = 1
        
        all_runs_correctness_matrices.append(is_correct_matrix)
        plt.figure(figsize=(18, 10))

        cmap = ListedColormap(['#E0E0E0', '#D62828'])

        plot_data_color = is_correct_matrix.T
        plot_data_annot = max_pos_matrix.T

        ax = sns.heatmap(
            plot_data_color,
            annot=plot_data_annot,
            fmt='d',
            cmap=cmap,
            linewidths=.5,
            linecolor='white',
            cbar=False,
            annot_kws={"size": 6, "color": "black"}
        )

        legend_elements = [Patch(facecolor='#D62828', edgecolor='black', label='Correct Focus (on Gold DB)'),
                        Patch(facecolor='#E0E0E0', edgecolor='black', label='Incorrect Focus (Distractor or System)')]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, num_layers, 2) + 0.5)
        ax.set_xticklabels(np.arange(0, num_layers, 2))
        ax.set_yticks(np.arange(0, num_heads, 2) + 0.5)
        ax.set_yticklabels(np.arange(0, num_heads, 2))

        plt.title(f"Positional Focus of Each Head (Layer, Head)\nGold DB ('{gold_db_id}') is at Position: {human_readable_position}", fontsize=16)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Head Index", fontsize=12)
        heatmap_filename_pos = f"positional_focus_map_pos_{human_readable_position:02d}.png"
        heatmap_path_pos = os.path.join(pos_focus_heatmap_dir, heatmap_filename_pos)
        plt.savefig(heatmap_path_pos, dpi=200, bbox_inches='tight')
        plt.close()
        
        # exp 5: Generate Normalized Focus Heatmap for the current position
        print(f"--- Generating normalized focus heatmap for position {human_readable_position}... ---")
        gold_db_matrix = attention_flow_matrices[gold_db_id]
        total_db_matrix = np.zeros((num_layers, num_heads))
        for block_name, matrix in attention_flow_matrices.items():
            if block_name in ordered_db_ids: # ensure only db blocks are added
                total_db_matrix += matrix

        normalized_focus_matrix = np.divide(
            gold_db_matrix, 
            total_db_matrix, 
            out=np.zeros_like(gold_db_matrix),
            where=total_db_matrix!=0
        )
        plt.figure(figsize=(16, 10))
        plot_data = normalized_focus_matrix.T
        ax = sns.heatmap(
            plot_data,
            cmap='Reds',      
            cbar=True,
            vmin=0,
            vmax=1,
            linewidths=.1
        )
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, num_layers, 2) + 0.5)
        ax.set_xticklabels(np.arange(0, num_layers, 2))
        ax.set_yticks(np.arange(0, num_heads, 2) + 0.5)
        ax.set_yticklabels(np.arange(0, num_heads, 2))

        plt.title(f"Normalized Focus on Gold DB ('{gold_db_id}')\nGold DB is at Position: {human_readable_position}", fontsize=16)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Head Index", fontsize=12)
        heatmap_filename = f"normalized_focus_heatmap_pos_{human_readable_position:02d}.png"
        heatmap_path = os.path.join(norm_heatmap_dir, heatmap_filename) # Use the new directory path
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        del messages, ordered_db_ids, boundaries, input_ids, attention_flow_matrices, gold_focus_map
        gc.collect()
        torch.cuda.empty_cache()
    print("\n--- Generating final summary heatmap of head success rates across all positions... ---")

    if not all_runs_correctness_matrices:
        print("No data was collected for the summary heatmap. Skipping.")
    else:
        total_correct_counts = np.sum(all_runs_correctness_matrices, axis=0)
        success_rate_matrix = total_correct_counts / NUM_DATABASES_IN_PROMPT
        vmax_global_best = np.max(success_rate_matrix)
        plt.figure(figsize=(16, 10))
        plot_data = success_rate_matrix.T
        ax = sns.heatmap(
            plot_data,
            cmap='Reds',      # A good colormap for showing intensity from low (white) to high (dark red).
            cbar=True,        # The color bar is essential to understand the scale (0.0 to 1.0).
            vmin=0,           # Fix the color scale minimum to 0.
            vmax=vmax_global_best,           # Fix the color scale maximum to the global best.
            annot=False       # We don't annotate, as the floats would be too cluttered. The color is the data.
        )
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, num_layers, 2) + 0.5)
        ax.set_xticklabels(np.arange(0, num_layers, 2))
        ax.set_yticks(np.arange(0, num_heads, 2) + 0.5)
        ax.set_yticklabels(np.arange(0, num_heads, 2))
        plt.title(f"Head Success Rate Across All 30 Positions\n(Query: {safe_qid}, Gold DB: {gold_db_id})", fontsize=16)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Head Index", fontsize=12)
        summary_heatmap_path = os.path.join(EXPERIMENT_PROJECT_DIR, f"{safe_qid}_head_success_rate_summary.png")
        plt.savefig(summary_heatmap_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Head success rate summary heatmap saved to: {summary_heatmap_path}")
    print(f"\n--- All runs complete. Generating final plot... ---")
    
    # the final total and max attention plot
    plt.figure(figsize=(12, 7))
    plt.plot(positions_for_plot, gold_db_scores_for_plot, marker='o', linestyle='-', color='b', label=f'Attention to Gold DB ({gold_db_id})')
    plt.plot(positions_for_plot, max_db_scores_for_plot, marker='x', linestyle='--', color='r', label='Max Attention to any DB')
    plt.title(f'Total Attention to Gold DB ({gold_db_id}) vs. Position in Prompt\n(Query: {safe_qid}, Total DBs: {NUM_DATABASES_IN_PROMPT})')
    plt.xlabel('Position of Gold Database in Prompt')
    plt.ylabel('Total Attention Flow (Sum over all Layers & Heads)')
    plt.xticks(np.arange(1, NUM_DATABASES_IN_PROMPT + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to: {plot_path}")
    print(f"Summary log saved to: {log_path}")
    print(f"Detailed head-by-head log saved to: {detailed_log_path}")
    print(f"\n--- Experiment finished in {time.time() - start_time:.2f} seconds. ---")
       
if __name__ == "__main__":
    main()