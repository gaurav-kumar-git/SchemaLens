import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv

MODEL_PATH = "/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BIRD_DIR = '/scratch/gaurav/data/BIRD'
OUTPUT_DIR = './scaled_attention_elastic_band_plots_analysis_changes_distance_based_m7'
# LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] 
os.makedirs(OUTPUT_DIR, exist_ok=True)
npz_dir=os.path.join(OUTPUT_DIR, "head_analysis_npz")
os.makedirs(npz_dir, exist_ok=True)

QUERIES_TO_AGGREGATE = [345]#, 674]#, 532, 785, 123, 654, 987, 234, 698, 1412]#, 1234, 1517, 867, 543, 512, 789, 432, 436, 124, 567]
QUERIES_TO_PLOT = [532, 785, 123, 654]#, 987]#[367, 567, 745, 1054]#, 1245] 
NUM_DBS = 30

print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

# def get_bird_data():
#     def load_json(p):
#         with open(p, 'r') as f: return json.load(f)
#     data = load_json(os.path.join(BIRD_DIR, 'dev', 'dev.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train.json'))
#     tables = load_json(os.path.join(BIRD_DIR, 'dev', 'dev_tables.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train_tables.json'))
#     return data, 

def get_bird_data():
    def load_json(p):
        with open(p, 'r') as f: return json.load(f)
    data = load_json(os.path.join(BIRD_DIR, 'dev', 'dev.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train.json'))
    tables = load_json(os.path.join(BIRD_DIR, 'dev', 'dev_tables.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train_tables.json'))
    sql_map = load_json('/scratch/gaurav/data/BIRD/formatted/birddb_dev_schema_info.json')
    sql_map.update(load_json('/scratch/gaurav/data/BIRD/formatted/birddb_train_schema_info.json'))
    return data, {db['db_id']: db for db in tables}, sql_map

all_questions, db_schemas, sql_map = get_bird_data()


def apply_manual_rope(q, k, seq_len):
    """
    Hardcoded Llama 3 RoPE implementation. 
    q shape: [1, seq, 32, 128] | k shape: [1, seq, 8, 128]
    """
    dim = 128
    theta = 500000.0 # Llama 3 base
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float().to("cuda") / dim))
    t = torch.arange(seq_len, device="cuda").float()
    freqs = torch.outer(t, inv_freq) # [seq, 64]
    emb = torch.cat((freqs, freqs), dim=-1) # [seq, 128]
    cos = emb.cos().view(1, seq_len, 1, dim)
    sin = emb.sin().view(1, seq_len, 1, dim)

    def rotate_half(x):
        x1, x2 = x[..., :dim//2], x[..., dim//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


query_success_matrices = []   
query_corrected_matrices = [] 
test_query_full_data = {} 

# unique_qids = sorted(list(set(QUERIES_TO_PLOT)))
unique_qids = sorted(list(set(QUERIES_TO_AGGREGATE + QUERIES_TO_PLOT)))




for q_id in unique_qids:
    print(f"\nProcessing QID: {q_id}")
    item = next(i for i in all_questions if str(i.get('question_id')) == str(q_id))
    gold_db = item['db_id']
    nl_query = item['question']
    
    distractors = sorted([d for d in db_schemas.keys() if d != gold_db])[:NUM_DBS-1] # 29 distractors + 1 gold = 30 total (0...28) 
    win_count_raw = np.zeros((32, 32)) 
    win_count_corr = np.zeros((32, 32))
    
    if q_id in QUERIES_TO_PLOT: test_query_full_data[q_id] = {}

    for pos in tqdm(range(NUM_DBS), desc=f"Positional Sweep"):
        ordered_dbs = distractors[:pos] + [gold_db] + distractors[pos:]
        
        
        db_text = ""
        sep = "\n\n----------------------------------------------------------\n\n"
        for i, db in enumerate(ordered_dbs):
            db_text += f"database_id: {db} {sql_map[db]}\n"
            if i < 29: db_text += sep

        # task_s = "# Task: Identify the correct database_id for the question below."
        anchor_text = "# This is a general text"
        query_s = f"# Task: Identify the correct database_id for the question below. QUESTION: {nl_query}"
        # prompt = f"DATABASES:\n{db_text}\n{anchor_text}\n{task_s}{query_s}\nASSISTANT:"
        prompt = f"DATABASES:\n{db_text}\n{anchor_text}\n{query_s}\nASSISTANT:"
        
        
        inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to("cuda")
        offsets = inputs['offset_mapping'][0] 
        # import ipdb; ipdb.set_trace()
        # t_start_c = prompt.find(task_s)
        # q_start_c = prompt.find(task_s)
        q_start_c = prompt.find(query_s)
        t_start_c = prompt.find(query_s)
        a_start_c = prompt.find(anchor_text)
        
        as_idx, ae_idx, qs_idx, qe_idx = -1, -1, -1, -1 # anchor start/end and query start/end token indices
        for i, (s, e) in enumerate(offsets):
            if s >= a_start_c and as_idx == -1: as_idx = i
            if e <= a_start_c + len(anchor_text): ae_idx = i + 1
            if s >= q_start_c and qs_idx == -1: qs_idx = i
            if e <= q_start_c + len(query_s): qe_idx = i + 1
        
        
        n_q = qe_idx - qs_idx 
        n_a = ae_idx - as_idx 

        db_bounds = []
        for i, db in enumerate(ordered_dbs):
            s_char = prompt.find(f"database_id: {db}")
            e_char = prompt.find(sep, s_char) if i < 29 else t_start_c
            s_t, e_t = -1, -1
            for idx, (s, e) in enumerate(offsets):
                if s >= s_char and s_t == -1: s_t = idx
                if e <= e_char: e_t = idx + 1
            db_bounds.append((s_t, e_t))
        # import ipdb; ipdb.set_trace()

        with torch.no_grad():
            out = model(inputs['input_ids'], output_hidden_states=True)
            kv = out.past_key_values 
            seq_len = inputs['input_ids'].shape[1]

            pos_raw_density = np.zeros((30, 32, 32)) # db, layer, head
            pos_corr_density = np.zeros((30, 32, 32))

            for l in range(32):
                layer = model.model.layers[l]
                layer_in = layer.input_layernorm(out.hidden_states[l]) # [1, seq, 4096]
                q_p = layer.self_attn.q_proj(layer_in) # [1, seq, 4096]
                k_p = layer.self_attn.k_proj(layer_in) # [1, seq, 1024]
                
                # q_view: [1, seq, 32, 128] | k_view: [1, seq, 8, 128]
                q_rot, k_rot = apply_manual_rope(q_p.view(1, seq_len, 32, 128), k_p.view(1, seq_len, 8, 128), seq_len)
                
                # q_q: [1, 32, n_q, 128] | a_q: [1, 32, n_a, 128]
                q_q = q_rot[:, qs_idx:qe_idx, :, :].transpose(1, 2) 
                a_q = q_rot[:, as_idx:ae_idx, :, :].transpose(1, 2)
                
                # Full K (Rotated): [1, 32, seq_len, 128]
                full_k = repeat_kv(k_rot.transpose(1, 2), 4)
                
                # Causal Attention Scores: [1, 32, n_q, seq_len] | [1, 32, n_a, seq_len]
                scores_q = torch.matmul(q_q, full_k.transpose(-2, -1)) / (128**0.5)
                scores_a = torch.matmul(a_q, full_k.transpose(-2, -1)) / (128**0.5)
                k_pos = torch.arange(seq_len, device="cuda").view(1, seq_len) 
                q_pos = torch.arange(qs_idx, qe_idx, device="cuda").view(n_q, 1)
                mask_q = k_pos > q_pos
                a_pos = torch.arange(as_idx, ae_idx, device="cuda").view(n_a, 1)
                mask_a = k_pos > a_pos
                scores_q.masked_fill_(mask_q.unsqueeze(0).unsqueeze(0), float("-inf"))
                scores_a.masked_fill_(mask_a.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn_q = torch.nn.functional.softmax(scores_q, dim=-1)
                attn_a = torch.nn.functional.softmax(scores_a, dim=-1)
                
                # This intermediate _ is for Method 6 and 7, where we need to calculate the global mean across DBs before applying the distance-based weighting.
                layer_db_densities = []
                for _db_i, (_ds, _de) in enumerate(db_bounds):
                    _n_db = _de - _ds
                    _q_dens = attn_q[:, :, :, _ds:_de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_q * _n_db)
                    layer_db_densities.append(_q_dens)
                    
                global_mean_attn = np.mean(layer_db_densities, axis=0) 
                avg_db_len = np.mean([b[1] - b[0] for b in db_bounds])
                # end of intermediate _ for Method 6, 7
                
                for db_i, (ds, de) in enumerate(db_bounds):
                    n_db = de - ds
                    q_dens = attn_q[:, :, :, ds:de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_q * n_db) # Average attention density from query tokens to this DB's tokens
                    a_dens = attn_a[:, :, :, ds:de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_a * n_db) # Average attention density from anchor tokens to this DB's tokens
                    
                    #METHOD 0: Simple Subtraction (Baseline)
                    # pos_corr_density[db_i, l, :] = q_dens - a_dens
                    
                    # METHOD 1: Change in log space is equivalent to scaling in normal space, and also prevents negative values which can be tricky to interpret
                    # eps = 1e-9
                    # pos_corr_density[db_i, l, :] = np.log((q_dens + eps) / (a_dens + eps))
                    
                    #  METHOD 2: Dampen the subtraction so the 'background' doesn't overwhelm the 'signal'
                    # lambda_factor = 0.5 
                    # pos_corr_density[db_i, l, :] = q_dens - (lambda_factor * a_dens)
                    
                    # METHOD 3:
                    # 1. Calculate token distance (center of current DB to center of Anchor)
                    # db_mid = (ds + de) / 2
                    # anchor_mid = (as_idx + ae_idx) / 2
                    # dist = abs(anchor_mid - db_mid)
                    # # 2. Parameters: 
                    # # k=100 means 'protection' fades 100 tokens away from anchor
                    # # tau=30 controls how smooth the transition is
                    # k = 600  
                    # tau = 30 
                    # # 3. Calculate Weight (Sigmoid)
                    # # When dist is small (near anchor), weight -> 0. 
                    # # When dist is large (far from anchor), weight -> 1.
                    # weight = 1 / (1 + np.exp(-(dist - k) / tau))
                    # # 4. Apply the weighted subtraction
                    # pos_raw_density[db_i, l, :] = q_dens
                    # pos_corr_density[db_i, l, :] = q_dens - (weight * a_dens)
                    
                    # METHOD 4: Learnable Weighted Subtraction
                    # avg_db_len = np.mean([b[1] - b[0] for b in db_bounds])
                    # avg_attn = 1.0 / 30.0 
                    # smooth = avg_attn * 0.1  # Additive smoothing
                    # for db_i, (ds, de) in enumerate(db_bounds):
                    #     n_db = de - ds
                    #     q_dens = attn_q[:, :, :, ds:de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_q * n_db)
                    #     a_dens = attn_a[:, :, :, ds:de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_a * n_db)
                        
                    #     db_mid = (ds + de) / 2
                    #     anchor_mid = (as_idx + ae_idx) / 2
                    #     dist = abs(anchor_mid - db_mid)
                    #     k = avg_db_len * 1.5 
                    #     tau = 30 
                    #     weight = 1 / (1 + np.exp(-(dist - k) / tau))
                    #     log_ratio = np.log((q_dens + smooth) / ((weight * a_dens) + smooth))
                    #     pos_raw_density[db_i, l, :] = q_dens
                    #     pos_corr_density[db_i, l, :] = log_ratio
                    
                    # METHOD 5 - Distance-Weighted Log Ratio (Combines the intuition of Methods 3, 4, and 1)
                    # for db_i, (ds, de) in enumerate(db_bounds):
                    #     n_db = de - ds
                    #     q_dens = attn_q[:, :, :, ds:de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_q * n_db)
                    #     a_dens = attn_a[:, :, :, ds:de].sum(dim=(2, 3)).squeeze().cpu().float().numpy() / (n_a * n_db)
                    #     db_mid = (ds + de) / 2
                    #     q_mid = (qs_idx + qe_idx) / 2
                    #     a_mid = (as_idx + ae_idx) / 2
                    #     dist_q = abs(q_mid - db_mid) # Distance from Query to DB
                    #     dist_a = abs(a_mid - db_mid) # Distance from Anchor to DB
                    #     gamma = 0.8
                    #     q_boosted = q_dens * (dist_q ** gamma)
                    #     a_boosted = a_dens * (dist_a ** gamma)
                    #     smooth = 0.05 # A slightly larger smooth helps stabilize the power-law
                    #     log_ratio = np.log((q_boosted + smooth) / (a_boosted + smooth))
                    #     pos_raw_density[db_i, l, :] = q_dens
                    #     pos_corr_density[db_i, l, :] = log_ratio
                    #     import ipdb; ipdb.set_trace()
                    
                    # METHOD 6
                    # db_mid = (ds + de) / 2
                    # q_mid = (qs_idx + qe_idx) / 2
                    # a_mid = (as_idx + ae_idx) / 2
                    # dist_q = abs(q_mid - db_mid)
                    # dist_a = abs(a_mid - db_mid)
                    # gamma = 1e-15  # Decay correction
                    # k = avg_db_len * 2 # Neighborhood Shield radius
                    # tau = 30 # Transition smoothness
                    # weight = 1 / (1 + np.exp(-(dist_a - k) / tau))
                    # q_signal = q_dens * (dist_q ** gamma)
                    # hybrid_baseline = (weight * a_dens * (dist_a ** gamma)) + ((1 - weight) * global_mean_attn)
                    # smooth = 0.0001 
                    # log_ratio = np.log((q_signal + smooth) / (hybrid_baseline + smooth))

                    # pos_raw_density[db_i, l, :] = q_dens
                    # pos_corr_density[db_i, l, :] = log_ratio
                    
                    # METHOD 7
                    db_mid = (ds + de) / 2
                    q_mid = (qs_idx + qe_idx) / 2
                    a_mid = (as_idx + ae_idx) / 2
                    dist_q = abs(q_mid - db_mid)
                    dist_a = abs(a_mid - db_mid)
                    gamma = 0.5
                    k = avg_db_len * 2 # Neighborhood Shield radius
                    tau = 30 # Transition smoothness
                    dist_q_boost = (dist_q ** gamma)
                    dist_a_boost = (dist_a ** gamma)
                    q_signal = q_dens * dist_q_boost
                    a_signal = a_dens * dist_a_boost
                    weight = 1 / (1 + np.exp(-(dist_a - k) / tau))
                    boosted_global_mean = global_mean_attn * dist_q_boost
                    hybrid_baseline = (weight * a_signal) + ((1 - weight) * boosted_global_mean)
                    # surplus_attention = q_signal - hybrid_baseline
                    surplus_attention = q_signal - (weight * a_signal)
                    pos_raw_density[db_i, l, :] = q_dens
                    pos_corr_density[db_i, l, :] = surplus_attention
                
            gold_i = ordered_dbs.index(gold_db)
            for l in range(32):
                for h in range(32):
                    if np.argmax(pos_raw_density[:, l, h]) == gold_i: win_count_raw[l, h] += 1
                    if np.argmax(pos_corr_density[:, l, h]) == gold_i: win_count_corr[l, h] += 1
            
            if q_id in QUERIES_TO_PLOT:
                others = [i for i in range(30) if i != gold_i]
                test_query_full_data[q_id][pos] = {
                    'rg': pos_raw_density[gold_i], 'rm': np.max(pos_raw_density[others], axis=0),
                    'cg': pos_corr_density[gold_i], 'cm': np.max(pos_corr_density[others], axis=0)
                } 
                
    current_save_path = os.path.join(npz_dir, f"qid_{q_id}.npz")
    np.savez(
        current_save_path,
        q_id=np.array([q_id]),
        pos_raw_density=pos_raw_density,
        pos_corr_density=pos_corr_density,
        win_count_raw=win_count_raw,
        win_count_corr=win_count_corr,
    )
    
    if q_id in QUERIES_TO_AGGREGATE:
        query_success_matrices.append(win_count_raw / 30.0) 
        query_corrected_matrices.append(win_count_corr / 30.0)
        
num_queries_processed = len(query_success_matrices)

if num_queries_processed > 0:
    final_raw_map = np.sum(query_success_matrices, axis=0) / num_queries_processed
    final_corr_map = np.sum(query_corrected_matrices, axis=0) / num_queries_processed
else:
    final_raw_map = np.zeros((32, 32))
    final_corr_map = np.zeros((32, 32))
    
def get_top_20(matrix):
    sorted_idx = np.argsort(matrix.flatten())[-20:][::-1]
    return [divmod(i,   32) for i in sorted_idx]

top_raw = get_top_20(final_raw_map)
top_corr = get_top_20(final_corr_map)

final_save_path = os.path.join(npz_dir, f"aggregated_results.npz")
np.savez(
        final_save_path,
        final_raw_map=final_raw_map,
        final_corr_map=final_corr_map,
        top_raw=np.array(top_raw),
        top_corr=np.array(top_corr)
    )

x_axis = np.arange(1, 31)
for q_id in QUERIES_TO_PLOT:
    q_dir = os.path.join(OUTPUT_DIR, f"qid_{q_id}")
    os.makedirs(q_dir, exist_ok=True)
    
    def plot_helper(heads, is_case_3, title, fname):
        yg, ym = [], []
        for p in range(30):
            d = test_query_full_data[q_id][p]
            gold_map, dist_map = (d['cg'], d['cm']) if is_case_3 else (d['rg'], d['rm'])
            if heads is None: # Case 1
                yg.append(np.sum(gold_map)); ym.append(np.sum(dist_map))
            else: # Case 2 & 3
                yg.append(sum(gold_map[l,h] for l,h in heads))
                ym.append(sum(dist_map[l,h] for l,h in heads))
        
        plt.figure(figsize=(9, 5))
        plt.plot(x_axis, yg, 'b-o', label='Gold'); plt.plot(x_axis, ym, 'r--x', label='Max Dist')
        plt.title(f"{title} (QID {q_id})"); plt.xlabel("Gold Pos"); plt.ylabel("Attention Share")
        plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(os.path.join(q_out := q_dir, fname)); plt.close()

    plot_helper(None, False, "Case 1: All Heads", "01_all.png")
    plot_helper(top_raw, False, "Case 2: Top 20 Raw", "02_success.png")
    plot_helper(top_corr, True, "Case 3: Top 20 Corrected", "03_corrected.png")