import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

EXPERIMENT_PROJECT_DIR = "/scratch/gaurav/in_context_+ve_-ve/attention_analysis_full_prompt_positional_sweep"
NUM_TOP_HEADS_TO_REPORT = 20

def generalize_across_queries(project_dir):
    """
    Analyzes head performance by aggregating results from multiple query-specific data directories.
    """
    data_dirs = [d for d in os.listdir(project_dir) if d.endswith('_gold_focus_data_arrays_30')]
    if not data_dirs:
        print(f"Error: No '*_gold_focus_data_arrays' directories found in '{project_dir}'.")
        print("Please run the main experiment script for a few queries first.")
        return
    print(f"--- Generalizing across {len(data_dirs)} different queries ---")
    is_initialized = False
    master_focus_score = None
    total_runs_aggregated = 0
    for query_dir_name in data_dirs:
        full_path = os.path.join(project_dir, query_dir_name)
        npy_files = [f for f in os.listdir(full_path) if f.endswith('.npy')]
        if not npy_files:
            print(f"Warning: Skipping empty directory '{query_dir_name}'.")
            continue
        print(f"  -> Aggregating {len(npy_files)} runs from {query_dir_name}...")
        if not is_initialized:
            first_map = np.load(os.path.join(full_path, npy_files[0]))
            master_focus_score = np.zeros_like(first_map, dtype=float)
            is_initialized = True
        query_total_score = np.zeros_like(master_focus_score)
        for f in npy_files:
            focus_map = np.load(os.path.join(full_path, f))
            query_total_score += focus_map
        if len(npy_files) > 0:
            normalized_query_score = query_total_score / len(npy_files)
            master_focus_score += normalized_query_score
            total_runs_aggregated += 1

    if total_runs_aggregated == 0:
        print("Error: No valid runs were aggregated.")
        return
    
    average_focus_rate = master_focus_score / total_runs_aggregated

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(average_focus_rate.T, cmap="magma", linewidths=.5, linecolor='gray', annot=False)
    ax.invert_yaxis()
    plt.title(f"Generalized Head Performance: Average Focus Rate on Gold DB\n(Aggregated across {total_runs_aggregated} queries)", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Head Index", fontsize=12)
    output_path = os.path.join(project_dir, f"GENERALIZED_head_performance_heatmap_{total_runs_aggregated}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[SUCCESS] Generalized heatmap saved to: {output_path}")

    print(f"\n--- Top {NUM_TOP_HEADS_TO_REPORT} Most Important Heads for the DB Routing Task (Generalized) ---")
    flat_indices = np.argsort(average_focus_rate, axis=None)[-NUM_TOP_HEADS_TO_REPORT:][::-1]
    for i, flat_idx in enumerate(flat_indices):
        layer_idx, head_idx = np.unravel_index(flat_idx, average_focus_rate.shape)
        rate = average_focus_rate[layer_idx, head_idx]
        print(f"#{i+1}: Layer {layer_idx:02d}, Head {head_idx:02d} --> Average Focus Rate: {rate:.2%}")

if __name__ == "__main__":
    generalize_across_queries(EXPERIMENT_PROJECT_DIR)