import pickle
import numpy as np
import cupy as cp
from typing import Dict
from scipy.sparse import issparse
from typing import List
def createTransitionMatrix(path_cellOc:str, output_path:str, allow_OE:bool=True):
    with open(path_cellOc, 'rb') as f:
        oracle = pickle.load(f)
    if not oracle:
        raise ValueError("CellOracle object could not be loaded from the specified path.")
        return
    genes_that_can_be_perturbed = oracle.return_active_reg_genes()
    total_genes_perturb = len(genes_that_can_be_perturbed) * (2 if allow_OE else 1)
    n_cells = oracle.adata.n_obs
    all_idx = np.arange(n_cells)
    transition_matrix = np.zeros((n_cells, total_genes_perturb), dtype=np.int32)
    #check if gpu is present:
    gpu_present  = False
    try:
        cp.cuda.Device(0).use()
        gpu_present = True
        print("CuPy is available and GPU is accessible.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        # Handle the case where CuPy cannot access the GPU
        print("CuPy cannot access GPU, falling back to CPU integration.")
        gpu_present = False
    #so for each action perturb all cells in the celloracle dataset

    high_ranges_dict = _calculate_gene_activation_values(oracle,genes_that_can_be_perturbed,1.5)
    for action_idx in range(total_genes_perturb):
        perturb_conditions = generate_perturb_condition(action_idx,genes_that_can_be_perturbed,high_ranges_dict)
        new_idx_list = []
        if gpu_present:
            # Use cupy for GPU acceleration
            _, new_idx_list, _, _, _ = oracle.training_phase_inference_batch_cp(
                batch_size=total_genes_perturb, idxs=all_idx.tolist(),
                perturb_condition=perturb_conditions, n_neighbors=200,
                n_propagation=3, threads=4,
                knockout_prev_used=False)
        else:
            _, new_idx_list, _, _, _ = oracle.training_phase_inference_batch(
                batch_size=total_genes_perturb, idxs=all_idx.tolist(),
                perturb_condition=perturb_conditions, n_neighbors=200, n_propagation=3, threads=4,knockout_prev_used=False)

        transition_matrix[:, action_idx] = new_idx_list

    #save a pkl of the transiton matrix
    with open(output_path, 'wb') as f:
        pickle.dump(transition_matrix, f)


def generate_perturb_condition(self, action_idx: int, genes_that_can_be_perturbed: list, high_ranges_dict:dict)->List[tuple]:
    perturb_conditions = []
    if action_idx < len(genes_that_can_be_perturbed):
        # does not matter if we allow gene activation or not, we just set the value to 0.0
        gene_to_knockout = genes_that_can_be_perturbed[action_idx]
        perturb_conditions.append((gene_to_knockout, 0.0))  # Assuming knockout value is always 0.0
        return perturb_conditions
    if action_idx > 2 * self.number_of_reg_genes:
        print("WARNING THIS SHOULD NOT HAPPEN MORE ACTIONS THAN 2*REG GENES")
        raise ValueError("Action index out of bounds for gene activation.")
    gene_to_activate = genes_that_can_be_perturbed[action_idx % len(genes_that_can_be_perturbed)]
    # Get the activation value from the high ranges dict
    activation_value = high_ranges_dict[gene_to_activate]
    perturb_conditions.append((gene_to_activate, activation_value))
    return perturb_conditions

def _calculate_gene_activation_values(oracle, genes_that_can_be_perturbed, sd_factor: float) -> Dict[str, float]:
    activation_values_dict = {}
    perturb_indices_in_adata = [oracle.adata.var.index.get_loc(g) for g in genes_that_can_be_perturbed if
                                g in oracle.adata.var.index]
    expression_data = oracle.adata[:, perturb_indices_in_adata].X
    if issparse(expression_data): expression_data = expression_data.toarray()

    for i, gene_name in enumerate(genes_that_can_be_perturbed):
        if gene_name not in oracle.adata.var.index: continue
        gene_expr = expression_data[:, i]
        q1, q3 = np.percentile(gene_expr, [25, 75])
        iqr = q3 - q1
        filtered_expr = gene_expr[(gene_expr >= q1 - 3 * iqr) & (gene_expr <= q3 + 3 * iqr)]
        activation_val = np.mean(filtered_expr) + sd_factor * np.std(filtered_expr)
        activation_values_dict[gene_name] = max(0.0, activation_val)
    return activation_values_dict




