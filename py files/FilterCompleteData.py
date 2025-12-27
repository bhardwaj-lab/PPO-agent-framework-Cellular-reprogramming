import scanpy as sc
import anndata as ad
from typing import Tuple, List, Optional
import numpy as np
import scipy.sparse as sp
import pandas as pd
import warnings


class FilterData:
    def __init__(
            self,
            raw_data: sc.AnnData,
            normalized_data: Optional[sc.AnnData] = None,
            celltype_to_remove: Optional[List[str]] = None,
            preferred_dims: Tuple[int, int] = (30000, 3000)
    ):
        """
        Initializes the FilterData class.
        Resulting object always has:
         - .X: Log1p Transformed + Scaled (Mean 0, Var 1)
         - .layers['counts']: Raw integer counts
         - .obs: Preserved from raw_data (plus updated QC metrics)
         - .obsm['X_pca']: PCA coordinates (50 dims)
         - .obsm['X_umap']: UMAP coordinates
        """
        self.raw_data = raw_data
        self.preferred_dims = preferred_dims
        self.celltype_to_remove = celltype_to_remove or []

        # Ensure names are unique
        if not self.raw_data.obs_names.is_unique:
            self.raw_data.obs_names_make_unique()
        if not self.raw_data.var_names.is_unique:
            self.raw_data.var_names_make_unique()

        # 1. Process Data (Filter, Norm, Scale)
        if normalized_data is None:
            self.processed_adata = self._process_from_scratch()
        else:
            self.processed_adata = self._process_sync_existing(normalized_data)

        # 2. Update QC metrics
        # This overwrites 'n_genes_by_counts' etc. with correct values for the subset
        sc.pp.calculate_qc_metrics(
            self.processed_adata,
            layer='counts',
            percent_top=None,
            log1p=False,
            inplace=True
        )

        # 3. Calculate Embeddings (PCA -> Neighbors -> UMAP)
        self._run_dimensionality_reduction()

    def _run_dimensionality_reduction(self):
        """Calculates PCA and UMAP on the processed data."""
        # Safety check: Ensure we don't request more PCs than we have min(cells, genes)
        n_obs, n_vars = self.processed_adata.shape
        n_comps = min(50, n_obs - 1, n_vars - 1)

        if n_comps < 50:
            warnings.warn(f"Dataset too small for 50 PCs. Using {n_comps} components instead.")

        # 1. PCA
        sc.tl.pca(self.processed_adata, n_comps=n_comps)

        # 2. Neighbors (uses X_pca automatically)
        sc.pp.neighbors(self.processed_adata, n_neighbors=50, n_pcs=n_comps)

        # 3. UMAP
        sc.tl.umap(self.processed_adata)

    def _get_keep_indices(self, adata: sc.AnnData) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates masks using raw matrix operations."""
        X = adata.X
        is_sparse = sp.issparse(X)

        # 1. Gene filtering (min_cells=3)
        if is_sparse:
            n_cells_per_gene = np.ravel(X.getnnz(axis=0))
        else:
            n_cells_per_gene = np.ravel((X > 0).sum(axis=0))
        gene_mask = n_cells_per_gene >= 3

        # 2. Cell filtering (min_genes=200)
        if is_sparse:
            n_genes_per_cell = np.ravel(X.getnnz(axis=1))
        else:
            n_genes_per_cell = np.ravel((X > 0).sum(axis=1))
        cell_mask = n_genes_per_cell >= 200

        # 3. MT Check
        mt_prefix = ("MT-", "mt-", "Mt-")
        is_mt = adata.var_names.str.startswith(mt_prefix)

        if is_mt.any():
            if is_sparse:
                # Optimized slicing for sparse matrices to avoid density
                mt_counts = np.ravel(X[:, is_mt].sum(axis=1))
                total_counts = np.ravel(X.sum(axis=1))
            else:
                mt_counts = np.ravel(X[:, is_mt].sum(axis=1))
                total_counts = np.ravel(X.sum(axis=1))

            total_counts[total_counts == 0] = 1
            pct_mt = (mt_counts / total_counts) * 100
            cell_mask = cell_mask & (pct_mt < 10)

        # 4. Celltype Filtering
        if self.celltype_to_remove and "celltype" in adata.obs.columns:
            ct_mask = ~adata.obs["celltype"].isin(self.celltype_to_remove).values
            cell_mask = cell_mask & ct_mask

        return cell_mask, gene_mask

    def _process_from_scratch(self) -> sc.AnnData:
        cell_mask, gene_mask = self._get_keep_indices(self.raw_data)

        # Subsampling valid indices
        valid_indices = np.where(cell_mask)[0]
        if self.preferred_dims[0] < len(valid_indices):
            rng = np.random.default_rng(123)
            selected = rng.choice(valid_indices, size=self.preferred_dims[0], replace=False)
            selected.sort()
        else:
            selected = valid_indices

        # Apply slices
        adata = self.raw_data[selected, gene_mask].copy()

        # Pearson HVG selection
        sc.experimental.pp.highly_variable_genes(
            adata, flavor="pearson_residuals", n_top_genes=self.preferred_dims[1]
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

        # Final Layers & Norm
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)

        return adata

    def _process_sync_existing(self, normalized_data: sc.AnnData) -> sc.AnnData:
        # 1. Alignment
        common_cells = self.raw_data.obs_names.intersection(normalized_data.obs_names)
        common_genes = self.raw_data.var_names.intersection(normalized_data.var_names)

        if len(common_cells) == 0 or len(common_genes) == 0:
            raise ValueError("No intersection found between raw_data and normalized_data.")

        # Create Views
        raw_view = self.raw_data[common_cells, common_genes]
        norm_view = normalized_data[common_cells, common_genes]

        # 2. QC on Raw
        cell_mask, gene_mask = self._get_keep_indices(raw_view)

        # 3. Subsample indices
        valid_indices = np.where(cell_mask)[0]
        if self.preferred_dims[0] < len(valid_indices):
            rng = np.random.default_rng(123)
            selected_indices = rng.choice(valid_indices, size=self.preferred_dims[0], replace=False)
            selected_indices.sort()
        else:
            selected_indices = valid_indices

        # SAFETY FIX: Retrieve specific cell names instead of relying on integer indices
        selected_cell_names = raw_view.obs_names[selected_indices]

        # 4. Pearson HVG (on raw subset) - Copy for contiguous memory
        raw_small = raw_view[selected_cell_names, gene_mask].copy()

        sc.experimental.pp.highly_variable_genes(
            raw_small, flavor="pearson_residuals", n_top_genes=self.preferred_dims[1]
        )

        keep_genes_bool = raw_small.var["highly_variable"]
        final_genes = raw_small.var_names[keep_genes_bool]

        # 5. Assemble final object Slice normalized data using NAMES (Safe)
        final_adata = norm_view[selected_cell_names, final_genes].copy()

        # Add the raw counts layer Ensure raw_small is aligned to final_genes
        final_adata.layers["counts"] = raw_small[:, final_genes].X.copy()

        # 6. Sync Metadata
        cols_to_copy = [c for c in self.raw_data.obs.columns if c not in final_adata.obs.columns]
        if cols_to_copy:
            subset_obs = self.raw_data.obs.loc[final_adata.obs_names, cols_to_copy]
            final_adata.obs = pd.concat([final_adata.obs, subset_obs], axis=1)

        # 7. Scaling
        if final_adata.X.max() > 50:
            warnings.warn(
                "Provided normalized_data appears to be raw counts (Max > 50). Applying Log1p before Scaling.")
            sc.pp.log1p(final_adata)

        sc.pp.scale(final_adata, max_value=10)

        return final_adata

    def get_data(self) -> sc.AnnData:
        return self.processed_adata