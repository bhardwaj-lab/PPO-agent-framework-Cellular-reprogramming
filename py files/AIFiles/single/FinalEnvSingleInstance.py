# File: cell_oracle_gym_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os
from typing import Any, Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import math


class CellOracleGymEnv(gym.Env):
    """
    A standard Gymnasium environment for cell reprogramming, designed to be a
    single-instance equivalent of the custom CellOracleSB3VecEnv.

    This version has a unified observation space and curriculum logic to ensure
    full compatibility with agents trained in the custom VecEnv. An agent
    trained with this environment (wrapped in an SB3 VecEnv like DummyVecEnv)
    will be directly compatible with the custom CellOracleSB3VecEnv.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 adata_object: object,
                 oracle_perturbable_genes: List[str],
                 transition_matrix_object: np.ndarray,
                 max_steps: int = 50,
                 step_penalty: float = -0.01,
                 goal_bonus: float = 1.0,
                 distance_reward_scale: float = 1.0,
                 discount_factor: float = 0.99,
                 allow_gene_activation: bool = True,
                 number_of_targets_curriculum: int = 4,
                 val_frac: float = 0.05,
                 test_frac: int = 0.05,
                 add_noise: bool = False,
                 noise_rate: float = 0.3,
                 seed: int = 77,
                 step_increase_per_phase: int = 6,
                 pgc_only_curriculum: bool = False):

        super().__init__()

        #crucial data members
        self.np_random = np.random.default_rng(seed)
        self.adata = adata_object
        self.genes_that_can_be_perturbed = oracle_perturbable_genes
        self.transition_matrix = transition_matrix_object

        #declare general bio params
        self.allow_gene_activation = allow_gene_activation
        self.n_cells = self.adata.n_obs
        self.all_genes = self.adata.var.index.tolist()
        self.number_of_reg_genes = len(self.genes_that_can_be_perturbed)
        self.celltypes = self.adata.obs['celltype'].unique().tolist()
        self.action_space_size = self.number_of_reg_genes * (2 if self.allow_gene_activation else 1)
        self.step_increase_per_phase = step_increase_per_phase
        print("IT IS RELOADED!")


        # make a very conservative train test split on test size side as wel want most cells to train on but still some test cells to evaluate on, we take 1%
        self.train_indices,self.val_indices, self.test_indices = self._get_all_indices_split(val_frac=val_frac, test_frac=test_frac)
        self.forbidden_indices = np.concatenate((self.val_indices, self.test_indices))


        self.AI_input = self._normalize_expression_counts()

        #Define general Env params
        self.action_space = spaces.Discrete(self.action_space_size)
        self.observation_space = self._create_observation_space()
        self.max_steps = max_steps
        self.add_noise = add_noise
        self.noise_rate = noise_rate
        self.eval_mode = False
        self.test_mode = False

        


        self.average_full_expression_vectors = self._precompute_average_full_expression_vectors()
        self.termination_thresholds = self._precompute_termination_thresholds()

        # start cell indices are pgc cells only from train set
        allowed_starter_celltypes = None
        allowed_start_indices = None
        if pgc_only_curriculum:
            allowed_starter_celltypes = [ct for ct in self.celltypes if 'PGC' in ct]
            pgc_mask = self.adata.obs['celltype'].str.contains("PGCs", na=False)
            allowed_start_indices = np.where(pgc_mask)[0]
        else:
            allowed_starter_celltypes = self.celltypes
            allowed_start_indices = np.arange(self.n_cells)

        self.n_available_start_types = len(allowed_starter_celltypes)
        self.start_cell_indices = np.intersect1d(allowed_start_indices, self.train_indices)
        self._original_start_indices = self.start_cell_indices.copy()

        self.reg_gene_adata_indices = np.array([self.all_genes.index(g) for g in self.genes_that_can_be_perturbed])
        self.total_curriculum_targets = self._compute_total_curriculum_targets(number_of_targets_curriculum,allowed_starter_celltypes, self.celltypes)
        self.n_available_phases = len(allowed_starter_celltypes)
        # compute average expressoin vector per celltype for target purposes and the temrination threshold based on this


        # --- Reward params ---
        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.distance_reward_scale = distance_reward_scale
        self.discount_factor = discount_factor

        # State variables for a SINGLE episode
        self.current_step = 0
        self.current_cell_index = 0
        self.current_cell_state = None
        self.target_cell_type = ""
        self.current_phase = 1

    def _get_all_indices_split(self, val_frac = 0.035, test_frac: float = 0.015) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if os.path.exists("train_indices.pkl") and os.path.exists("val_indices.pkl") and os.path.exists("test_indices.pkl"):
            with open("train_indices.pkl", "rb") as f:
                train_indices = pickle.load(f)
            with open("val_indices.pkl", "rb") as f:
                val_indices = pickle.load(f)
            with open("test_indices.pkl", "rb") as f:
                test_indices = pickle.load(f)
            return train_indices, val_indices, test_indices

        train_indices, val_indices, test_indices = self._create_start_and_test_indices(val_frac=val_frac, test_frac=test_frac)
        with open("train_indices.pkl", "wb") as f:
            pickle.dump(train_indices, f)
        with open("val_indices.pkl", "wb") as f:
            pickle.dump(val_indices, f)
        with open("test_indices.pkl", "wb") as f:
            pickle.dump(test_indices, f)
        return train_indices, val_indices, test_indices

    def _create_start_and_test_indices(self, val_frac = 0.035, test_frac: float = 0.015) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_indices_list = []
        test_indices_list = []
        val_indices_list = []
        all_indices = np.arange(self.adata.n_obs)
        for celltype in self.celltypes:
            celltype_mask = self.adata.obs['celltype'] == celltype
            celltype_indices = all_indices[celltype_mask]
            #shuffled indices just to make sure randomality
            self.np_random.shuffle(celltype_indices)
            n_test = math.ceil(len(celltype_indices) * test_frac)
            n_val = math.ceil(len(celltype_indices) * val_frac)
            #select n_test at random
            test_indices = celltype_indices[:n_test]
            val_indices = celltype_indices[n_test:n_test+n_val]
            train_indices = celltype_indices[n_test+n_val:]
            train_indices_list.extend(train_indices)
            test_indices_list.extend(test_indices)
            val_indices_list.extend(val_indices)
        return np.array(train_indices_list), np.array(val_indices_list), np.array(test_indices_list)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.episode_actions = []
        self.episode_path_nodes = []
        if self.eval_mode and self.fixed_eval_set is not None:
            if self.eval_eps_counter >= len(self.fixed_eval_set):
                self.eval_eps_counter = 0
            start_idx , target_type = self.fixed_eval_set[self.eval_eps_counter]
            self.current_cell_index = start_idx
            self.current_cell_state = self._get_expression_vector_for_index(self.current_cell_index)
            self.target_cell_type = target_type
            self.eval_eps_counter += 1
            self.episode_path_nodes = [self.current_cell_index]
            self.current_step = 0
            self.initial_dist = self._calculate_cosine_dist_to_target(self.current_cell_index)
            self.start_node_for_episode = self.current_cell_index
            self.target_celltype_for_episode = self.target_cell_type
            info = {"initial_distance": self.initial_dist}
            return self._get_obs(), info

        if self.test_mode and self.fixed_test is not None:
            if self.test_eps_counter >= len(self.fixed_test):
                self.test_eps_counter = 0
            start_idx , target_type = self.fixed_test[self.test_eps_counter]
            self.current_cell_index = start_idx
            self.current_cell_state = self._get_expression_vector_for_index(self.current_cell_index)
            self.target_cell_type = target_type
            self.test_eps_counter += 1
            self.current_step = 0
            self.episode_path_nodes = [self.current_cell_index]
            self.initial_dist = self._calculate_cosine_dist_to_target(self.current_cell_index)
            self.start_node_for_episode = self.current_cell_index
            self.target_celltype_for_episode = self.target_cell_type
            info = {"initial_distance": self.initial_dist}
            return self._get_obs(), info

        self.current_step = 0
        self.current_cell_index = self.np_random.choice(self.start_cell_indices)
        self.current_cell_state = self._get_expression_vector_for_index(self.current_cell_index)
        self.target_cell_type = self._get_single_target_type(self.current_cell_index)
        self.episode_path_nodes = [self.current_cell_index]
        self.start_node_for_episode = self.current_cell_index
        self.target_celltype_for_episode = self.target_cell_type
        obs = self._get_obs()
        self.initial_dist = self._calculate_cosine_dist_to_target(self.current_cell_index)
        info = {"initial_distance": self.initial_dist}


        return obs, info

    def _calculate_cosine_dist_to_target(self, cell_index: int) -> float:
        """Helper to calculate Cosine Distance for the current target."""
        raw_vec = self.current_cell_state.reshape(1, -1)
        target_avg_vector = self.average_full_expression_vectors[self.target_cell_type].reshape(1, -1)

        return pairwise_distances(raw_vec, target_avg_vector, metric='cosine')[0, 0]

    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        self.current_step += 1
        previous_cell_index = self.current_cell_index
        previous_cell_state = self.current_cell_state
        self.current_cell_index = self.transition_matrix[previous_cell_index, action]
        self.current_cell_state = self._get_expression_vector_for_index(self.current_cell_index)
        stagnated = bool(self.current_cell_index == previous_cell_index)
        self.episode_path_nodes.append(self.current_cell_index)
        self.episode_actions.append(action)


        pbrs_reward, cosine_sim_before, consine_sim_after = self._calculate_pbrs_reward(previous_cell_state, self.current_cell_state)
        dist_before = 1-cosine_sim_before
        dist_after = 1-consine_sim_after
        progress_made = dist_before - dist_after
        reward = self.step_penalty + pbrs_reward
        goal_reached = self._is_goal_reached(self.current_cell_index)
        timeout = self.current_step >= self.max_steps
        terminated = bool(goal_reached)
        truncated = bool(timeout and not terminated)

        if terminated:
            reward += self.goal_bonus

        obs = self._get_obs()

        info = {"goal_reached": terminated, "stagnated": stagnated,"initial_distance" : self.initial_dist ,"progress_made": progress_made,
                "reward_components":{
            "pbrs": pbrs_reward,
            "step": self.step_penalty
        }}
        if terminated or truncated:
            info["episode_actions"] = self.episode_actions
            info["episode_path_nodes"] = self.episode_path_nodes

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Generates an observation matching the custom VecEnv's structure.
        """
        # 1. Current state (gene expression)
        target_full_vector = self.average_full_expression_vectors[self.target_cell_type]

        return {
            "current_expression": self.current_cell_state,
            "target_expression": target_full_vector,
        }

    def _is_goal_reached(self, cell_index: int) -> bool:
        is_correct_type = self.adata.obs['celltype'].iloc[cell_index] == self.target_cell_type
        if not is_correct_type:
            return False

        current_vector = self.current_cell_state.reshape(1, -1)
        target_avg_vector = self.average_full_expression_vectors[self.target_cell_type].reshape(1, -1)

        distance = pairwise_distances(current_vector, target_avg_vector, metric='cosine')[0, 0]
        threshold = self.termination_thresholds[self.target_cell_type]

        return distance <= threshold

    def _calculate_pbrs_reward(self, prev_state: np.ndarray, next_state: np.ndarray) -> Tuple[float, float, float]:
        potential_before = self._calculate_cosine_sim_to_target(prev_state)
        potential_next = self._calculate_cosine_sim_to_target(next_state)
        # Apply the PBRS formula: F(s, s') = γ * Φ(s') - Φ(s)
        pbrs = (self.discount_factor * potential_next) - potential_before
        return pbrs * self.distance_reward_scale, potential_before, potential_next

    def _calculate_umap_dist_to_target(self, cell_index: int) -> float:
        current_coord = self.umap_coords_normalized[cell_index]
        target_coord = self.average_normalized_umap_coordinates[self.target_cell_type]
        return np.linalg.norm(current_coord - target_coord)

    def _calculate_cosine_sim_to_target(self, cell_state: np.ndarray) -> float:
        """Calc cosine sim as pbrs potential function."""
        if cell_state.ndim == 1:
            cell_state = cell_state.reshape(1, -1)
        target_avg_vector = self.average_full_expression_vectors[self.target_cell_type].reshape(1, -1)
        return cosine_similarity(cell_state, target_avg_vector)[0, 0]

    def _create_observation_space(self) -> spaces.Dict:
        """
        Creates a unified observation space matching the custom VecEnv.
        """
        num_genes = self.adata.n_vars
        z_score_bound = 100
        return spaces.Dict({
            "current_expression": spaces.Box(low=-z_score_bound, high=z_score_bound, shape=(num_genes,),
                                             dtype=np.float32),
            "target_expression": spaces.Box(low=-z_score_bound, high=z_score_bound, shape=(num_genes,),
                                            dtype=np.float32),
        })

    def _get_single_target_type(self, start_cell_index: int) -> str:
        """Choose a target cell type based on the current curriculum phase."""
        start_cell_type = self.adata.obs['celltype'].iloc[start_cell_index]
        if self.test_mode:
            available_targets = [ct for ct in self.celltypes if ct != start_cell_type]
        else:
            available_targets = self.total_curriculum_targets[self.current_phase - 1].get(start_cell_type, [])

        #once in like every 10000 random chance we print the number of available targets
        if self.np_random.random() < 0.0001:
            print(f"Available targets for start cell type '{start_cell_type}' in phase {self.current_phase}: {available_targets}")

        if not available_targets:
            # Fallback for non-PGC start cells or if curriculum is misconfigured
            pgc_celltypes = [ct for ct in self.celltypes if 'PGC' in ct]
            fallback_targets = [ct for ct in self.celltypes if ct not in pgc_celltypes]
            if not fallback_targets: return start_cell_type  # Edge case
            return self.np_random.choice(fallback_targets)

        return self.np_random.choice(available_targets)

    def set_phase(self, new_phase: int, step_increase: int) -> bool:
        if new_phase > len(self.total_curriculum_targets) or new_phase <= self.current_phase:
            return False

        phase_diff = new_phase - self.current_phase
        if not (self.eval_mode or self.test_mode):
            self.max_steps += self.step_increase_per_phase * phase_diff
        self.current_phase = new_phase
        print(f"INFO: Env advanced to Curriculum Phase {self.current_phase}. Max steps now: {self.max_steps}")
        return True

    def get_phase(self) -> int:
        return self.current_phase

    def _load_transition_data(self, transition_matrix_path: str):
        with open(transition_matrix_path, 'rb') as f:
            transition_matrix = pickle.load(f)
        return transition_matrix

    def _compute_total_curriculum_targets_from_pgcs(self, number_of_targets_per_phase: int) -> List[
        Dict[str, List[str]]]:
        """
        CRITICAL CHANGE: This curriculum logic matches the VecEnv.
        It computes the nearest non-PGC targets, but only for PGC starting cell types.
        """
        num_phases = math.ceil((len(self.celltypes) - 3) / number_of_targets_per_phase) + 1  # -3 for num PGCs
        all_phase_targets = []
        pgc_celltypes = [ct for ct in self.celltypes if 'PGC' in ct]

        for phase in range(1, num_phases + 1):
            num_targets_for_this_phase = min(
                (phase * number_of_targets_per_phase),
                len(self.celltypes) - len(pgc_celltypes)
            )

            phase_dict = {}
            for start_celltype in pgc_celltypes:
                distances = []
                start_coord = self.average_normalized_umap_coordinates[start_celltype]
                for target_celltype in self.celltypes:
                    if target_celltype in pgc_celltypes: continue  # Only target non-PGCs

                    target_coord = self.average_normalized_umap_coordinates[target_celltype]
                    dist = np.linalg.norm(start_coord - target_coord)
                    distances.append((target_celltype, dist))

                distances.sort(key=lambda x: x[1])
                phase_dict[start_celltype] = [ct for ct, dist in distances[:num_targets_for_this_phase]]
            all_phase_targets.append(phase_dict)

        return all_phase_targets

    def _compute_total_curriculum_targets(self, number_of_targets_per_phase: int, start_types: List[str],
                                          target_pool: List[str]) -> List[Dict[str, List[str]]]:
        """
        Computes a curriculum by finding the N nearest neighbors for a given set of start types.

        Args:
            number_of_targets_per_phase: How many new targets to add per curriculum phase.
            start_types: A list of cell types to build the curriculum for (e.g., just PGCs or all types).
            target_pool: A list of valid cell types that can be targeted.
        """
        max_targets = len(target_pool) - 1

        num_phases = math.ceil(max_targets / number_of_targets_per_phase)
        all_phase_targets = []
        for phase in range(1, num_phases + 1):
            num_targets_for_this_phase = min(
                phase * number_of_targets_per_phase,
                max_targets
            )

            phase_dict = {}
            for start_celltype in start_types:
                distances = []
                start_vec = self.average_full_expression_vectors[start_celltype].reshape(1, -1)

                for target_celltype in target_pool:
                    if start_celltype == target_celltype:
                        continue

                    target_vec = self.average_full_expression_vectors[target_celltype].reshape(1, -1)
                    dist = pairwise_distances(start_vec, target_vec, metric='cosine')[0, 0]
                    distances.append((target_celltype, dist))

                distances.sort(key=lambda x: x[1])

                phase_dict[start_celltype] = [ct for ct, dist in distances[:num_targets_for_this_phase]]

            all_phase_targets.append(phase_dict)

        return all_phase_targets

    def _precompute_average_full_expression_vectors(self, outlier_percentile: int = 95) -> Dict[str, np.ndarray]:
        """
        Pre-computes and stores the average FULL expression vector for each cell type. only do it on the closest 95% to avoid outliers that skew results
        """
        average_vectors = {}
        for celltype in self.celltypes:
            cell_indices = np.where(self.adata.obs['celltype'] == celltype)[0]
            cell_indices = np.intersect1d(cell_indices, self.train_indices)
            if len(cell_indices) <= 0:
                ValueError(f"No cells found for cell type '{celltype}' in the dataset.")

            all_vectors_for_celltype = self._get_expression_vector_for_indices(cell_indices)
            temp_avg = np.mean(all_vectors_for_celltype, axis=0).reshape(1, -1)
            distances = pairwise_distances(all_vectors_for_celltype, temp_avg, metric='cosine').flatten()
            distance_cutoff = np.percentile(distances, outlier_percentile)
            core_cells_mask = distances <= distance_cutoff
            core_vectors = all_vectors_for_celltype[core_cells_mask]
            average_vectors[celltype] = np.mean(core_vectors, axis=0)
        return average_vectors

    def _precompute_termination_thresholds(self, termination_percentile: int = 25) -> Dict[str, float]:
        thresholds = {}
        if self.average_full_expression_vectors is None:
            raise ValueError("Average full expression vectors have not been computed.")
        for celltype in self.celltypes:
            avg_vector = self.average_full_expression_vectors.get(celltype)
            if avg_vector is None:
                continue
            avg_vector = avg_vector.reshape(1, -1)

            cell_indices = np.where(self.adata.obs['celltype'] == celltype)[0]
            cell_indices = np.intersect1d(cell_indices, self.train_indices)

            if len(cell_indices) == 0:
                continue
            all_vectors_for_celltype = self._get_expression_vector_for_indices(cell_indices)
            all_distances = pairwise_distances(all_vectors_for_celltype, avg_vector, metric='cosine').flatten()


            if all_distances.shape[0] > 0:
                thresholds[celltype] = np.percentile(all_distances, termination_percentile)
            else:
                raise ValueError(f"No core distances found for cell type '{celltype}' after outlier removal.")

        return thresholds

    def _normalize_expression_counts(self) -> np.ndarray:
        raw_data = self.adata.layers["imputed_count"][self.train_indices]
        #check sparisty
        if sp.issparse(raw_data):
            raw_data = raw_data.toarray()
        #print the min and max of the raw data
        self.standardScaler = StandardScaler()
        self.standardScaler.fit(raw_data)

        all_input_data = self.adata.layers["imputed_count"]
        if sp.issparse(all_input_data):
            all_input_data = all_input_data.toarray()
        AI_input = self.standardScaler.transform(all_input_data)
        AI_input_np = np.asarray(AI_input, dtype=np.float32)
        self.adata.X = AI_input
        return AI_input_np


    def _get_expression_vector_for_index(self, index: int) -> np.ndarray:
        """Get expression for index, if index is equal to current cell index just return the current cell state"""
        data = self.AI_input[index]
        if self.add_noise and self.np_random.random() < self.noise_rate and self.eval_mode==False and self.test_mode==False:
            fractional_noise = 0.05
            noise = self.np_random.normal(0, fractional_noise, size=data.shape)
            return (data+ noise).squeeze()
        return data.squeeze()

    def _get_expression_vector_for_indices(self, indices: List[int]) -> np.ndarray:
        """Efficiently gets expression vectors for specified indices and genes."""
        data = self.AI_input[indices]
        if self.add_noise and self.np_random.random() < self.noise_rate and self.eval_mode==False and self.test_mode==False:
            fractional_noise = 0.05
            noise = self.np_random.normal(0, fractional_noise, size=data.shape)
            return np.asarray(data) + noise
        return data

    def close(self):
        """Clean up any resources if needed."""
        pass

    def get_env_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the environment relevant for curriculum learning.
        This is called by the CustomCheckpointCallbackWithStates.
        """
        return {
            "max_steps": self.max_steps,
            "current_phase": self.current_phase,
        }

    def switch_between_modes(self, is_eval_mode: bool):
        """Switch start to val indices to evalute for optuna and just general monitoring"""
        if is_eval_mode:
            print("EVAL MODE INITIATED")
            self.start_cell_indices = self.val_indices
            self.eval_mode = True
            val_obs, val_starts, val_targets = self.create_val_set_for_eval(n_episodes=1000)
            self.fixed_eval_set = list(zip(val_starts, val_targets))
            self.eval_eps_counter = 0
        else:
            print("TRAIN MODE INITIATED")
            if hasattr(self, '_original_start_indices'):
                self.start_cell_indices = self._original_start_indices
            self.eval_mode = False

    def set_to_max_phase(self):
        """Set curriculum to max phase for eval purposes"""
        self.current_phase = len(self.total_curriculum_targets)
        print(f"Curriculum phase set to max phase: {self.current_phase}")

    def switch_to_test_mode(self, test_mode_on: bool):
        """Switch start to test indices to evalute final performance"""
        print("TEST MODE INITIATED")
        self.np_random = np.random.default_rng(77)
        if test_mode_on:
            self.start_cell_indices = self.test_indices
            self.test_mode = True
            test_obs, test_starts, test_targets = self.create_test_set(n_episodes=2000)
            self.fixed_test = list(zip(test_starts, test_targets))
            self.test_eps_counter = 0
        else:
            if hasattr(self, '_original_start_indices'):
                self.start_cell_indices = self._original_start_indices
            self.test_mode = False

    def create_val_set_for_eval(self, n_episodes:int = 1000) -> Tuple[List[Dict[str, np.ndarray]], List[int], List[str]]:
        """this returns obs from validation set combined with different target , used externally by model.predict for eval purpsoes, not internally handled as done through the optuna callback wiht sb3 model.evaluate func, we make from the validation ste 1000 start points"""
        val_set_obs = []
        n_episodes = int(n_episodes)
        n_cells = len(self.val_indices)
        n_cells = int(n_cells)
        base_episodes_per_cell = n_episodes // n_cells
        remainder = n_episodes % n_cells
        episode_start_indices = []
        current_episode_count = 0
        chosen_targets = []
        for i, cell_idx in enumerate(self.val_indices):
            targets_to_generate = base_episodes_per_cell
            if i < remainder:
                targets_to_generate += 1
            if current_episode_count >= n_episodes or targets_to_generate == 0:
                break
            targets_to_generate = min(targets_to_generate, n_episodes - current_episode_count)
            start_cell_type = self.adata.obs['celltype'].iloc[cell_idx]
            possible_targets = self.total_curriculum_targets[self.current_phase - 1].get(start_cell_type, [])
            selected_targets = self.np_random.choice(
                possible_targets,
                size=targets_to_generate,
                replace=True
            )
            chosen_targets.extend(selected_targets)
            current_full_vector = self._get_expression_vector_for_index(cell_idx)
            for target in selected_targets:
                target_full_vector = self.average_full_expression_vectors[target]
                obs = {
                    "current_expression": current_full_vector,
                    "target_expression": target_full_vector,
                }
                val_set_obs.append(obs)
                episode_start_indices.append(cell_idx)
            current_episode_count += targets_to_generate
        return val_set_obs, episode_start_indices, chosen_targets


    def create_test_set(self, n_episodes:int = 2000) -> Tuple[List[Dict[str, np.ndarray]], List[int], List[str]]:
        """this returns obs from test set combined with different target , used externally by model.predict for eval purpsoes, not internally handled as done through the optuna callback wiht sb3 model.evaluate func, we make from the validation ste 1000 start points"""
        val_set_obs = []
        n_episodes = int(n_episodes)
        n_cells = len(self.test_indices)
        n_cells = int(n_cells)
        base_episodes_per_cell = n_episodes // n_cells
        remainder = n_episodes % n_cells
        episode_start_indices = []
        current_episode_count = 0
        chosen_targets = []
        for i, cell_idx in enumerate(self.test_indices):
            targets_to_generate = base_episodes_per_cell
            if i < remainder:
                targets_to_generate += 1
            if current_episode_count >= n_episodes or targets_to_generate == 0:
                break
            targets_to_generate = min(targets_to_generate, n_episodes - current_episode_count)
            start_cell_type = self.adata.obs['celltype'].iloc[cell_idx]
            possible_targets = [ct for ct in self.celltypes if ct != start_cell_type]
            selected_targets = self.np_random.choice(
                possible_targets,
                size=targets_to_generate,
                replace=True
            )
            chosen_targets.extend(selected_targets)
            current_full_vector = self._get_expression_vector_for_index(cell_idx)
            for target in selected_targets:
                target_full_vector = self.average_full_expression_vectors[target]
                obs = {
                    "current_expression": current_full_vector,
                    "target_expression": target_full_vector,
                }
                val_set_obs.append(obs)
                episode_start_indices.append(cell_idx)
            current_episode_count += targets_to_generate
        return val_set_obs, episode_start_indices, chosen_targets

    def compute_action_mask_for_state(self):
        #return numpy arrayw with 1 for valid actions and 0 for invalid actions, invalid if the action would lead to the same cell
        results_for_cell = self.transition_matrix[self.current_cell_index]
        is_valid = (results_for_cell != self.current_cell_index)
        action_mask = is_valid.astype(np.int8)
        if not self.eval_mode and not self.test_mode:
            action_leakage =~np.isin(results_for_cell, self.forbidden_indices)
            action_mask = action_mask & action_leakage
        return action_mask

class ActionMaskingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.n_actions = int(env.action_space.n)
        base = env.observation_space

        self.observation_space = gym.spaces.Dict({
            "current_expression": base["current_expression"],
            "target_expression": base["target_expression"],
            "action_mask": gym.spaces.MultiBinary(self.n_actions),
        })

    def observation(self, obs):
        mask = self.compute_mask()
        return {
            "current_expression": obs["current_expression"],
            "target_expression": obs["target_expression"],
            "action_mask": mask,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), rew, terminated, truncated, info

    def compute_mask(self):
        #only print if it contains a 0 in the action mask
        action_mask = self.unwrapped.compute_action_mask_for_state()
        return action_mask

    def set_phase(self, new_phase: int, step_increase: int) -> bool:
        return self.unwrapped.set_phase(new_phase, step_increase)

    def get_phase(self) -> int:
        phse = self.unwrapped.get_phase()
        print("Getting phase from wrapper: ", phse)
        return self.unwrapped.get_phase()






    #
    # def _create_cell_type_to_hot_encoded_vec_dict(self, celltypes: List[str]) -> Dict[str, np.ndarray]:
    #     """
    #     NEW: Helper method ported from VecEnv to create one-hot encodings.
    #     The order of `cell_type_string` must be preserved for consistency.
    #     """
    #     cell_type_to_hot_vec = {}
    #     cell_type_string = ["Primitive Streak", "Caudal Epiblast", "Epiblast", "Naive PGCs", "Epidermis Progenitors",
    #                         "Caudal Mesoderm", "Parietal Endoderm", "PGCs", "(pre)Somitic/Wavefront",
    #                         "Nascent Mesoderm", "NMPs", "LP/Intermediate Mesoderm", "Neural Progenitors",
    #                         "(early) Somite", "Cardiac Mesoderm", "Endothelium", "Visceral Endoderm", "Dermomyotome",
    #                         "Erythrocytes", "Reprogramming PGCs", "Sclerotome", "Roof Plate Neural Tube",
    #                         "Floor Plate Neural Tube", "ExE Endoderm", "Cardiomyocytes", "Pharyngeal Mesoderm",
    #                         "Early Motor Neurons", "Late Motor Neurons", "Myotome", "Megakaryocytes"]
    #
    #     one_hot_length = len(celltypes)
    #     celltype_to_idx = {name: i for i, name in enumerate(cell_type_string)}
    #
    #     for celltype in celltypes:
    #         if celltype in celltype_to_idx:
    #             one_hot_vec = np.zeros(one_hot_length, dtype=np.float32)
    #             # Find the correct index in the *current adata's* list of celltypes
    #             true_idx = self.celltypes.index(celltype)
    #             one_hot_vec[true_idx] = 1.0
    #             cell_type_to_hot_vec[celltype] = one_hot_vec
    #         else:
    #             print(
    #                 f"Warning: Cell type '{celltype}' not found in predefined list. One-hot encoding may be inconsistent if not all cell types are present.")
    #
    #     return cell_type_to_hot_vec

    # def _create_normalized_embedding_coordinates(self) -> np.ndarray:
    #     umap = self.oracle.adata.obsm[self.embedding_name].copy()
    #     scaler = StandardScaler()
    #     return scaler.fit_transform(umap)

    # def _calculate_centre_umap_coords_target(self, outlier_percentile: float = 95) -> Tuple[
    #     Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    #     target_centres = {}
    #     target_centres_norm = {}
    #     raw_embedding = self.oracle.adata.obsm[self.embedding_name]
    #     norm_embedding = self.umap_coords_normalized
    #
    #     for celltype in self.celltypes:
    #         cell_mask = self.oracle.adata.obs["celltype"] == celltype
    #         if not np.any(cell_mask): continue
    #
    #         raw_cluster_coords = raw_embedding[cell_mask]
    #         norm_cluster_coords = norm_embedding[cell_mask]
    #
    #         temp_centre = np.mean(raw_cluster_coords, axis=0)
    #         distances = np.linalg.norm(raw_cluster_coords - temp_centre, axis=1)
    #         distance_cutoff = np.percentile(distances, outlier_percentile)
    #         core_cells_mask = distances < distance_cutoff
    #
    #         final_raw_centroid = np.mean(raw_cluster_coords[core_cells_mask], axis=0)
    #         final_norm_centroid = np.mean(norm_cluster_coords[core_cells_mask], axis=0)
    #
    #         target_centres[celltype] = final_raw_centroid
    #         target_centres_norm[celltype] = final_norm_centroid
    #
    #     return target_centres, target_centres_norm
