import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.sparse import issparse
import cupy as cp
from stable_baselines3.common.vec_env import VecEnv
import math
from sklearn.preprocessing import StandardScaler



class CellOracleSB3VecEnv(VecEnv):
    metadata = {"render_modes": []}

    def __init__(self,
                 oracle_path: str,batch_size: int = 64,max_steps: int = 50,step_penalty: float = -0.01,
                 goal_bonus: float = 1.0,fail_penalty: float = -1.0,distance_reward_scale: float = 1.0,same_cell_penalty: float = -0.2,
                 gamma_distance: float = 1.0,allow_gene_activation: bool = False,use_prev_perturbs: bool = False,number_of_targets_curriculum: int = 4,target_bounds: float = 0.2,

                 standard_sd_factor: float = 1.5,use_similarity: bool = False,initial_cell_idx: Optional[List[int]] = None,target_cell_types: Optional[List[int]] = None):

        self._allow_gene_activation = allow_gene_activation
        self.max_steps = max_steps
        self.use_similarity = use_similarity
        self.wandb_run_id = None
        self.wandb_run_name = None
        self.embedding_name = "X_umap"
        self.check_path(oracle_path)
        try:
            cp.cuda.Device(0).use()
            self.cupy_integration_use = True
            print("CuPy is available and GPU is accessible.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            # Handle the case where CuPy cannot access the GPU
            print("CuPy cannot access GPU, falling back to CPU integration.")
            self.cupy_integration_use = False

        self.number_of_episodes_started_completed = 0
        self.number_of_episodes_started_overal = 0
        self.number_of_goal_reached = 0

        with open(oracle_path, 'rb') as f:
            self.oracle = pickle.load(f)

        self.oracle.init(embedding_type="X_umap", n_neighbors=200, torch_approach=False,
                         cupy_approach=self.cupy_integration_use, batch_size=batch_size)
        self.use_prev_perturbs = use_prev_perturbs

        self.n_cells = self.oracle.adata.n_obs
        self.all_genes = self.oracle.adata.var.index.tolist()
        self.genes_that_can_be_perturbed = self.oracle.return_active_reg_genes()
        self.number_of_reg_genes = len(self.genes_that_can_be_perturbed)
        self.celltypes = self.oracle.adata.obs[
            'celltype'].unique().tolist() if 'celltype' in self.oracle.adata.obs.columns else None

        # This is the attribute that was missing before it was used
        self.reg_gene_adata_indices = np.array([self.all_genes.index(g) for g in self.genes_that_can_be_perturbed])
        self.reg_gene_to_full_idx = {name: i for i, name in enumerate(self.all_genes) if
                                     name in self.genes_that_can_be_perturbed}

        self.action_space_size = self.number_of_reg_genes
        self.action_space_size += self.number_of_reg_genes if self._allow_gene_activation else 0

        self.celltype_to_one_hot = self._create_cell_type_to_hot_encoded_vec_dict(self.celltypes)

        self.high_ranges_dict = self._calculate_gene_activation_values(sd_factor=standard_sd_factor) if self._allow_gene_activation else None
        self.average_expression_vectors = self._compute_average_expression_vectors()  # Now this will work
        self.umap_coords_normalized = self._create_normalized_embedding_coordinates()
        self.average_umap_coordinates, self.average_normalized_umap_coordinates = self.calculate_centre_umap_coords_target()  # Compute average UMAP coordinates for each cell type
        self.total_curriculum_targets = self._compute_total_curriculum_targets(number_of_targets_curriculum)


        self.debug_cell_idx = initial_cell_idx if initial_cell_idx is not None else None
        self.debug_target_cell_types = target_cell_types if target_cell_types is not None else None

        observation_space = self._create_observation_space()
        action_space = self._create_action_space()

        super().__init__(num_envs=batch_size,
                         observation_space=observation_space,
                         action_space=action_space)

        self.current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_start_times = np.array([time.time()] * self.num_envs, dtype=np.float32)

        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.fail_penalty = fail_penalty
        self.distance_reward_scale = distance_reward_scale
        self.same_cell_penalty = same_cell_penalty
        self.gamma = gamma_distance

        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_cell_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.current_target_cell_types = np.full(self.num_envs, None, dtype=object)
        self.current_phase = 1

        cell_types_series = self.oracle.adata.obs['celltype']
        pgc_mask = cell_types_series.str.contains("PGCs", na=False)
        pgc_positional_indices = np.where(pgc_mask)[0].astype(np.int32)
        self.temp_indices_to_choose_from = pgc_positional_indices
        self.target_bounds = target_bounds
        self._actions: Optional[np.ndarray] = None

    def set_wandb_id(self, wandb_run_id: str, wandb_run_name: str):
        self.wandb_run_id = wandb_run_id
        self.wandb_run_name = wandb_run_name

    def _create_cell_type_to_hot_encoded_vec_dict(self, celltypes: List[str]) -> dict:
        cell_type_to_hot_vec = {}
        cell_type_string = ["Primitive Streak", "Caudal Epiblast", "Epiblast", "Naive PGCs", "Epidermis Progenitors",
                            "Caudal Mesoderm", "Parietal Endoderm", "PGCs", "(pre)Somitic/Wavefront",
                            "Nascent Mesoderm", "NMPs", "LP/Intermediate Mesoderm", "Neural Progenitors",
                            "(early) Somite", "Cardiac Mesoderm", "Endothelium", "Visceral Endoderm", "Dermomyotome",
                            "Erythrocytes", "Reprogramming PGCs", "Sclerotome", "Roof Plate Neural Tube",
                            "Floor Plate Neural Tube", "ExE Endoderm", "Cardiomyocytes", "Pharyngeal Mesoderm",
                            "Early Motor Neurons", "Late Motor Neurons", "Myotome", "Megakaryocytes"]
        one_hot_length = len(celltypes)
        for i, celltype in enumerate(cell_type_string):
            if celltype in celltypes:
                one_hot_vec = np.zeros(one_hot_length, dtype=np.float32)
                one_hot_vec[i] = 1.0
                cell_type_to_hot_vec[celltype] = one_hot_vec
            else:
                raise ValueError(f"Cell type '{celltype}' not found in predefined list.")
        return cell_type_to_hot_vec

    def _create_action_space(self) -> spaces.Space:
        return spaces.Discrete(self.action_space_size)

    def _create_normalized_embedding_coordinates(self) -> np.ndarray:
        umap = self.oracle.adata.obsm[self.embedding_name].copy()
        # normalize the coords
        scaler = StandardScaler()
        umap = scaler.fit_transform(umap)
        print("UMAP coordinates normalized with StandardScaler with amx: ", np.max(umap), " and min: ", np.min(umap))
        return umap


    def _create_observation_space(self) -> spaces.Dict:
        z_score_bound = 100  # Assuming z-scored gene expression
        return spaces.Dict({
            "current_state": spaces.Box(low=-z_score_bound, high=z_score_bound, shape=(self.number_of_reg_genes,),
                                        dtype=np.float32),
            "target_state": spaces.Box(low=-z_score_bound, high=z_score_bound, shape=(self.number_of_reg_genes,),
                                       dtype=np.float32),
        })

    def reset(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(
            self._np_random_seed if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)

        self.current_cell_indices = rng.choice(self.temp_indices_to_choose_from, size=self.num_envs, replace=False)

        # now sample target celltypes
        self.current_target_cell_types = self._get_target_cell_types(self.current_cell_indices)
        # for debug purposses
        if self.debug_cell_idx is not None:
            self.current_cell_indices = np.array(self.debug_cell_idx)
        if self.debug_target_cell_types is not None:
            for i in range(self.num_envs):
                self.current_target_cell_types[i] = self.debug_target_cell_types[i]

        self.oracle.reset_info_during_training_for_batch_instance(np.arange(self.num_envs))
        self.current_steps.fill(0)
        self.current_episode_rewards.fill(0.0)
        self.current_episode_lengths.fill(0)
        self.current_episode_start_times[:] = time.time()

        return self._get_obs()

    def reset_for_inference(self, start_idx: int, target_name: str) -> Dict[str, np.ndarray]:
        if self.num_envs > 1:
            print("Warning: reset_for_inference is designed for a single environment (num_envs=1).")

        self.current_steps.fill(0)
        self.current_episode_rewards.fill(0.0)
        self.current_episode_lengths.fill(0)
        self.current_episode_start_times[:] = time.time()
        self.oracle.reset_info_during_training_for_batch_instance(np.arange(self.num_envs))

        self.current_cell_indices[0] = start_idx
        self.current_target_cell_types[0] = target_name

        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        """Performs the simulation and returns results (synchronous execution).
            FOR NOW:
            1) WE DO NOT INTEGRATE A POLICY MASK (ACTIONS THAT ARE NOT POSSIBLE)
            2) WE DO INTEGRATE A FAILURE PENALTY


        """
        if self._actions is None:
            raise RuntimeError("Cannot call step_wait without calling step_async first.")

        actions = self._actions.copy()  # Copy actions to avoid modifying the original
        self._actions = None

        perturb_conditions = []
        action_indices = actions.flatten().astype(int)
        total_activations_per_step = 0
        for i in range(self.num_envs):
            action_idx = action_indices[i]
            # Basic check for action validity (important if not using MaskablePPO perfectly)
            if action_idx < self.number_of_reg_genes:
                # does not matter if we allow gene activation or not, we just set the value to 0.0
                gene_to_knockout = self.genes_that_can_be_perturbed[action_idx]
                perturb_conditions.append((gene_to_knockout, 0.0))  # Assuming knockout value is always 0.0
                continue
            if self._allow_gene_activation:
                if action_idx > 2 * self.number_of_reg_genes:
                    print("WARNING THIS SHOULD NOT HAPPEN MORE ACTIONS THAN 2*REG GENES")
                    raise ValueError("Action index out of bounds for gene activation.")
                gene_to_activate = self.genes_that_can_be_perturbed[action_idx % self.number_of_reg_genes]
                # Get the activation value from the high ranges dict
                activation_value = self.high_ranges_dict[gene_to_activate]
                perturb_conditions.append((gene_to_activate, activation_value))
                total_activations_per_step += 1

        n_neighbors_sim, n_propagation_sim, threads_sim = 200, 3, 4
        simulation_success = True
        try:
            # sgtart timer to keep track of how long it takes
            start = time.time()
            end = ""
            if self.cupy_integration_use:
                # Use cupy for GPU acceleration
                _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch_cp(
                    batch_size=self.num_envs, idxs=self.current_cell_indices.tolist(),
                    perturb_condition=perturb_conditions, n_neighbors=n_neighbors_sim,
                    n_propagation=n_propagation_sim, threads=threads_sim,
                    knockout_prev_used=self.use_prev_perturbs
                )
                end = time.time()

            else:
                _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch(
                    batch_size=self.num_envs, idxs=self.current_cell_indices.tolist(),
                    perturb_condition=perturb_conditions, n_neighbors=n_neighbors_sim,
                    n_propagation=n_propagation_sim, threads=threads_sim,
                    knockout_prev_used=self.use_prev_perturbs
                )
                end = time.time()
            print("Time taken for simulation: ", end - start)
            new_cell_indices = np.array(new_idx_list, dtype=np.int32)  # Ensure correct type
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print the full traceback
            print('WATCH OUT THIS IS WRONG')
            print(e)
            simulation_success = False
            new_cell_indices = self.current_cell_indices

        # init all necessary info
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)  # For info dict

        if not simulation_success:
            print("THIS IS A PROBLEM IF THIS HAPPENS")
            # just terminate the entire episode if it fails, this indicates a problem
            rewards.fill(self.fail_penalty)
            dones.fill(True)
            truncations.fill(False)
            return self._get_obs(), rewards, dones, self._create_info_dict(
                goal_reached=np.zeros(self.num_envs, dtype=bool), truncated=np.zeros(self.num_envs, dtype=bool),
                simulation_failed=np.ones(self.num_envs, dtype=bool),
                same_cell_indices_as_before_mask=np.zeros(self.num_envs, dtype=bool),
                done_episode_rewards=self.current_episode_rewards, done_episode_lengths=self.current_episode_lengths,
                done_episode_start_times=self.current_episode_start_times)

        # The chance of this happening above is slim but we just implement it to be sure and catch if it goes wrong, can save alot of pain debugging DO NOT REMOVE LATER !

        # first set the neew info correctly
        same_cell_indices_as_before_mask = new_cell_indices == self.current_cell_indices
        rewards_distance_phi = self._reward_system_distance_calc(
            new_cell_indices)  # DO NOT CALL AFTER SETTING THE CURRENT CELL INDICES
        self.current_cell_indices = new_cell_indices
        self.current_steps += 1

        celltypes_of_new_indices = self.oracle.adata[new_cell_indices].obs['celltype'].to_numpy()
        # TODO NAAR DEZE STATEMENT KIJKEN
        goal_reached_mask = self.which_cell_have_reached_target(new_cell_indices)

        is_timeout_mask = self.current_steps >= self.max_steps  # yes this is an actual mask that is created DO NOT MESS WITH IT LATER
        # set to default step penalty
        rewards.fill(self.step_penalty)
        rewards += rewards_distance_phi  # add the distance reward
        # then add a possible time out penalty
        rewards[is_timeout_mask] += self.fail_penalty
        # but if goal reached then add the goal reached bonus
        rewards[goal_reached_mask] += self.goal_bonus
        # add additgional if same cell is returned, very bad perturbation
        rewards[same_cell_indices_as_before_mask] += self.same_cell_penalty

        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1
        dones[is_timeout_mask] = True
        dones[goal_reached_mask] = True
        done_and_goal = goal_reached_mask & dones
        self.number_of_goal_reached += np.sum(done_and_goal)
        # same for here, first do timeout, then overwrite with goal reached
        truncations[is_timeout_mask] = True
        truncations[goal_reached_mask] = False

        # so this is for logging purposes, we will calc averages and log then
        fail_pen = np.zeros(self.num_envs, dtype=np.float32)
        fail_pen[is_timeout_mask] += self.fail_penalty
        average_pen = np.mean(fail_pen)
        goal_reach = np.zeros(self.num_envs, dtype=np.float32)
        goal_reach[goal_reached_mask] += self.goal_bonus
        average_goal = np.mean(goal_reach)
        av_same_cell_indices_as_before_mask = np.zeros(self.num_envs, dtype=np.float32)
        av_same_cell_indices_as_before_mask[same_cell_indices_as_before_mask] += self.same_cell_penalty
        average_same_cell = np.mean(av_same_cell_indices_as_before_mask)
        average_distance = np.mean(rewards_distance_phi)

        #
        print("average overall reward: ", np.mean(rewards), " average fail penalty: ", average_pen,
              " and average goal bonus: ", average_goal, " and average same cell penalty: ", average_same_cell,
              "average distance reward: ", average_distance, " and average step penalty: ", self.step_penalty)

        done_episode_rewards_for_info = self.current_episode_rewards.copy()
        done_episode_lengths_for_info = self.current_episode_lengths.copy()
        done_episode_start_times_for_info = self.current_episode_start_times.copy()

        obs_before_reset = self._get_obs()  # Get the observation before resetting

        info_dict = self._create_info_dict(goal_reached=goal_reached_mask, truncated=truncations,
                                           simulation_failed=np.zeros(self.num_envs, dtype=bool),
                                           same_cell_indices_as_before_mask=same_cell_indices_as_before_mask,
                                           obs_before_reset=obs_before_reset, average_goal=average_goal,
                                           average_penalty=average_pen, average_distance=average_distance,
                                           average_same_cell=average_same_cell,
                                           percentage_of_activation=total_activations_per_step / self.num_envs,
                                           current_cell_types=celltypes_of_new_indices,
                                           target_cell_types=self.current_target_cell_types,
                                           done_episode_rewards=done_episode_rewards_for_info,
                                           done_episode_lengths=done_episode_lengths_for_info,
                                           done_episode_start_times=done_episode_start_times_for_info)

        self._handle_batch_resets(dones)  # Handle individual resets

        obs_after_reset = self._get_obs()  # Get the observation after resetting, this is given back in this step
        return obs_after_reset, rewards, dones, info_dict


    def which_cell_have_reached_target(self, next_cell_indices:np.ndarray):

        has_same_celltype_mask = self.oracle.adata[next_cell_indices].obs['celltype'].to_numpy() == self.current_target_cell_types
        #now c heck if it is in the correct bounds
        average_umap_coords_target = np.array([self.average_umap_coordinates[t] for t in self.current_target_cell_types])
        distances_current_cells_to_target = np.linalg.norm(self.oracle.adata.obsm[self.embedding_name][next_cell_indices] - average_umap_coords_target, axis=1)
        within_bounds_mask = distances_current_cells_to_target <= self.target_bounds
        # print("so many in boudns: ",np.sum(within_bounds_mask) , " with this any correct celltype: ", + np.sum(has_same_celltype_mask), " and total: ", np.sum(within_bounds_mask & has_same_celltype_mask))
        # Combine both masks
        return has_same_celltype_mask & within_bounds_mask

    def calculate_centre_umap_coords_target(self,
                                   outlier_percentile: float = 95) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Calculates the robust centroid for each cell type in an embedding.
        """
        target_centres = {}
        target_centres_norm = {}

        unique_celltypes = self.oracle.adata.obs["celltype"].unique()
        raw_embedding = self.oracle.adata.obsm[self.embedding_name]
        norm_embedding = self.umap_coords_normalized

        for celltype in unique_celltypes:
            cell_mask = self.oracle.adata.obs["celltype"] == celltype
            raw_cluster_coords = raw_embedding[cell_mask]
            norm_cluster_coords = norm_embedding[cell_mask]
            print(f"Calculating centroid for celltype: {celltype}, number of cells: {len(raw_cluster_coords)}")

            if len(raw_cluster_coords) == 0:
                continue
            temp_centre = np.mean(raw_cluster_coords, axis=0)
            distances = np.linalg.norm(raw_cluster_coords - temp_centre, axis=1)
            #select only 90% closest, get rid of outliers
            distance_cutoff = np.percentile(distances, outlier_percentile)
            core_cells_mask = distances < distance_cutoff
            core_raw_coords = raw_cluster_coords[core_cells_mask]
            core_norm_coords = norm_cluster_coords[core_cells_mask]

            final_raw_centroid = np.mean(core_raw_coords, axis=0)
            final_norm_centroid = np.mean(core_norm_coords, axis=0)
            print(f"Final centroid for celltype {celltype}: {final_raw_centroid}, norm: {final_norm_centroid}")


            target_centres[celltype] = final_raw_centroid
            target_centres_norm[celltype] = final_norm_centroid


        return target_centres, target_centres_norm

    def _handle_batch_resets(self, dones: np.ndarray) -> None:
        reset_indices = np.where(dones)[0]
        num_to_reset = len(reset_indices)

        if num_to_reset <= 0:
            return

        reset_rng = np.random.default_rng(self._np_random_seed + self.current_steps.sum() if hasattr(self,
                                                                                                     "_np_random_seed") and self._np_random_seed is not None else None)
        new_start_indices = reset_rng.choice(self.temp_indices_to_choose_from, size=num_to_reset, replace=False)
        new_target_types_arr = self._get_target_cell_types(new_start_indices, reset_rng)

        self.current_cell_indices[reset_indices] = new_start_indices
        self.current_target_cell_types[reset_indices] = new_target_types_arr
        self.current_steps[reset_indices] = 0
        self.current_episode_rewards[reset_indices] = 0.0
        self.current_episode_lengths[reset_indices] = 0
        self.current_episode_start_times[reset_indices] = time.time()
        self.oracle.reset_info_during_training_for_batch_instance(reset_indices)
        self.number_of_episodes_started_completed += num_to_reset
        self.number_of_episodes_started_overal += num_to_reset

    def _reward_system_distance_calc(self, next_indices_after_perturb: np.ndarray) -> np.ndarray:
        distances_next = self._calculate_expression_distances_to_target(next_indices_after_perturb)
        distances_before = self._calculate_expression_distances_to_target(self.current_cell_indices)
        if self.use_similarity:
            epsilon = 1e-8
            potential_next = 1.0 / (1.0 + distances_next + epsilon)
            potential_before = 1.0 / (1.0 + distances_before + epsilon)
            phi_reward = self.gamma * potential_next - potential_before
        else:
            costs_change = self.gamma * distances_next - distances_before
            phi_reward = -costs_change

        phi_reward *= self.distance_reward_scale
        return phi_reward

    def _calculate_expression_distances_to_target(self, cell_indices: np.ndarray) -> np.ndarray:
        current_expr_vectors = self._get_current_expression_vector(cell_indices)
        target_expr_vectors = self._get_target_expression_vectors()

        diff = current_expr_vectors - target_expr_vectors
        distances = np.linalg.norm(diff, axis=1)
        return np.maximum(distances, 0.0).astype(np.float32)

    def _create_info_dict(self, goal_reached: np.ndarray, truncated: np.ndarray, simulation_failed: np.ndarray,
                          same_cell_indices_as_before_mask: np.ndarray,
                          obs_before_reset: Dict[str, np.ndarray],
                          done_episode_rewards: np.ndarray,
                          done_episode_lengths: np.ndarray,
                          done_episode_start_times: np.ndarray,
                          percentage_of_activation: float = None,
                          average_goal: float = None,
                          average_penalty: float = None, average_distance: float = None,
                          average_same_cell: float = None, current_cell_types: np.ndarray = None,
                          target_cell_types: np.ndarray = None) -> List[Dict[str, Any]]:
        """Creates the info dictionary list."""
        infos = []
        is_done = goal_reached | truncated | simulation_failed
        # print how many instnances are done
        print("Number of done instances: ", np.sum(is_done))
        unique_cell_types = np.unique(current_cell_types) if current_cell_types is not None else None
        unique_target_cell_types = np.unique(target_cell_types) if target_cell_types is not None else None

        for i in range(self.num_envs):
            info_dict_per_env = {}
            info_dict_per_env["TimeLimit.truncated"] = truncated[i]
            if average_goal is not None:
                info_dict_per_env["step_avg/average_goal"] = average_goal
            if average_penalty is not None:
                info_dict_per_env["step_avg/average_penalty"] = average_penalty
            if average_distance is not None:
                info_dict_per_env["step_avg/average_distance"] = average_distance
            if average_same_cell is not None:
                info_dict_per_env["step_avg/average_same_cell"] = average_same_cell
                info_dict_per_env["diagnostics/same_cell_event_this_step"] = bool(same_cell_indices_as_before_mask[i])
            if current_cell_types is not None:
                info_dict_per_env["batch_diversity/current_cell_types_unique_in_batch"] = unique_cell_types
            if target_cell_types is not None:
                info_dict_per_env["batch_diversity/target_cell_types_unique_in_batch"] = unique_target_cell_types
            if percentage_of_activation is not None:
                info_dict_per_env["step_avg/percentage_of_activation"] = percentage_of_activation
            if is_done[i]:
                final_info_payload = {
                    "steps": self.max_steps if truncated[i] else self.current_steps[i],
                    "goal_reached": goal_reached[i],
                    "truncated": truncated[i],
                    "simulation_failed": simulation_failed[i],
                    "episode": {  # Standard Monitor keys
                        "r": done_episode_rewards[i],
                        "l": done_episode_lengths[i],
                        "t": time.time() - done_episode_start_times[i]
                    },
                }
                info_dict_per_env["succes"] = goal_reached[i]
                info_dict_per_env["terminal_observation"] = {k: v[i] for k, v in obs_before_reset.items()}
                info_dict_per_env["episode"] = {
                    "r": float(done_episode_rewards[i]),
                    "l": int(done_episode_lengths[i]),
                    "t": float(time.time() - done_episode_start_times[i])
                }
                info_dict_per_env["final_info_payload"] = final_info_payload
            infos.append(info_dict_per_env)
        return infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(self, attr_name) for _ in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> None:
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        elif hasattr(self.oracle, attr_name):
            setattr(self.oracle, attr_name, value)
        else:
            raise AttributeError(f"Attribute '{attr_name}' not found on environment or oracle.")

    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, List[int], np.ndarray]] = None,
                   **method_kwargs) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            result = method(*method_args, **method_kwargs)
            return [result] * len(target_envs)
        elif hasattr(self.oracle, method_name):
            method = getattr(self.oracle, method_name)
            result = method(*method_args, **method_kwargs)
            return [result] * len(target_envs)
        else:
            raise AttributeError(f"Method '{method_name}' not found on environment or oracle.")

    def env_is_wrapped(self, wrapper_class: type, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[
        bool]:
        return [False] * self.num_envs

    def _get_target_envs(self, indices: Optional[Union[int, List[int], np.ndarray]]) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        elif isinstance(indices, int):
            return [indices]
        return list(indices)

    def check_path(self, oracle_path: str):
        if not oracle_path.endswith('.pkl'): raise ValueError('Pickle file needed')
        if not os.path.exists(oracle_path): raise ValueError(f'Path does not exist: {oracle_path}')

    def _get_current_expression_vector(self, cell_indices: np.ndarray) -> np.ndarray:
        data = self.oracle.get_AI_input_for_cell_indices(cell_indices, self.reg_gene_adata_indices, self.use_prev_perturbs)
        return data.toarray().astype(np.float32) if issparse(data) else np.array(data, dtype=np.float32)

    def _get_target_expression_vectors(self) -> np.ndarray:
        target_vectors = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=np.float32)
        for i, target_type in enumerate(self.current_target_cell_types):
            avg_vector = self.average_expression_vectors.get(target_type)
            if avg_vector is not None:
                target_vectors[i] = avg_vector
        return target_vectors

    def _get_target_cell_types(self, cell_indices: np.ndarray, rng_gen=None) -> np.ndarray:
        current_cell_types = self.oracle.adata[cell_indices].obs['celltype'].to_numpy()
        target_cell_types = []
        target_rng = rng_gen if rng_gen is not None else np.random.default_rng(
            self._np_random_seed if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)

        for i in range(len(current_cell_types)):
            phase_target_options = self.total_curriculum_targets[self.current_phase - 1].get(current_cell_types[i], [])
            if not phase_target_options:
                all_other_types = [ct for ct in self.celltypes if ct != current_cell_types[i] and 'PGC' not in ct]
                target_cell_types.append(
                    target_rng.choice(all_other_types) if all_other_types else current_cell_types[i])
            else:
                target_cell_types.append(target_rng.choice(phase_target_options))
        return np.array(target_cell_types, dtype=object)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        current_state_vecs = self._get_current_expression_vector(self.current_cell_indices)
        target_state_vecs = self._get_target_expression_vectors()

        return {
            "current_state": current_state_vecs,
            "target_state": target_state_vecs,
        }

    def _compute_average_expression_vectors(self) -> Dict[str, np.ndarray]:
        average_vectors = {}
        for celltype in self.celltypes:
            boolean_mask_celltype = (self.oracle.adata.obs['celltype'] == celltype).values
            cluster_reg_gene_expr = self._get_current_expression_vector(np.where(boolean_mask_celltype)[0])
            if cluster_reg_gene_expr.shape[0] > 0:
                average_vectors[celltype] = np.mean(cluster_reg_gene_expr, axis=0)
        return average_vectors

    def _compute_total_curriculum_targets(self, number_of_targets, start_with_pgcs=True) -> List:
        """init the total target curriculum using the _compute_nearest_celltype_targets func """
        if not start_with_pgcs:
            if self.average_umap_coordinates is None:
                raise ValueError("Average UMAP coordinates not computed. Call _compute_average_umap_coordinates first.")
            number_of_phases = (len(self.celltypes) // number_of_targets) + 1
            total_curriculum_targets = []
            for phase in range(1, number_of_phases + 1):
                phase_targets = self._compute_nearest_celltype_targets(phase, number_of_targets)
                total_curriculum_targets.append(phase_targets)
            return total_curriculum_targets

        number_of_phases = math.ceil(len(self.celltypes) / number_of_targets) + 1
        total_curriculum_targets = []
        for phase in range(1, number_of_phases + 1):
            phase_targets = self._compute_nearest_celltype_targets_only_pgcs(phase, number_of_targets)
            total_curriculum_targets.append(phase_targets)
        return total_curriculum_targets

    def _compute_nearest_celltype_targets(self, phase: int, number_of_targets: int) -> dict:
        """
        THIS FUNCTION DOES NOT INCLUDE ITSELF IN TARGET!
        Compute the nearest celltype targets for all celltypes, this is used for curriculum learning
        Based on phase, this function spits out a dict of celltypes that are direct neighbors, or 2nd neighbors etc dependent on phase
        It returns a list of dictionary, with keys -> celltype, value -> list of possible target celltypes
        Variables: phase indicates how many different celltypes we allow for, like phase =1 -> equal to the first phase of training, phase =2 -> equal to the second phase of training, etc
        In the end, phase =1 allows for number_of_targets celltypes, phase 2 allows for 2*number_of_targets celltypes, etc
        NOT SURE IF WE WANT TO IMPLEMENT SLICING IN A WAY THAT THE CELLTYPES FROM THE PREVIOUS CELLTYPES ARE NOT INCLUDED AS THE FIRST PHASE CELLTYPES WOULD THEN BE VERY HEAVILY TARGETTED AND SEEN WAY MORE?
        """
        # Get the UMAP coordinates for the clusters
        celltype_dict = {}
        for celltype in self.celltypes:
            average_coord_celltype = self.average_normalized_umap_coordinates[celltype]
            distances = []
            for other_celltype in self.celltypes:
                if celltype != other_celltype:
                    average_coord_other_celltype = self.average_normalized_umap_coordinates[other_celltype]
                    distance = np.linalg.norm(average_coord_celltype - average_coord_other_celltype)
                    distances.append((other_celltype, distance))

            distances.sort(key=lambda x: x[1])
            # now choose the clostest one according to phase
            # get rid of the current celltype as this is not relevant
            closest_celltypes = []
            for i in range(phase * number_of_targets):
                if i < len(distances):
                    closest_celltypes.append(distances[i][0])
            celltype_dict[celltype] = closest_celltypes
        return celltype_dict

    def _compute_nearest_celltype_targets_only_pgcs(self, phase: int, number_of_targets: int) -> dict:
        # retrieve the celltypes with pgc
        celltype_dict = {}
        pgc_celltypes = [celltype for celltype in self.celltypes if 'PGC' in celltype]
        for celltype in pgc_celltypes:
            average_coord_celltype = self.average_normalized_umap_coordinates[celltype]
            distances = []
            for other_celltype in self.celltypes:
                if other_celltype in pgc_celltypes:
                    continue
                if celltype == other_celltype:  # this should be handled by pervius but just to be sure
                    continue
                average_coord_other_celltype = self.average_normalized_umap_coordinates[other_celltype]
                distance = np.linalg.norm(average_coord_celltype - average_coord_other_celltype)
                distances.append((other_celltype, distance))
            # sort
            distances.sort(key=lambda x: x[1])
            # now choose the clostest one according to phase
            # get rid of the current celltype as this is not relevant
            closest_celltypes = []
            for i in range(phase * number_of_targets):
                if i < len(distances):
                    closest_celltypes.append(distances[i][0])
            celltype_dict[celltype] = closest_celltypes
        return celltype_dict


    def _calculate_gene_activation_values(self, sd_factor: float) -> Dict[str, float]:
        activation_values_dict = {}
        perturb_indices_in_adata = [self.oracle.adata.var.index.get_loc(g) for g in self.genes_that_can_be_perturbed if
                                    g in self.oracle.adata.var.index]
        expression_data = self.oracle.adata[:, perturb_indices_in_adata].X
        if issparse(expression_data): expression_data = expression_data.toarray()

        for i, gene_name in enumerate(self.genes_that_can_be_perturbed):
            if gene_name not in self.oracle.adata.var.index: continue
            gene_expr = expression_data[:, i]
            q1, q3 = np.percentile(gene_expr, [25, 75])
            iqr = q3 - q1
            filtered_expr = gene_expr[(gene_expr >= q1 - 3 * iqr) & (gene_expr <= q3 + 3 * iqr)]
            activation_val = np.mean(filtered_expr) + sd_factor * np.std(filtered_expr)
            activation_values_dict[gene_name] = max(0.0, activation_val)
        return activation_values_dict

    def set_phase(self, phase: int, step_increases: int = 5) -> bool:
        if phase > len(self.total_curriculum_targets): return False
        phase_diff = phase - self.current_phase
        self.max_steps += step_increases * phase_diff
        self.current_phase = phase
        return True

    def get_env_state(self) -> Dict[str, Any]:
        return {"max_steps": self.max_steps, "current_phase": self.current_phase, "wandb_run_id": self.wandb_run_id,
                "wandb_run_name": self.wandb_run_name}

    def set_env_state(self, state: Dict[str, Any]):
        self.max_steps = state.get("max_steps", self.max_steps)
        self.current_phase = state.get("current_phase", self.current_phase)
        self.wandb_run_id = state.get("wandb_run_id", self.wandb_run_id)
        self.wandb_run_name = state.get("wandb_run_name", self.wandb_run_name)

    def is_curriculum_finished(self) -> bool:
        return self.current_phase >= len(self.total_curriculum_targets)

    def get_current_goal_reached_percentage(self) -> float:
        return (
                           self.number_of_goal_reached / self.number_of_episodes_started_overal) * 100.0 if self.number_of_episodes_started_overal > 0 else 0.0

    def get_action_details(self, action_idx: int) -> str:
        if not (0 <= action_idx < self.action_space_size): return f"INVALID_ACTION_IDX_{action_idx}"
        if action_idx < self.number_of_reg_genes:
            return f"KO_{self.genes_that_can_be_perturbed[action_idx]}"
        else:
            return f"Activate_{self.genes_that_can_be_perturbed[action_idx - self.number_of_reg_genes]}"