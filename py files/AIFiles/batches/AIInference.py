import os
import pickle
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from Env import CellOracleSB3VecEnv as CellOracleCustomEnv
from Env_with_new_input import CellOracleSB3VecEnv as CellOracleCustomEnvNewInput
from MoreGeneralizableEnv import CellOracleSB3VecEnv as CellOracleCustomEnvGeneral
from FastEnv import CellOracleSB3VecEnv as CellOracleCustomEnvFast
from stable_baselines3 import PPO

def getCorrectEnv(env_type:str, config: dict):
    """
    Returns the correct environment class based on the env_type.
    """
    env_params = {
        "oracle_path": config.get("ORACLE_PATH"),
        "batch_size": config.get("BATCH_SIZE"),
        "max_steps": config.get("MAX_STEPS"),  # Use a consistent key name
        "step_penalty": config.get("STEP_PENALTY", -0.01),
        "goal_bonus": config.get("GOAL_BONUS", 1.0),
        "fail_penalty": config.get("FAIL_PENALTY", -1.0),
        "distance_reward_scale": config.get("DISTANCE_REWARD_SCALE", 1.0),
        "allow_gene_activation": config.get("ALLOW_GENE_ACTIVATION", True),
        "gamma_distance": config.get("DISTANCE_GAMMA", 0.99),
        "number_of_targets_curriculum": config.get("TARGET_CELLS_PER_PHASE", 4),
        "use_prev_perturbs": config.get("USE_PREV_KNOCKOUT", False),
        "same_cell_penalty": config.get("SAME_CELL_PENALTY", -0.2),
        "standard_sd_factor": config.get("STANDARD_SD_FACTOR", 1.5),
        "use_similarity": config.get("USE_SIMILARITY_REWARD", False),
        "target_bounds": config.get("TARGET_BOUNDS", 0.2),
        "transition_path": config.get("TRANSITION_MATRIX", None),
    }
    if env_type == "FIRST":
        return CellOracleCustomEnv(**env_params
        )
    elif env_type == "NEW_INPUT":
        return CellOracleCustomEnvNewInput(
            **env_params,
        )
    elif env_type == "GENERAL":
        return CellOracleCustomEnvGeneral(
            **env_params,
        )
    elif env_type == "FAST":
        return CellOracleCustomEnvFast(
            **env_params,
        )

    else:
        raise ValueError(f"Unknown env_type: {env_type}. Expected 'NEW_INPUT' or 'OLD_INPUT'.")

def run_inference_on_single_cell(
        model_path: str,
        oracle_path: str,
        config: Dict,
        start_cell_idxs: List[int],
        target_cell_types: List[str],
        max_inference_steps: int = 30,
        deterministic: bool = True  ):


    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    inference_config = config.copy()
    inference_config["BATCH_SIZE"] = 1

    env = getCorrectEnv("FAST", inference_config)
    model = PPO.load(model_path, env=env)

    results_idx = []
    results_string = []
    goals_reached = []
    number_of_steps = []
    for counter,start_cell_idx in enumerate(start_cell_idxs):
        try:
            obs = env.reset_for_inference(start_idx=start_cell_idx, target_name=target_cell_types[counter])
        except Exception as e:
            print(f"Error during env.reset(). Ensure your reset method can handle options: {e}")
            print("Please modify your reset method to accept an 'options' dictionary.")
            return None

        action_sequence = []
        action_details = []
        goals_reached.append(False)
        results_idx.append([])
        results_string.append([])

        step_counter = 0
        for step in range(max_inference_steps):
            action_idx, _states = model.predict(obs, deterministic=deterministic)
            action_detail_str = env.env_method("get_action_details", int(action_idx[0]))[0]  # Using env_method
            action_sequence.append(int(action_idx[0]))
            action_details.append(action_detail_str)
            results_idx[counter].append(int(action_idx[0]))
            results_string[counter].append(action_detail_str)

            step_counter += 1


            obs, rewards, dones, infos = env.step(action_idx)

            if infos[0].get('succes', False):
                goals_reached[counter] = True
                break
        number_of_steps.append(step_counter)



    #print all results
    for idx, goal_reached in enumerate(goals_reached):
        if goal_reached:
            print(f"\nGoal reached for start cell index {start_cell_idxs[idx]} in {number_of_steps[idx]} steps with target cell type '{target_cell_types[idx]}'.")
        else:
            print(f"\nGoal not reached for start cell index {start_cell_idxs[idx]} for {number_of_steps[idx]} steps with target cell type '{target_cell_types[idx]}'.")
            continue
        for step, action in enumerate(results_string[idx]):
            print(f"Step {step + 1}: Action -> {action}")

    env.close()
    return results_string, results_string, goals_reached

