# File: final_evaluation_module.py (Simplified and Corrected)

import os
import pickle
import copy
from datetime import datetime
import numpy as np

# --- Import your custom modules ---
from SingleEnvRunBuildAI import run_training
from FinalEnvSingleInstance import CellOracleGymEnv as Env
from FinalEnvSingleInstance import ActionMaskingWrapper


def evaluate_on_test_set_simple(model, config: dict, n_eval_episodes: int = 2000):
    """
    A simplified evaluation function that only calculates success rate and path length.
    """
    print(f"\n--- Starting Evaluation on TEST Set for {n_eval_episodes} episodes ---")

    # Create a single, non-vectorized environment for evaluation
    eval_env = ActionMaskingWrapper(Env(
        adata_object=config["ORACLE_ADATA"],
        oracle_perturbable_genes=config["ORACLE_PERTURBABLE_GENES"],
        transition_matrix_object=config["_SHARED_TRANSITION_MATRIX"],
        max_steps=config["MAX_STEPS_PER_EPISODE"],
        step_penalty=config["STEP_PENALTY"],
        goal_bonus=config.get("GOAL_BONUS", 0),
        discount_factor=config.get("GAMMA", 0.99),
        allow_gene_activation=config.get("ALLOW_GENE_ACTIVATION", True),
        number_of_targets_curriculum=config.get("TARGET_CELLS_PER_PHASE", 6),
        distance_reward_scale=config.get("DISTANCE_REWARD_SCALE", 5),
        val_frac=config.get("VAL_FRAC", 0.02),
        test_frac=config.get("TEST_FRAC", 0.03),
        seed=config.get("RANDOM_SEED", 77)
    ))

    # VERY IMPORTANT: Switch the environment to use the TEST data split
    eval_env.unwrapped.switch_to_test_mode(True)
    print("Environment switched to TEST mode.")

    successes, path_lengths = [], []

    for _ in range(n_eval_episodes):
        obs, info = eval_env.reset()
        terminated, truncated, steps = False, False, 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1

        successes.append(1.0 if info.get("goal_reached", False) else 0.0)
        path_lengths.append(steps)

    eval_env.close()

    mean_success_rate = float(np.mean(successes))
    mean_path_length = float(np.mean(path_lengths))

    print(f"--- Test Set Evaluation Complete ---")
    print(f"Success Rate: {mean_success_rate:.4f}")
    print(f"Average Path Length: {mean_path_length:.2f}")

    return mean_success_rate, mean_path_length


def run_final_evaluation(base_config: dict):
    """
    Loads the best config, trains on 3 seeds, evaluates success rate and path length,
    and saves a complete summary.
    """
    EVALUATION_SEEDS = [77, 777, 7777]
    BEST_PARAMS_PKL_PATH = base_config["BEST_PARAMS_PKL_PATH"]

    if not os.path.exists(BEST_PARAMS_PKL_PATH):
        raise FileNotFoundError(f"Best parameters file not found at: {BEST_PARAMS_PKL_PATH}")

    with open(BEST_PARAMS_PKL_PATH, "rb") as f:
        best_result = pickle.load(f)
        best_params = best_result['params']

    print("--- Loaded Best Hyperparameter Configuration ---")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    print("-" * 40)

    # Pre-load shared data objects to avoid loading them in a loop
    with open(base_config["ORACLE_ADATA_PATH"], 'rb') as f:
        base_config["ORACLE_ADATA"] = pickle.load(f)
    with open(base_config["TRANSITION_MATRIX"], 'rb') as f:
        base_config["_SHARED_TRANSITION_MATRIX"] = pickle.load(f)
    with open(base_config["PERTURBABLE_GENES_PATH"], 'rb') as f:
        base_config["ORACLE_PERTURBABLE_GENES"] = pickle.load(f)

    all_success_rates, all_path_lengths = [], []

    for seed in EVALUATION_SEEDS:
        print(f"\n{'=' * 60}\n--- Starting Final Run with Seed: {seed} ---\n{'=' * 60}")

        run_config = copy.deepcopy(base_config)
        run_config.update(best_params)
        run_config["RANDOM_SEED"] = seed
        run_config["TOTAL_TIMESTEPS"] = base_config["TOTAL_TIMESTEPS_FINAL"]

        run_name = f"final_run_seed_{seed}"
        run_config["MODEL_SAVE_PATH"] = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], run_name)
        run_config["LOG_DIR"] = os.path.join(base_config["LOG_DIR_BASE"], run_name)
        run_config["WANDB_GROUP"] = "Final_Evaluation_Runs"
        run_config["WANDB_RUN_NAME"] = run_name

        try:
            # Train the model from scratch
            trained_model = run_training(run_config, externall_callbacks=[])

            # Evaluate the trained model on the test set
            success_rate, avg_path_length = evaluate_on_test_set_simple(
                trained_model, run_config, n_eval_episodes=base_config["N_TEST_EPISODES"]
            )

            # Store the results
            all_success_rates.append(success_rate)
            all_path_lengths.append(avg_path_length)
        except Exception as e:
            print(f"!!! ERROR during run for Seed {seed}: {e} !!!")
            import traceback
            traceback.print_exc()
            all_success_rates.append(np.nan)  # Use NaN for failed runs
            all_path_lengths.append(np.nan)

    # --- Build the Final Summary ---
    summary_lines = [f"--- FINAL EVALUATION SUMMARY ---"]

    # 1. Report individual run results
    summary_lines.append("\n--- Individual Run Results ---")
    for i, seed in enumerate(EVALUATION_SEEDS):
        sr = all_success_rates[i]
        pl = all_path_lengths[i]
        if np.isnan(sr):
            line = f"Seed {seed:4d}: RUN FAILED"
        else:
            line = f"Seed {seed:4d}: Success Rate = {sr:.4f}, Avg Path Length = {pl:.2f}"
        summary_lines.append(line)

    # 2. Report aggregated statistics, safely ignoring failed runs
    mean_success = np.nanmean(all_success_rates)
    std_success = np.nanstd(all_success_rates)
    mean_length = np.nanmean(all_path_lengths)
    std_length = np.nanstd(all_path_lengths)

    summary_lines.append(f"\n--- Aggregated Statistics ({len(EVALUATION_SEEDS)} seeds) ---")
    summary_lines.append("Success Rate:")
    summary_lines.append(f"  - Mean: {mean_success:.4f}")
    summary_lines.append(f"  - Standard Deviation: {std_success:.4f}")
    summary_lines.append("Average Path Length:")
    summary_lines.append(f"  - Mean: {mean_length:.2f}")
    summary_lines.append(f"  - Standard Deviation: {std_length:.2f}")
    summary_lines.append('\n' + '=' * 60)

    summary = "\n".join(summary_lines)
    print("\n\n" + summary)

    # 3. Save the summary to a file
    summary_path = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], "final_evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nComprehensive summary saved to {summary_path}")

    # 4. Return a structured dictionary for the notebook
    return {
        "mean_success_rate": mean_success,
        "std_dev_success_rate": std_success,
        "mean_path_length": mean_length,
        "std_dev_path_length": std_length,
        "individual_runs": [
            {"seed": seed, "success_rate": sr, "path_length": pl}
            for seed, sr, pl in zip(EVALUATION_SEEDS, all_success_rates, all_path_lengths)
        ],
        "summary_text": summary
    }