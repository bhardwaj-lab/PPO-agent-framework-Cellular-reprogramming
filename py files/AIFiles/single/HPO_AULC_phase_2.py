# File: run_phase2_validation.py

import os
import pickle
import copy
from datetime import datetime

import numpy as np
import optuna

# --- Import your main training function and the NEW AULC callback ---
from SingleEnvRunBuildAI import run_training, OptunaCallBackAULC
# Import the evaluation function that the callback depends on
from stable_baselines3.common.evaluation import evaluate_policy
from FinalEnvSingleInstance import CellOracleGymEnv as Env
from FinalEnvSingleInstance import ActionMaskingWrapper as ActionMaskEnv


def run_phase2_validation(base_config):
    """
    Performs Phase 2 validation using the OptunaCallBackAULC to calculate the final AULC
    based on performance on a held-out validation set.
    """

    # --- 1. Configuration ---
    TOP_PARAMS_PKL_PATH = base_config["TOP_PARAMS_PKL_PATH"]
    VALIDATION_SEEDS = [77, 777]  # Using the seeds you requested


    # --- 2. Load Top Configurations ---
    if not os.path.exists(TOP_PARAMS_PKL_PATH):
        raise FileNotFoundError(f"Top parameters file not found at: {TOP_PARAMS_PKL_PATH}")
    with open(TOP_PARAMS_PKL_PATH, "rb") as f:
        top_configurations = pickle.load(f)
    print(f"--- Loaded {len(top_configurations)} top configurations for validation. ---")

    final_results = []


    #we are going to make a dictionary with success rate as key (can loop through keys easily and sort on highest) and then a new dicth wiht keys params and average length
    with open(base_config["ORACLE_ADATA_PATH"], 'rb') as f:
        ORACLE_ADATA = pickle.load(f)
    with open(base_config["TRANSITION_MATRIX"], 'rb') as f:
        _SHARED_TRANSITION_MATRIX = pickle.load(f)
    with open(base_config["PERTURBABLE_GENES_PATH"], 'rb') as f:
        ORACLE_PERTURBABLE_GENES = pickle.load(f)

    for i, params in enumerate(top_configurations):
        rank = i + 1
        print(f"\n{'=' * 50}\n--- Starting validation for Rank {rank} Config ---\n")


        success_rate =[]
        average_steps = []
        for seed in VALIDATION_SEEDS:
            print(f"\n--- Running Full Training with Seed: {seed} ---")

            # --- a. Create the specific config for this run ---
            run_config = copy.deepcopy(base_config)
            #overwrite base config params with the params of the trial, update dict with new params dict, should overwrite perfectly the existing keys
            run_config.update(params)
            run_config["SEED"] = seed
            run_config["TOTAL_TIMESTEPS"] = base_config["TOTAL_TIMESTEPS"]  # Use the FULL budget

            run_name = f"rank_{rank:02d}_seed_{seed}"
            run_config["MODEL_SAVE_PATH"] = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], run_name)
            run_config["LOG_DIR"] = os.path.join(base_config["LOG_DIR_BASE"], run_name)
            run_config["WANDB_GROUP"] = f"Phase2_Validation_Rank_{rank:02d}"
            run_config["WANDB_RUN_NAME"] = run_name
            print("starting run with config:", run_config)


            # --- b. Run the full training ---
            # We don't need any special callbacks for this part, just the standard ones
            # from your run_training function (like WandbLoggingCallback).
            try:
                # The run_training function will train and save the final model
                trained_model = run_training(run_config, [])

                # --- c. Evaluate the FINAL trained model ---
                print(f"\n--- Starting Final Evaluation for Seed: {seed} ---")


                # Create a single, dedicated evaluation environment
                env_kwargs = dict(
                    adata_object=ORACLE_ADATA,
                    oracle_perturbable_genes=ORACLE_PERTURBABLE_GENES,
                    transition_matrix_object=_SHARED_TRANSITION_MATRIX,
                    max_steps=run_config["MAX_STEPS_PER_EPISODE"],
                    step_penalty=run_config["STEP_PENALTY"],
                    goal_bonus=run_config["GOAL_BONUS"],
                    discount_factor=run_config["GAMMA"],
                    allow_gene_activation=run_config["ALLOW_GENE_ACTIVATION"],
                    number_of_targets_curriculum=run_config.get("TARGET_CELLS_PER_PHASE", 4),
                    distance_reward_scale=run_config.get("DISTANCE_REWARD_SCALE", 1),
                    add_noise=run_config.get("ADD_NOISE", False),
                    test_frac=run_config.get("TEST_FRAC", 0.1),
                    val_frac=run_config.get("VAL_FRAC", 0.1),
                    seed=run_config.get("SEED", 77),
                    pgc_only_curriculum=False,
                )
                eval_env = ActionMaskEnv(Env(**env_kwargs))
                n_eval_episodes = run_config.get("N_EVAL_EPISODES", 1000)
                eval_env.unwrapped.switch_between_modes(is_eval_mode=True)
                eval_env.unwrapped.set_phase(5)
                success_percentage, step_counts_av = evaluate_model(trained_model, eval_env, n_eval_episodes)
                print("--- Final Evaluation Results ---: ", f"Success Rate: {success_percentage:.4f}, Average Steps: {step_counts_av:.2f}")
                success_rate.append(success_percentage)
                average_steps.append(step_counts_av)
            except Exception as e:
                print(f"!!! ERROR during full run for Rank {rank}, Seed {seed}: {e} !!!")
                import traceback
                traceback.print_exc()
                success_rate.append(-1.0)


        # Store final evaluation scores for this configuration
        final_results.append({
            "rank": rank, "params": params, "eval_scores": success_rate,
            "mean_score": np.mean(success_rate), "average_steps": np.mean(average_steps)
        })

    final_results = sorted(final_results, key=lambda x: x["mean_score"], reverse=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_pkl_path = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], f"phase2_validation_results_{timestamp}.pkl")
    with open(results_pkl_path, "wb") as f:
        pickle.dump(final_results, f)
    best_result = final_results[0]
    best_result_path = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], f"best_phase2_result_{timestamp}.pkl")
    with open (best_result_path, "wb") as f:
        pickle.dump(best_result, f)
    summary_path = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], f"phase2_validation_summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write("--- Phase 2 Validation: Final Ranked Results ---\n\n")
        for result in final_results:
            f.write(f"--- HPO Rank {result['rank']} ---\n")
            f.write(f"  Mean Success Rate: {result['mean_score']:.6f}\n")
            f.write(f"  Std Dev Success Rate: {np.std(result['eval_scores']):.6f}\n")
            f.write(f"  Mean Episode Length: {result['average_steps']:.2f}\n")
            f.write(f"  Individual Success Scores: {result['eval_scores']}\n")
            f.write("  Hyperparameters:\n")
            for key, value in result['params'].items():
                f.write(f"    {key}: {value}\n")
            f.write("\n" + "-" * 40 + "\n\n")

def evaluate_model(model, eval_env, n_eval_episodes):
    successes = []
    step_counts = []

    for ep in range(n_eval_episodes):
        obs, info = eval_env.reset()

        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1

        successes.append(1.0 if info.get("goal_reached") else 0.0)
        step_counts.append(steps)

    return float(np.mean(successes)), float(np.mean(step_counts))


if __name__ == "__main__":
    run_phase2_validation()