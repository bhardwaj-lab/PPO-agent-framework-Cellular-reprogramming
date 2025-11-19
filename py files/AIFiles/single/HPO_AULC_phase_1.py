# run_optuna_hpo.py

import optuna
import os
import copy
import time  # For unique run IDs, not strictly necessary if trial.number is enough
from datetime import datetime
import torch.nn as nn  # For default activation function
import numpy as np
import wandb
import pickle

# --- Your Custom Modules ---
# Ensure these can be imported. If they are in a subdirectory,
# you might need to adjust Python's path or use relative imports if this script becomes part of a package.
import SingleEnvRunBuildAI as ai_module
from SingleEnvRunBuildAI import OptunaCallBackAULC

# from Env import CellOracleSB3VecEnv # Or your Env_with_new_input
# from AI import SB3OptunaPruningCallback # Assuming it's defined in AI.py or imported there

# --- Base Configuration (Modify as needed) ---
# This will be the starting point, Optuna will override parts of it.


def objective(trial: optuna.Trial, base_config: dict, gpu_id = None) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # --- 1. Setup Configuration for this Specific Trial ---
    trial_config = copy.deepcopy(base_config)

    # Suggest hyperparameters to be tuned by Optuna
    trial_config["LEARNING_RATE"] = trial.suggest_float("LEARNING_RATE", 3e-5, 3e-4, log=True)
    trial_config["PPO_N_STEPS"] = trial.suggest_categorical("PPO_N_STEPS", [1024, 2048,4096])
    trial_config["PPO_N_EPOCHS"] = trial.suggest_int("PPO_N_EPOCHS", 4, 16)
    trial_config["CLIP_RANGE"] = trial.suggest_float("CLIP_RANGE", 0.1, 0.4)
    trial_config["ENT_COEF"] = trial.suggest_float("ENT_COEF", 1e-5, 0.02)
    trial_config["GAMMA"] = trial.suggest_float("GAMMA", 0.98, 0.999)
    trial_config["PPO_BATCH_SIZE"] = trial.suggest_categorical("PPO_BATCH_SIZE", [64, 128, 256])
    trial_config["VF_COEF"] = trial.suggest_float("VF_COEF", 0.1, 1.0)
    trial_config["MAX_GRAD_NORM"] = trial.suggest_float("MAX_GRAD_NORM", 0.3, 2.0)
    trial_config["GAE_LAMBDA"] = trial.suggest_float("GAE_LAMBDA", 0.90, 0.999)


    # Suggest environment reward hyperparameters
    trial_config["NET_WIDTH_FIRST_P"] = trial.suggest_categorical("NET_WIDTH_FIRST_P", [64, 128, 256])
    trial_config["NET_WIDTH_SECOND_P"] = trial.suggest_categorical("NET_WIDTH_SECOND_P", [64, 128, 256])
    trial_config["NET_WIDTH_FIRST_V"] = trial.suggest_categorical("NET_WIDTH_FIRST_V", [64, 128, 256])
    trial_config["NET_WIDTH_SECOND_V"] = trial.suggest_categorical("NET_WIDTH_SECOND_V", [64, 128, 256])
    trial_config["ACTIVATION_FN"] = trial.suggest_categorical("ACTIVATION_FN", ["relu","tanh","leaky_relu"])
    trial_config["DISTANCE_REWARD_SCALE"] = trial.suggest_float("GOAL_BONUS", 1.0, 15.0)


    trial_config["TOTAL_TIMESTEPS"] = base_config["TOTAL_TIMESTEPS_HPO"]
    PHASE_1_BUDGET = base_config["TOTAL_TIMESTEPS_HPO"]

    seed = base_config.get("RANDOM_SEED", 77)

    CHECKPOINTS = [
        int(PHASE_1_BUDGET * 0.1),
        int(PHASE_1_BUDGET * 0.2),
        int(PHASE_1_BUDGET * 0.3),
        int(PHASE_1_BUDGET * 0.4),
        int(PHASE_1_BUDGET * 0.5),
        int(PHASE_1_BUDGET * 0.6),
        int(PHASE_1_BUDGET * 0.7),
        int(PHASE_1_BUDGET * 0.8),
        int(PHASE_1_BUDGET * 0.9),
        PHASE_1_BUDGET,  # 100% of Phase 1 budget
    ]
    trial_config["AULC_CHECKPOINTS"] = CHECKPOINTS


    # --- Setup unique paths and IDs ---
    trial_suffix = f"trial_{trial.number}"
    trial_config["MODEL_SAVE_PATH"] = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], trial_suffix)
    trial_config["LOG_DIR"] = os.path.join(base_config["LOG_DIR_BASE"], trial_suffix)
    trial_config["WANDB_RUN_NAME"] = f"t{trial.number}_lr{trial_config['LEARNING_RATE']:.1e}_ent{trial_config['ENT_COEF']:.2f}"
    with open(trial_config["ORACLE_ADATA_PATH"], 'rb') as f:
        ORACLE_ADATA = pickle.load(f)

    with open(trial_config["PERTURBABLE_GENES_PATH"], 'rb') as f:
        ORACLE_PERTURBABLE_GENES = pickle.load(f)

    with open(trial_config["TRANSITION_MATRIX"], 'rb') as f:
        _SHARED_TRANSITION_MATRIX = pickle.load(f)
    env_kwargs = dict(
        adata_object=ORACLE_ADATA,
        oracle_perturbable_genes=ORACLE_PERTURBABLE_GENES,
        transition_matrix_object=_SHARED_TRANSITION_MATRIX,
        max_steps=trial_config["MAX_STEPS_PER_EPISODE"],
        step_penalty=trial_config["STEP_PENALTY"],
        goal_bonus=trial_config["GOAL_BONUS"],
        discount_factor=trial_config["GAMMA"],
        allow_gene_activation=trial_config["ALLOW_GENE_ACTIVATION"],
        distance_reward_scale=trial_config.get("DISTANCE_REWARD_SCALE", 1),
        number_of_targets_curriculum=trial_config.get("TARGET_CELLS_PER_PHASE", 4),
        add_noise=trial_config.get("ADD_NOISE", False),
        test_frac=trial_config.get("TEST_FRAC", 0.1),
        val_frac=trial_config.get("VAL_FRAC", 0.1),
        seed=trial_config.get("RANDOM_SEED", 77),
        step_increase_per_phase=trial_config.get("MAX_STEP_INCREASE_PER_PHASE", 5),
        pgc_only_curriculum=False,
    )
    trial_config["optuna_trial_obj"] = trial
    AUC_Callback = OptunaCallBackAULC(checkpoints=CHECKPOINTS,
                                      eval_env_params = env_kwargs,
                                      trial=trial,
                                      n_eval_eps =1000)
    externall_callbacks = [AUC_Callback]



    os.makedirs(trial_config["MODEL_SAVE_PATH"], exist_ok=True)
    os.makedirs(trial_config["LOG_DIR"], exist_ok=True)

    print(f"\n---> Starting Optuna Trial {trial.number} <---")
    print("  --- Hyperparameters for this Trial ---")
    print(f"  PPO Agent:")
    print(f"    Learning Rate: {trial_config['LEARNING_RATE']:.6f}")  # More precision
    print(f"    N_Steps: {trial_config['PPO_N_STEPS']}")
    print(f"    N_Epochs: {trial_config['PPO_N_EPOCHS']}")
    print(f"    Clip Range: {trial_config['CLIP_RANGE']:.4f}")
    print(f"    Entropy Coef: {trial_config['ENT_COEF']:.6f}")
    print(f"    Gamma (Discount): {trial_config['GAMMA']:.4f}")

    

    print(f"  Reward Function:")
    print(f"    Goal Bonus: {trial_config['GOAL_BONUS']:.4f}")
    print(f"    Step Penalty: {trial_config['STEP_PENALTY']:.4f}")
    print(f"    Distance Reward Scale: {trial_config['DISTANCE_REWARD_SCALE']:.4f}")
    print("  ------------------------------------")


    metric_to_optimize = 0.0
    trial_status = "unknown"
    try:
        trained_model = ai_module.run_training(trial_config, externall_callbacks=externall_callbacks)
        final_aulc_score = AUC_Callback.compute_aulc()
        print(f"Trial {trial.number} completed. Final AULC score: {final_aulc_score:.4f}")
        trial_status = "completed"
        return final_aulc_score


    except optuna.exceptions.TrialPruned:
        trial_status = "pruned"
        print(f"Optuna Trial {trial.number} pruned.")
        raise

    except Exception as e:
        trial_status = "failed"
        print(f"Optuna Trial {trial.number} FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_optuna_hpo(base_config, gpu_id=None):
    try:
        study_name = base_config["STUDY_NAME"]
        storage_name = base_config["STORAGE_NAME"]

        os.makedirs(base_config["MODEL_SAVE_PATH_BASE"], exist_ok=True)
        os.makedirs(base_config["LOG_DIR_BASE"], exist_ok=True)

        hpo_total_sb3_timesteps = base_config["TOTAL_TIMESTEPS_HPO"]
        assumed_reports_per_trial = max(1, hpo_total_sb3_timesteps // 20000)

        hpo_report_interval = base_config.get("OPTUNA_REPORT_INTERVAL", 1000)  # Get from config
        min_timesteps_before_pruning = hpo_total_sb3_timesteps * 0.4

        n_warmup_steps_calculated = min_timesteps_before_pruning // hpo_report_interval

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=min_timesteps_before_pruning,
            interval_steps=1
        )

        sampler = optuna.samplers.TPESampler(
            seed=base_config.get("RANDOM_SEED", 77),
            n_startup_trials=10
        )

        #check if study exists and load otherwise create new one
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False
        )



        n_hpo_trials_to_run = 100

        print("running number of trials:", n_hpo_trials_to_run)
        study_timeout_seconds = 3600 * 768
        print(f"\n--- Starting Optuna HPO Study: {study_name} ---")
        try:
            print("test")
            study.optimize(
                lambda trial: objective(trial, base_config),
                n_trials=n_hpo_trials_to_run,
                timeout=study_timeout_seconds,
                show_progress_bar=False,
            )
        except Exception as e_study:
            print(f"ERROR weird: {e_study}")
            import traceback
            traceback.print_exc()

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("No trials completed successfully. Cannot determine best configurations.")
            return

        completed_trials.sort(key=lambda t: t.value, reverse=True)

        N_TOP_CONFIGS = 5
        top_n_trials = completed_trials[:N_TOP_CONFIGS]
        #how many trials are purned?


        print(f"\n--- Top {len(top_n_trials)} Hyperparameter Configurations ---")

        # Prepare a list of  to save, if needed
        top_n_params_list = []

        for i, trial in enumerate(top_n_trials):
            print(f"\n--- Rank {i + 1} (Trial #{trial.number}) ---")
            print(f"  Value (AULC): {trial.value:.6f}")
            print("  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            top_n_params_list.append(trial.params)

        save_path_txt = os.path.join(base_config["MODEL_SAVE_PATH_BASE"],
                                     f"{study_name}_top_{N_TOP_CONFIGS}_params.txt")
        with open(save_path_txt, "w") as f:
            f.write(f"Top {len(top_n_trials)} configurations from Optuna study '{study_name}'\n")
            f.write("=" * 50 + "\n")
            for i, trial in enumerate(top_n_trials):
                f.write(f"\n--- Rank {i + 1} | Trial #{trial.number} | Value (AULC): {trial.value:.6f} ---\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
        print(f"\nSummary of top configurations saved to {save_path_txt}")

        save_path_pkl = os.path.join(base_config["MODEL_SAVE_PATH_BASE"],
                                     f"{study_name}_top_{N_TOP_CONFIGS}_params.pkl")
        with open(save_path_pkl, "wb") as f:
            pickle.dump(top_n_params_list, f)
        print(f"Top {len(top_n_trials)} configurations also saved to {save_path_pkl} for programmatic access.")

        print("\n--- Optuna Study Finished ---")
        print(f"Study dashboard command: optuna-dashboard {storage_name}")
        print(f"Number of finished trials in study: {len(study.trials)}")

        # try:
        #     pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        #     complete_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        #     failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        #     print(f"  Trials: Completed={complete_trials}, Pruned={pruned_trials}, Failed={failed_trials}")
        #
        #     if complete_trials > 0:
        #         best_trial = study.best_trial
        #         print("\nBest trial:")
        #         print(f"  Value (Maximized Metric): {best_trial.value:.4f}")
        #         print("  Params: ")
        #         for key, value in best_trial.params.items():
        #             print(f"    {key}: {value}")
        #
        #         # Save best parameters to a file
        #         best_params_filename = os.path.join(base_config["MODEL_SAVE_PATH_BASE"],
        #                                             f"{study_name}_best_params.txt")
        #         with open(best_params_filename, "w") as f:
        #             f.write(f"Study Name: {study_name}\n")
        #             f.write(f"Best trial number: {best_trial.number}\n")
        #             f.write(f"Best trial value: {best_trial.value}\n\n")
        #             f.write("Best Hyperparameters:\n")
        #             for key, value in best_trial.params.items():
        #                 f.write(f"  {key}: {value}\n")
        #         print(f"Best parameters saved to {best_params_filename}")
        #     else:
        #         print("No trials completed successfully, cannot determine the best one.")
        #
        # except ValueError:
        #     print("ERROR: No completed trials found.")
        # except Exception as e_results:
        #     print("WRONG", e_results)
    except Exception as e_outer:
        print("ERROR OUTER", e_outer)
        import traceback
        traceback.print_exc()
