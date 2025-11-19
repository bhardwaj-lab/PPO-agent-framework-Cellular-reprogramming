import sys
import os
import multiprocessing
import time
from datetime import datetime
import logging
sys.path.append("/home/caspar/thesis_code/CellOracle")
sys.path.append("/home/caspar/thesis_code/complete_project/py files")
sys.path.append("/home/caspar/thesis_code/complete_project/py files/AIFiles")
sys.path.append("/home/caspar/thesis_code/complete_project/py files/baseGRNConstructionFiles")
sys.path.append("/home/caspar/thesis_code/complete_project/py files/oracleInferenceFiles")
sys.path.append("/home/caspar/thesis_code/complete_project/py files/oracleSetup")
# ==============================================================================
# --- FIX #1: ROBUST PATH MANAGEMENT ---
# Do not use os.chdir. Instead, define a project root based on this file's location
# and add all necessary subdirectories to the Python path dynamically.
# This makes the script runnable from anywhere.
# ==============================================================================
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Detected PROJECT_ROOT: {PROJECT_ROOT}")
except NameError:

    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../..")) # Fallback
    print("something else: ", PROJECT_ROOT)
# 2. Define the Data Root based on the Project Root's location
# This finds the parent directory (/home/caspar) and then joins it with 'celloracle_data'
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), 'celloracle_data')

print(f"Project Root detected: {PROJECT_ROOT}")

# Add all necessary project paths


print("Current sys.path:")
for p in sys.path:
    print(f"  {p}")



# --- All your imports can now safely run ---
import torch.nn as nn
import optuna
import AI as ai_module
import ParameterTuningForRunPY as tuning_module


# ==============================================================================
# --- FIX #2: JUPYTER MAGIC REMOVED ---
# The %autoreload lines were here. They are invalid syntax and have been deleted.
# ==============================================================================

# def setup_logging(process_id="main"):
#     """Sets up logging for each process to avoid file conflicts."""
#     log_dir = os.path.join(PROJECT_ROOT, 'logs')
#     os.makedirs(log_dir, exist_ok=True)
#     log_filename = os.path.join(log_dir, f"hpo_process_{process_id}_{datetime.now().strftime('%Y%m%d')}.log")
#
#     logging.basicConfig(
#         filename=log_filename,
#         filemode='a',
#         level=logging.INFO,
#         format=f'%(asctime)s - PROCESS {process_id} - %(levelname)s - %(message)s'
#     )
#     logging.info(f"Logging configured for process {process_id}")


# This is the target for each worker process. It's correctly at the top level.
def worker_process(gpu_id, config):
    # Each worker gets its own log file
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"Worker started on process {os.getpid()}, assigned to physical GPU {gpu_id}")
    tuning_module.run_optuna_hpo(config, gpu_id=gpu_id)


def start():
    # This MUST be the first call in the function that launches processes.
    multiprocessing.set_start_method('fork', force=True)

    data_path_new_data = os.path.join(DATA_ROOT,
                                      "celloracle_object/new_promoter_without_mescs_trimmed_test_own_umap")
    base_config = {
        # Paths
        "ORACLE_PATH": os.path.join(data_path_new_data, "ready_oracle.pkl"),
        "MODEL_SAVE_PATH_BASE": os.path.join(data_path_new_data,"optuna" ,"hpo_ppo_models2"),
        "LOG_DIR_BASE": os.path.join(data_path_new_data, "optuna", "hpo_ppo_logs2"),
        "WAND_B_LOG_DIR_BASE": os.path.join(data_path_new_data, "optuna", "hpo_ppo_wandb_logs2"),
        # Base for HPO trial wandb logs
        "HPO_FIXED_MAX_STEPS_PER_EPISODE": 20,
        "TRANSITION_MATRIX": os.path.join(data_path_new_data,"transition_matrix", "transition_matrix.pkl"),
        # Path to transition matrix

        "BATCH_SIZE": 256,
        "USE_MASKABLE_PPO": False,
        "VERBOSE": 0,
        "LOG_INTERVAL": 10,
        "RESET_NUM_TIMESTEPS": False,
        "DEVICE": "auto",
        "USE_PROGRESS_BAR": False,
        "STEP_SAVE_FREQ": 1000000000,
        "PPO_N_STEPS": 512,
        "PPO_N_EPOCHS": 10,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_RANGE": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LEARNING_RATE": 3e-4,

        # Environment Specific
        "ALLOW_GENE_ACTIVATION": True,
        "STANDARD_SD_FACTOR": 1.5,
        "USE_LEARNING_RATE_SCHEDULE": False,
        "USE_SIMILARITY_REWARD": True,
        "ENV_TYPE": "FAST",  # options: FIRST,NEW_INPUT, GENERAL,FAST
        "USE_PREV_KNOCKOUT": False,

        # ENV HYPERPARAMETERS (Some will be tuned by Optuna)
        "DIVIDER_OF_TOTAL_STEPS_FOR_CURRICULUM_LEARNING": 20,
        "TARGET_CELLS_PER_PHASE": 14,
        "MAX_STEP_INCREASE_PER_PHASE": 8,
        "MAX_STEPS_FIRST_PHASE": 50000,
        "PHASE_STEP_INCREASE": 15000,
        "MAX_STEPS_PER_EPISODE": 42,
        "OPTUNA_REPORT_INTERVAL": 10000,

        # Other ENV params
        "GENE_ACTIVITY_THRESHOLD": 0.01,
        "TARGET_DISTANCE_THRESHOLD": 0.1,
        "DISTANCE_GAMMA": 0.99,
        "STEP_PENALTY": -0.05,
        "GOAL_BONUS": 10,
        "FAIL_PENALTY": -3,
        "SAME_CELL_PENALTY": -0.5,
        "DISTANCE_REWARD_SCALE": 5,
        "PARAMETER_TUNING": True,
        "TOTAL_TIMESTEPS_HPO": 100000,
        "STUDY_NAME": "ppo_celloracle_hpo_general3",
        "STORAGE_NAME": "sqlite:///ppo_celloracle_hpo_general3.db",

        # NN Architecture (Fixed for this example, but can be tuned)
        "PI_ARCH": [512, 256],
        "VF_ARCH": [512, 256],
        "ACTIVATION_FN": nn.ReLU,  # Make sure nn is imported if you use this directly

    }

    num_gpus = 2
    processes = []
    for gpu_id in range(num_gpus):
        logging.info(f"Main process: starting worker for GPU {gpu_id}...")
        p = multiprocessing.Process(target=worker_process, args=(gpu_id, base_config.copy()))
        processes.append(p)
        p.start()
        time.sleep(5)


    # ==============================================================================
    # --- FIX #3: IMPLEMENT THE MANAGER-WORKER LOGIC ---
    # This loop monitors the study and terminates workers when the target
    # number of trials is reached.
    # ==============================================================================
    n_total_trials_to_run = 2  # <-- SET YOUR GLOBAL GOAL HERE

    time.sleep(15)  # Give workers time to start and create the .db file

    try:
        study = optuna.load_study(
            study_name=base_config["STUDY_NAME"],
            storage=base_config["STORAGE_NAME"]
        )
        while True:
            finished_trials = study.get_trials(deepcopy=False, states=[
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL
            ])
            num_finished = len(finished_trials)
            if num_finished >= n_total_trials_to_run:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                break
            time.sleep(60)
    except Exception as e:
        # In case of error, still try to clean up
        for p in processes:
            if p.is_alive():
                p.terminate()

    # Wait for all terminated processes to be cleaned up
    for p in processes:
        p.join()

    logging.info("All worker processes terminated. Main script exiting.")
    final_study = optuna.load_study(
        study_name=base_config["STUDY_NAME"],
        storage=base_config["STORAGE_NAME"]
    )

    pruned_trials = len([t for t in final_study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete_trials = len([t for t in final_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed_trials = len([t for t in final_study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"  Total Trials: Completed={complete_trials}, Pruned={pruned_trials}, Failed={failed_trials}")

    if complete_trials > 0:
        best_trial = final_study.best_trial
        print("\nBest trial:")
        print(f"  Value (Maximized Metric): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Save best parameters to a file
        best_params_filename = os.path.join(base_config["MODEL_SAVE_PATH_BASE"],
                                            f"{final_study.study_name}_best_params.txt")
        with open(best_params_filename, "w") as f:
            f.write(f"Study Name: {final_study.study_name}\n")
            f.write(f"Best trial number: {best_trial.number}\n")
            f.write(f"Best trial value: {best_trial.value}\n\n")
            f.write("Best Hyperparameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        print(f"Best parameters saved to {best_params_filename}")
    else:
        print("No trials completed successfully, cannot determine the best one.")

    print("Main script is now exiting.")


if __name__ == '__main__':
    # Setup logging for the main manager process
    start()