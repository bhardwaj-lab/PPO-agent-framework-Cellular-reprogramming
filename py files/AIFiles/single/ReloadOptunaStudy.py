# File: analyze_hpo_results.py

import optuna
import os
import pickle

# --- Configuration ---
# These MUST match the values from your main HPO script
STUDY_NAME = "hpo_study_final"
STORAGE_NAME = "sqlite:///hpo_study_final.db"
SAVE_PATH_BASE = "/path/to/save/models/"  # The same base path
N_TOP_CONFIGS_TO_SAVE = 5  # You can set this to 5, 10, or whatever you need


def analyze_and_save_results(study_name, storage_name, save_path, n_top):
    """
    Loads a completed Optuna study, analyzes its results, and saves the top
    N hyperparameter configurations.
    """
    print(f"--- Loading Optuna study '{study_name}' from '{storage_name}' ---")

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # --- Analysis of Trial States ---
    all_trials = study.trials
    completed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n--- Study Summary ---")
    print(f"Total trials in study: {len(all_trials)}")
    print(f"  - Completed: {len(completed_trials)}")
    print(f"  - Pruned:    {len(pruned_trials)}")
    print(f"  - Failed:    {len(failed_trials)}")

    if not completed_trials:
        print("\nNo trials completed successfully. Cannot save top configurations.")
        return

    # --- Get and Save Top N Configurations ---
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    top_n_trials = completed_trials[:n_top]

    print(f"\n--- Top {len(top_n_trials)} Hyperparameter Configurations ---")
    top_n_params_list = []

    for i, trial in enumerate(top_n_trials):
        print(f"\n--- Rank {i + 1} (Trial #{trial.number}) ---")
        print(f"  Value (AULC): {trial.value:.6f}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        top_n_params_list.append(trial.params)

    # --- Saving to files (identical logic to your main script) ---
    os.makedirs(save_path, exist_ok=True)

    # Save summary text file
    summary_path_txt = os.path.join(save_path, f"{study_name}_top_{n_top}_summary.txt")
    with open(summary_path_txt, "w") as f:
        f.write(f"Top {len(top_n_trials)} configurations from Optuna study '{study_name}'\n")
        f.write("=" * 50 + "\n")
        for i, trial in enumerate(top_n_trials):
            f.write(f"\n--- Rank {i + 1} | Trial #{trial.number} | Value (AULC): {trial.value:.6f} ---\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
    print(f"\nSummary of top configurations saved to {summary_path_txt}")

    # Save pickle file
    save_path_pkl = os.path.join(save_path, f"{study_name}_top_{n_top}_params.pkl")
    with open(save_path_pkl, "wb") as f:
        pickle.dump(top_n_params_list, f)
    print(f"Top configurations also saved to {save_path_pkl} for programmatic access.")


if __name__ == "__main__":
    analyze_and_save_results(
        study_name=STUDY_NAME,
        storage_name=STORAGE_NAME,
        save_path=SAVE_PATH_BASE,
        n_top=N_TOP_CONFIGS_TO_SAVE
    )