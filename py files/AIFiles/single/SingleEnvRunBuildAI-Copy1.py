import os
from datetime import datetime
import pickle
from sched import scheduler
from typing import Callable, Dict, Any, List, Optional

import gymnasium as gym
import optuna

import torch.nn as nn
import wandb
import cupy as cp

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from FinalEnvSingleInstance import CellOracleGymEnv as Env
from multiprocessing import set_start_method


def run_training(config: dict):
    """
    Sets up the standard Gymnasium environment, PPO agent, and runs the training loop.
    """
    try:
        set_start_method("fork")
        print("--- Multiprocessing start method set to 'fork'. ---")
    except RuntimeError:
        print("--- Multiprocessing start method was already set. ---")

    try:
        cp.cuda.Device(0).use()
        print("CuPy is available and GPU is accessible.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        # Handle the case where CuPy cannot access the GPU
        print("CuPy cannot access GPU, falling back to CPU integration.")

    print("gpu available: " + str(torch.cuda.is_available()))
    wandbd_run_id = config.get("WANDB_RUN_ID", None)
    wand_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    project_name_wandb = "test_runs"
    if config.get("optuna_trial_obj") is not None:
        project_name_wandb = "celloracle optuna opt"

    with open(config["ORACLE_PATH"], 'rb') as f:
        _SHARED_ORACLE = pickle.load(f)

    with open(config["TRANSITION_MATRIX"], 'rb') as f:
        _SHARED_TRANSITION_MATRIX = pickle.load(f)

    wandb.init(
        project=project_name_wandb,
        config=config,
        name=wand_name,
        sync_tensorboard=True,  # Syncing is helpful here
        id=wandbd_run_id,
        reinit=True,
        dir=config['WAND_B_LOG_DIR_BASE']
    )

    print("--- Starting Training Run with Standard Gym Env ---")
    print(f"Configuration: {config}")
    os.makedirs(config["LOG_DIR"], exist_ok=True)

    # --- 2. Environment Setup (Using make_vec_env) ---
    print(f"Initializing {config['N_ENVS']} parallel CellOracleGymEnv environments...")

    # Parameters for the environment's __init__ method
    env_kwargs = dict(
        oracle_path=config["ORACLE_PATH"],
        transition_path=config["TRANSITION_MATRIX"],
        max_steps=config["MAX_STEPS_PER_EPISODE"],
        step_penalty=config["STEP_PENALTY"],
        goal_bonus=config["GOAL_BONUS"],
        fail_penalty=config["FAIL_PENALTY"],
        distance_reward_scale=config["DISTANCE_REWARD_SCALE"],
        allow_gene_activation=config["ALLOW_GENE_ACTIVATION"],
        gamma_distance=config.get("DISTANCE_GAMMA", 0.99),
        number_of_targets_curriculum=config.get("TARGET_CELLS_PER_PHASE", 4),
        same_cell_penalty=config.get("SAME_CELL_PENALTY", -0.1),
        use_similarity_for_pbrs=config.get("USE_SIMILARITY_REWARD", False),
    )

    # make_vec_env creates a vectorized environment from your Gym env
    # It automatically wraps each instance with a Monitor for episode stats
    print("does this step take so long")
    env = make_vec_env(
        Env,
        n_envs=config['N_ENVS'],
        env_kwargs=env_kwargs,
    )
    print("Environment initialization complete.")

    # --- 3. Agent Setup ---
    policy_kwargs = dict(
        net_arch=dict(
            pi=config.get("PI_ARCH", [64, 64]),
            vf=config.get("VF_ARCH", [64, 64])
        ),
        activation_fn=config.get("ACTIVATION_FN", nn.ReLU)
    )

    curriculum_finished_flag = [False]
    doing_param_opt = config.get("HPO_MODE", False)
    scheduler = None
    print("doing param opt:", doing_param_opt)
    if doing_param_opt:
        print("linear schedule????")
        scheduler = config.get("LEARNING_RATE", 3e-4)
    else:
        print('cur scheudler??')
        scheduler = CurriculumAwareScheduler(
            initial_lr=int(config.get("LEARNING_RATE", 3e-4)),
            end_lr=int(config.get("END_LEARNING_RATE", 1e-6)),
            total_timesteps=config["TOTAL_TIMESTEPS"],
            curriculum_finished_flag=curriculum_finished_flag
        )

    AgentClass = PPO

    model = AgentClass(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=config.get("VERBOSE", 1),
        learning_rate=scheduler,
        n_steps=config["PPO_N_STEPS"],
        batch_size=config["PPO_BATCH_SIZE"],
        n_epochs=config["PPO_N_EPOCHS"],
        gamma=config.get("GAMMA", 0.99),
        gae_lambda=config.get("GAE_LAMBDA", 0.95),
        clip_range=config.get("CLIP_RANGE", 0.2),
        ent_coef=config.get("ENT_COEF", 0.0),
        vf_coef=config.get("VF_COEF", 0.5),
        max_grad_norm=config.get("MAX_GRAD_NORM", 0.5),
        tensorboard_log=config["LOG_DIR"],
        device=config.get("DEVICE", "auto")
    )

    wandb.watch(model.policy, log="gradients", log_freq=config.get("LOG_FREQ", 1000))

    # --- 4. Callbacks Setup ---
    callbacks = []

    # Curriculum Callback
    curriculum_callback = CurriculumCallback(
        initial_phase_duration=config.get("MAX_STEPS_FIRST_PHASE"),
        max_step_increase_per_phase=config["MAX_STEP_INCREASE_PER_PHASE"],
        phase_duration_increase=config["PHASE_STEP_INCREASE"],
        curriculum_finished_flag=curriculum_finished_flag,
        verbose=1,
        using_hpo=doing_param_opt
    )
    callbacks.append(curriculum_callback)

    # Logging Callback
    logging_callback = WandbLoggingCallback(
        log_freq=config.get("LOG_FREQ", 1000),
        verbose=1
    )
    callbacks.append(logging_callback)

    # Checkpoint Callback
    os.makedirs(config['MODEL_SAVE_PATH'], exist_ok=True)
    checkpoint_callback = CustomCheckpointCallbackWithStates(
        save_freq=config["STEP_SAVE_FREQ"],
        save_path=config['MODEL_SAVE_PATH'],
        name_prefix="ppo_model_gym",
        curriculum_callback=curriculum_callback,
        config=config
    )
    callbacks.append(checkpoint_callback)

    # Optuna Pruning Callback (if used)
    optuna_trial_obj = config.get("optuna_trial_obj")
    print("optuna trial obj:", optuna_trial_obj)
    if optuna_trial_obj:
        optuna_callback = SB3OptunaPruningCallback(trial=optuna_trial_obj)
        callbacks.append(optuna_callback)

    # --- 5. Training ---
    print(f"Starting training for {config['TOTAL_TIMESTEPS']} timesteps...")
    try:
        print(f" PPO model was successfully placed on device: {model.device}")
        print("starting learning")
        model.learn(
            total_timesteps=config["TOTAL_TIMESTEPS"],
            callback=callbacks,
            reset_num_timesteps=False
        )
        model.save(os.path.join(config['MODEL_SAVE_PATH'], "final_model_gym.zip"))
    except optuna.exceptions.TrialPruned:
        print("Trial pruned by Optuna. Exiting training.")
    finally:
        env.close()
        wandb.finish()

    return model


class CurriculumAwareScheduler:
    """
    A learning rate scheduler that holds the LR constant until the curriculum is finished,
    then begins a linear decay to a final value.
    """
    def __init__(self, initial_lr: float, end_lr: float, total_timesteps: int, curriculum_finished_flag: List[bool]):
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_timesteps = total_timesteps
        self.curriculum_finished_flag = curriculum_finished_flag
        self.decay_start_step = -1  # Flag to record when decay should start

    def __call__(self, progress_remaining: float) -> float:
        if not self.curriculum_finished_flag[0]:
            return self.initial_lr

        # --- Part 2: Start and manage decay after curriculum ends ---
        current_timestep = self.total_timesteps * (1.0 - progress_remaining)

        # On the first step after curriculum is finished, record the start time of decay
        if self.decay_start_step == -1:
            self.decay_start_step = round(current_timestep)

        decay_period_duration = self.total_timesteps - self.decay_start_step
        if decay_period_duration <= 0:
            return self.end_lr  # Avoid division by zero if decay starts at the end

        timesteps_into_decay = current_timestep - self.decay_start_step
        progress_in_decay = min(1.0, timesteps_into_decay / decay_period_duration)

        # Linearly interpolate between the initial and end LR
        current_lr = self.initial_lr - progress_in_decay * (self.initial_lr - self.end_lr)
        return max(current_lr, self.end_lr)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class WandbLoggingCallback(BaseCallback):
    """A simplified callback for logging to W&B with a standard Gym Env."""

    def __init__(self, log_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.total_episodes = 0
        self.total_successes = 0

    def _on_step(self) -> bool:
        # Log episode stats when an episode finishes
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.total_episodes += 1
                # The Monitor wrapper adds the "episode" key to info
                info = self.locals["infos"][i]
                if info.get("goal_reached", False):
                    self.total_successes += 1

        # Log training metrics at the specified frequency
        if self.n_calls % self.log_freq == 0:
            log_dict = {}

            # Standard SB3 rollout metrics from the Monitor wrapper
            if 'rollout/ep_rew_mean' in self.logger.name_to_value:
                log_dict["rollout/ep_reward_mean"] = self.logger.name_to_value['rollout/ep_rew_mean']
            if 'rollout/ep_len_mean' in self.logger.name_to_value:
                log_dict["rollout/ep_length_mean"] = self.logger.name_to_value['rollout/ep_len_mean']

            # Overall success rate
            if self.total_episodes > 0:
                log_dict["episode_overall/success_rate"] = self.total_successes / self.total_episodes
            log_dict["episode_overall/total_episodes"] = self.total_episodes

            # Training metrics
            log_dict["train/learning_rate"] = self.model.learning_rate
            if 'train/value_loss' in self.logger.name_to_value:
                log_dict["train/value_loss"] = self.logger.name_to_value['train/value_loss']
            if 'train/policy_loss' in self.logger.name_to_value:
                log_dict["train/policy_loss"] = self.logger.name_to_value['train/policy_loss']

            # Environment curriculum state (get from the first env)
            current_phase = self.training_env.get_attr('current_phase')[0]
            max_steps = self.training_env.get_attr('max_steps')[0]
            log_dict["curriculum/current_phase"] = current_phase
            log_dict["curriculum/max_episode_steps"] = max_steps

            wandb.log(log_dict, step=self.num_timesteps)
        return True


class CurriculumCallback(BaseCallback):
    """Manages curriculum learning for the vectorized standard Gym environment."""

    def __init__(self, initial_phase_duration: int, phase_duration_increase: int,
                 max_step_increase_per_phase: int, curriculum_finished_flag: List[bool], verbose: int = 0, using_hpo: bool = False):
        super().__init__(verbose)
        self.steps_required_for_current_phase = initial_phase_duration
        self.phase_duration_increase = phase_duration_increase
        self.max_step_increase_per_phase = max_step_increase_per_phase
        self.curriculum_finished_flag = curriculum_finished_flag
        self.timesteps_at_current_phase_start = 0
        self.current_phase = 1
        self.curriculum_finished_flag[0] = False
        self.using_hpo = using_hpo

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "timesteps_at_current_phase_start": self.timesteps_at_current_phase_start,
            "steps_required_for_current_phase": self.steps_required_for_current_phase,
        }

    def set_state(self, state: Dict[str, Any]):
        self.current_phase = state.get("current_phase", 1)
        self.timesteps_at_current_phase_start = state.get("timesteps_at_current_phase_start", 0)
        self.steps_required_for_current_phase = state.get("steps_required_for_current_phase",
                                                          self.steps_required_for_current_phase)

    def _on_step(self) -> bool:
        if self.curriculum_finished_flag[0]:
            return True
        if self.using_hpo:
            return True

        time_in_phase = self.num_timesteps - self.timesteps_at_current_phase_start
        if self.num_timesteps > self.timesteps_at_current_phase_start and time_in_phase >= self.steps_required_for_current_phase:
            next_phase = self.current_phase + 1

            # Use env_method to call 'set_phase' on all parallel environments
            results = self.training_env.env_method('set_phase', next_phase, self.max_step_increase_per_phase)

            # Check if the phase change was successful (at least in the first env)
            if results and results[0]:
                self.current_phase = next_phase
                self.timesteps_at_current_phase_start = self.num_timesteps
                self.steps_required_for_current_phase += self.phase_duration_increase
                if self.verbose > 0:
                    print(f"Advanced to Curriculum Phase {self.current_phase} at timestep {self.num_timesteps}")
            else:
                self.curriculum_finished_flag[0] = True
                print("Curriculum has finished.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if self.verbose > 0:
                    print(f"Curriculum finished at phase {self.current_phase}. No further phases available.")
        return True


class CustomCheckpointCallbackWithStates(CheckpointCallback):
    """Saves model, curriculum state, and env state."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str,
                 curriculum_callback: CurriculumCallback, config: Dict):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        self.curriculum_callback = curriculum_callback
        self.config = config

    def _on_step(self) -> bool:
        # Let the parent class handle the model saving logic
        continue_training = super()._on_step()

        # If the parent saved the model, we save our states
        if continue_training and self.n_calls % self.save_freq == 0:
            model_filename_base = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")

            # Use env_method to get state from underlying envs, save the first one
            env_states = self.training_env.env_method('get_env_state')
            if env_states:
                with open(f"{model_filename_base}_env_state.pkl", "wb") as f:
                    pickle.dump(env_states[0], f)

            curriculum_state = self.curriculum_callback.get_state()
            with open(f"{model_filename_base}_curriculum_state.pkl", "wb") as f:
                pickle.dump(curriculum_state, f)

            with open(f"{model_filename_base}_config.pkl", "wb") as f:
                pickle.dump(self.config, f)

        return continue_training


class SB3OptunaPruningCallback(BaseCallback):
    """Stops training early if Optuna's pruner suggests it."""

    def __init__(self, trial: optuna.Trial, report_interval_steps: int = 5000):
        super().__init__()
        self.trial = trial
        self.report_interval_steps = report_interval_steps
        self.last_report_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_report_step >= self.report_interval_steps:
            self.last_report_step = self.num_timesteps
            # Use the mean reward logged by the Monitor wrapper
            ep_rew_mean = self.logger.name_to_value.get('rollout/ep_rew_mean', 0.0)
            self.trial.report(ep_rew_mean, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return True


if __name__ == '__main__':
    # --- Example Configuration ---
    config = {
        # --- Paths ---
        "ORACLE_PATH": "path/to/your/oracle.pkl",
        "TRANSITION_MATRIX": "path/to/your/transition_matrix.pkl",
        "LOG_DIR": "./logs/ppo_gym_env/",
        "MODEL_SAVE_PATH": "./models/ppo_gym_env/",
        "WAND_B_LOG_DIR_BASE": "./wandb_logs/",

        # --- Training Params ---
        "TOTAL_TIMESTEPS": 1_000_000,
        "N_ENVS": 8,  # Number of parallel environments
        "DEVICE": "cuda",
        "LEARNING_RATE": 3e-4,

        # --- PPO Agent Params ---
        "PPO_N_STEPS": 2048,
        "PPO_BATCH_SIZE": 64,
        "PPO_N_EPOCHS": 10,
        "GAMMA": 0.99,

        # --- Environment Params ---
        "MAX_STEPS_PER_EPISODE": 50,
        "STEP_PENALTY": -0.01,
        "GOAL_BONUS": 1.0,
        "FAIL_PENALTY": -1.0,
        "DISTANCE_REWARD_SCALE": 1.0,
        "SAME_CELL_PENALTY": -0.2,
        "ALLOW_GENE_ACTIVATION": False,

        # --- Curriculum Params ---
        "MAX_STEPS_FIRST_PHASE": 50000,
        "PHASE_STEP_INCREASE": 20000,
        "MAX_STEP_INCREASE_PER_PHASE": 5,
        "TARGET_CELLS_PER_PHASE": 4,

        # --- Logging ---
        "LOG_FREQ": 2048,
        "STEP_SAVE_FREQ": 50000,
        "VERBOSE": 1,
    }

    run_training(config)