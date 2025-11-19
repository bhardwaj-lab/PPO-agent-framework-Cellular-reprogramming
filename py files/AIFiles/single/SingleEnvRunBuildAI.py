import os
from cgi import maxlen
from datetime import datetime
import pickle
from sched import scheduler
from typing import Callable, Dict, Any, List, Optional, Tuple, DefaultDict
from stable_baselines3.common.evaluation import evaluate_policy
from torch.optim import AdamW
import math
import gymnasium as gym
import optuna

import torch.nn as nn
import wandb
from collections import deque, defaultdict
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from FinalEnvSingleInstance import CellOracleGymEnv as Env
from FinalEnvSingleInstance import ActionMaskingWrapper as ActionMaskingWrapper
from multiprocessing import set_start_method
from CustomNeuralNetwork import Embedding
from CustomNeuralNetwork import MaskedActorCriticPolicy
#import the monitor from sb3
import stable_baselines3.common.monitor as Monitor


def run_training(config: dict, externall_callbacks:List[BaseCallback]=[]):
    """
    Sets up the standard Gymnasium environment, PPO agent, and runs the training loop.
    """

    print("gpu available: " + str(torch.cuda.is_available()))
    wand_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    project_name_wandb = "single_gym_env_runs"
    if config.get("optuna_trial_obj") is not None:
        project_name_wandb = "celloracle optuna opt 2"

    with open(config["ORACLE_ADATA_PATH"], 'rb') as f:
        ORACLE_ADATA = pickle.load(f)

    with open(config["PERTURBABLE_GENES_PATH"], 'rb') as f:
        ORACLE_PERTURBABLE_GENES = pickle.load(f)

    with open(config["TRANSITION_MATRIX"], 'rb') as f:
        _SHARED_TRANSITION_MATRIX = pickle.load(f)

    #set random seet
    random_seed = config.get("RANDOM_SEED", 77)
    np.random.seed(random_seed)
    settings = wandb.Settings(quiet=True)

    wandb.init(
        project=project_name_wandb,
        config=config,
        name=wand_name,
        sync_tensorboard=True,  # Syncing is helpful here
        reinit=False,
        dir=config['WAND_B_LOG_DIR_BASE'],
        settings=settings
    )

    print(f"Configuration: {config}")
    os.makedirs(config["LOG_DIR"], exist_ok=True)

    # --- 2. Environment Setup (Using make_vec_env) ---
    print(f"Initializing {config['N_ENVS']} parallel CellOracleGymEnv environments...")

    # Parameters for the environment's __init__ method
    env_kwargs = dict(
        adata_object = ORACLE_ADATA,
        oracle_perturbable_genes = ORACLE_PERTURBABLE_GENES,
        transition_matrix_object =  _SHARED_TRANSITION_MATRIX,
        max_steps=config["MAX_STEPS_PER_EPISODE"],
        step_penalty=config["STEP_PENALTY"],
        goal_bonus=config["GOAL_BONUS"],
        discount_factor=config["GAMMA"],
        allow_gene_activation=config["ALLOW_GENE_ACTIVATION"],
        distance_reward_scale=config.get("DISTANCE_REWARD_SCALE", 1),
        number_of_targets_curriculum=config.get("TARGET_CELLS_PER_PHASE", 4),
        add_noise = config.get("ADD_NOISE", False),
        test_frac = config.get("TEST_FRAC", 0.1),
        val_frac = config.get("VAL_FRAC", 0.1),
        seed = config.get("RANDOM_SEED", 77),
        step_increase_per_phase = config.get("MAX_STEP_INCREASE_PER_PHASE", 5),
        pgc_only_curriculum = False,
    )

    env = make_vec_env(
        Env,
        n_envs=config['N_ENVS'],
        env_kwargs=env_kwargs,
        wrapper_class=ActionMaskingWrapper,
    )
    ACTS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }

    policy_kwargs = dict(
        features_extractor_class=Embedding,
        features_extractor_kwargs=dict(
            features_dim=256
        ),
        net_arch=dict(
            pi=[config["NET_WIDTH_FIRST_P"], config["NET_WIDTH_SECOND_P"]],
            vf=[config["NET_WIDTH_FIRST_V"], config["NET_WIDTH_SECOND_V"]]
        ),
        activation_fn=ACTS.get(config.get("ACTIVATION_FN", "relu")),
        optimizer_class=AdamW,
        optimizer_kwargs=dict(
            weight_decay=1e-2,
        )
    )
    curriculum_finished_flag = [False]
    doing_param_opt = config.get("HPO_MODE", False)
    scheduler = None
    print("doing param opt:", doing_param_opt)
    if doing_param_opt:
        scheduler = config.get("LEARNING_RATE", 3e-4)
    else:
        scheduler = CurriculumAwareScheduler(
            initial_lr=config.get("LEARNING_RATE", 3e-4),
            end_lr=config.get("END_LEARNING_RATE", 1e-6),
            total_timesteps=config["TOTAL_TIMESTEPS"],
            curriculum_finished_flag=curriculum_finished_flag
        )

    scheduler = CurriculumAwareScheduler(
        initial_lr=config.get("LEARNING_RATE", 3e-4),
        end_lr=config.get("END_LEARNING_RATE", 1e-6),
        total_timesteps=config["TOTAL_TIMESTEPS"],
        curriculum_finished_flag=curriculum_finished_flag
    )
    AgentClass = PPO

    model = AgentClass(
        policy = MaskedActorCriticPolicy,
        env= env,
        policy_kwargs=policy_kwargs,
        verbose=0,
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
    # print("architecture model:")
    # print(model.policy)

    wandb.watch(model.policy, log="gradients", log_freq=config.get("LOG_FREQ", 1000))


    callbacks = []
    metric_calculator_callback = CallbackThatCalculatesAllImportantValuesEveryStep(eval_freq=40000, n_eval_eps=2000, eval_env_params=env_kwargs)
    callbacks.append(metric_calculator_callback)

    curriculum_callback = CurriculumCallback(
        initial_phase_duration=config.get("MAX_STEPS_FIRST_PHASE"),
        max_step_increase_per_phase=config["MAX_STEP_INCREASE_PER_PHASE"],
        phase_duration_increase=config["PHASE_STEP_INCREASE"],
        curriculum_finished_flag=curriculum_finished_flag,
        using_hpo=doing_param_opt
    )
    callbacks.append(curriculum_callback)

    logging_callback = WandbLoggingCallback(
        log_freq=config.get("LOG_FREQ", 1000),
        calculator_callback=metric_calculator_callback
    )
    callbacks.append(logging_callback)

    os.makedirs(config['MODEL_SAVE_PATH'], exist_ok=True)
    checkpoint_callback = CustomCheckpointCallbackWithStates(
        save_freq=config["STEP_SAVE_FREQ"],
        save_path=config['MODEL_SAVE_PATH'],
        name_prefix="ppo_model_gym",
        curriculum_callback=curriculum_callback,
        config=config
    )
    callbacks.append(checkpoint_callback)
    callbacks.extend(externall_callbacks)

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

class CallbackThatCalculatesAllImportantValuesEveryStep(BaseCallback):

    def __init__(self, eval_freq:int = 20000, n_eval_eps : int = 1000, eval_env_params:dict={}):
        super().__init__(verbose=0)

        # Periodic smoothed (last N eps = 1000) buffers
        self.success_history_roll = deque(maxlen=100)
        self.ep_length_history_roll = deque(maxlen=100)
        self.action_history_periodic = deque(maxlen=100)
        self.progress_made_period_periodic = deque(maxlen=100)
        self.periodic_stagnation = deque(maxlen=100)

        # Batch buffers, cleared after each update to the logger
        self.action_history_batch = deque()
        self.progress_made_period_batch = deque()
        self.cumulative_rewards_batch = defaultdict(float)

        # Overall statistic counters
        self.lifetime_total_episodes = 0
        self.lifetime_total_successes = 0

        #eval vars
        env = Env(**eval_env_params)
        env.switch_between_modes(is_eval_mode=True)
        self.eval_env= ActionMaskingWrapper(Monitor.Monitor(env, filename=None))
        self._success_buffer = []
        self.n_eval_eps = n_eval_eps
        self.next_eval_step = eval_freq
        self.eval_freq = eval_freq


    def _on_step(self) -> bool:
        # Store data in buffers from episodes
        actions = self.locals.get("actions", [])
        action_space_size = self.training_env.get_attr("action_space_size")[0]
        n_knockout = action_space_size // 2 if self.training_env.get_attr("allow_gene_activation")[
            0] else action_space_size

        for i in range(self.training_env.num_envs):
            info = self.locals["infos"][i]

            if "episode" in info:
                self.lifetime_total_episodes += 1
                is_success = info.get("goal_reached", False)
                if is_success:
                    self.lifetime_total_successes += 1
                self.ep_length_history_roll.append(info["episode"]["l"])
                self.success_history_roll.append(1.0 if is_success else 0.0)
                self.periodic_stagnation.append(1.0 if info.get("stagnated", False) else 0.0)

            if "progress_made" in info: self.progress_made_period_batch.append(info["progress_made"])
            if "progress_made" in info: self.progress_made_period_periodic.append(info["progress_made"])
            if "reward_components" in info:
                for key, value in info["reward_components"].items():
                    self.cumulative_rewards_batch[key] += value
            self.action_history_batch.append(1 if actions[i] >= n_knockout else 0)
            self.action_history_periodic.append(1 if actions[i] >= n_knockout else 0)

        # Calc and log metrics at each step

        # Do periodic metric
        if self.success_history_roll:
            self.logger.record("periodic/success_rate", np.mean(self.success_history_roll))
        if self.ep_length_history_roll:
            self.logger.record("periodic/ep_len_mean", np.mean(self.ep_length_history_roll))
        if self.action_history_periodic:
            self.logger.record("periodic/activation_fraction", np.mean(self.action_history_periodic))
        if self.progress_made_period_periodic:
            self.logger.record("periodic/avg_progress_per_step", np.mean(self.progress_made_period_periodic))
        if self.periodic_stagnation:
            self.logger.record("periodic/stagnation_fraction", np.mean(self.periodic_stagnation))


        #do batch update metrics:
        if self.action_history_batch:
            self.logger.record("batch/activation_fraction", np.mean(self.action_history_batch))
        self.logger.record("batch/cumulative_rewards", self.cumulative_rewards_batch)
        if self.progress_made_period_batch:
            self.logger.record("batch/avg_progress_per_step", np.mean(self.progress_made_period_batch))
        # Overall etmrics
        if self.lifetime_total_episodes > 0:
            self.logger.record("episode_overall/success_rate",self.lifetime_total_successes / self.lifetime_total_episodes)

        #handle eval
        if self.num_timesteps >= self.next_eval_step:
            current_phase = 0
            if self.training_env.num_envs > 0:
                current_phase = self.training_env.env_method("get_phase")[0]
            else:
                current_phase = self.training_env.get_phase()
            self.eval_env.set_phase(current_phase, 0)
            self._success_buffer = []
            self.eval_env.unwrapped.switch_between_modes(is_eval_mode=True)
            evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_eps,
                render=False,
                deterministic=True,
                callback=self._collect_successes_callback
            )
            print("Evaluation done, len success rates:", len(self._success_buffer))
            eval_success_rate = np.mean(self._success_buffer) if self._success_buffer else 0.0
            self.logger.record("eval/success_rate", eval_success_rate)
            self.next_eval_step += self.eval_freq
            self._success_buffer = []
        return True

    def _collect_successes_callback(self, locals_dict: Dict, globals_dict: Dict) -> None:
        """grab the feedback"""
        for done, info in zip(locals_dict['dones'], locals_dict['infos']):
            if done:
                self._success_buffer.append(1.0 if info.get("goal_reached", False) else 0.0)

    def clear_batch_data(self):
        """ Clears ONLY the buffers used for periodic/batch calculations. """
        self.action_history_batch.clear()
        self.cumulative_rewards_batch.clear()
        self.progress_made_period_batch.clear()

class CurriculumAwareScheduler:
    """
    A learning rate scheduler that holds the LR constant until the curriculum is finished,
    then begins a linear decay to a final value.
    """

    def __init__(self, initial_lr: float, end_lr: float, total_timesteps: int,
                 curriculum_finished_flag: List[bool]):
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
    """
    A pure logging callback. It reads pre-calculated metrics from the SB3 logger
    (provided by MetricCalculatorCallback) and logs them to Weights & Biases in a
    structured format. It performs NO calculations and has NO long-term state.
    """

    def __init__(self, calculator_callback: CallbackThatCalculatesAllImportantValuesEveryStep, log_freq: int):
        super().__init__(0)
        self.calculator_callback = calculator_callback
        self.log_freq = log_freq
        self._last_logged_step = 0

    def _on_step(self) -> bool:
        # Check if it's time to log
        if (self.num_timesteps - self._last_logged_step) < self.log_freq:
            return True

        # #just do this as it saves a shit ton of self.logger.nametovalue
        nv, log_dict = self.logger.name_to_value, {}

        # log the periodic metrics
        if 'periodic/success_rate' in nv: log_dict["rollout/success_rate"] = nv['periodic/success_rate']
        if 'periodic/ep_len_mean' in nv: log_dict["rollout/ep_length_mean"] = nv['periodic/ep_len_mean']
        if 'rollout/ep_rew_mean' in nv: log_dict["rollout/ep_reward_mean"] = nv['rollout/ep_rew_mean']  # From SB3
        if 'periodic/activation_fraction' in nv: log_dict["rollout/periodic_activation_fraction"] = nv[
            'periodic/activation_fraction']
        if 'periodic/avg_progress_per_step' in nv: log_dict["rollout/periodic_avg_progress_per_step"] = nv[
            'periodic/avg_progress_per_step']
        if 'periodic/stagnation_fraction' in nv: log_dict["rollout/stagnation_fraction"] = nv["periodic/stagnation_fraction"]

        # ppo traning important
        if 'train/learning_rate' in nv: log_dict["train/learning_rate"] = nv['train/learning_rate']
        if 'train/loss' in nv: log_dict["train/loss"] = nv['train/loss']  # Total Loss
        if 'train/policy_loss' in nv: log_dict["train/policy_loss"] = nv['train/policy_loss']
        if 'train/value_loss' in nv: log_dict["train/value_loss"] = nv['train/value_loss']  #
        if 'train/entropy_loss' in nv: log_dict["train/entropy_loss"] = nv['train/entropy_loss']
        if 'train/approx_kl' in nv: log_dict["train/approx_kl"] = nv['train/approx_kl']
        if 'train/clip_fraction' in nv: log_dict["train/clip_fraction"] = nv['train/clip_fraction']
        if 'train/explained_variance' in nv: log_dict["train/explained_variance"] = nv['train/explained_variance']

        # progress trackers
        batch_rewards = nv.get("batch/cumulative_rewards", {})
        total_reward_mag = sum(abs(v) for v in batch_rewards.values()) + 1e-9
        for key, value in batch_rewards.items():
            log_dict[f"progress_tracker/reward_composition/{key}_perc"] = (abs(value) / total_reward_mag) * 100
        log_dict["progress_tracker/reward_batch_total"] = sum(batch_rewards.values())
        log_dict["progress_tracker/curriculum_phase"] = self.training_env.get_attr('current_phase')[0]


        # batch behavior
        if 'batch/activation_fraction' in nv: log_dict["progress_tracker/batch_activation_fraction"] = nv[
            'batch/activation_fraction']
        if 'batch/avg_progress_per_step' in nv: log_dict["progress_tracker/batch_avg_progress_per_step"] = nv[
            'batch/avg_progress_per_step']
        if 'eval/success_rate' in nv: log_dict["eval/success_rate"] = nv['eval/success_rate']
        
        wandb.log(log_dict, step=self.num_timesteps)
        self._last_logged_step = self.num_timesteps
        #clear batch stuf
        self.calculator_callback.clear_batch_data()
        return True

class CurriculumCallback(BaseCallback):
    """Manages curriculum learning for the vectorized standard Gym environment."""

    def __init__(self, initial_phase_duration: int, phase_duration_increase: int,
                 max_step_increase_per_phase: int, curriculum_finished_flag: List[bool],
                 using_hpo: bool = False):
        super().__init__(0)
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
            else:
                print("Curriculum finished at phase", self.current_phase)
                self.curriculum_finished_flag[0] = True
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


class OptunaCallBackAULC(BaseCallback):
    """
    [REVISED AND CORRECTED]
    A callback for Optuna pruning that uses a dedicated evaluation environment
    and correctly synchronizes the curriculum phase for fair evaluation.
    """

    def __init__(self,
                 trial: optuna.Trial,
                 checkpoints: List[int],
                 eval_env_params: dict,  # <-- New required parameter
                 n_eval_eps: int = 1000,
                 ):
        super().__init__(verbose=0)
        self.trial = trial
        self.checkpoints = sorted(checkpoints)
        self.n_eval_eps = n_eval_eps

        # --- New: Create a dedicated evaluation environment ---
        print("Optuna Callback: Initializing dedicated evaluation environment...")
        eval_env = Env(**eval_env_params)
        eval_env.switch_between_modes(is_eval_mode=True)  # Set to eval mode once at creation
        # We wrap it in Monitor to ensure 'episode' info is available if needed, though we only need 'goal_reached'
        self.eval_env= ActionMaskingWrapper(Monitor.Monitor(eval_env, filename=None))
        # --------------------------------------------------------

        self._success_buffer: List[float] = []
        self.history: List[Tuple[int, float]] = []
        self.next_checkpoint_idx = 0

    def _on_step(self) -> bool:
        if self.next_checkpoint_idx >= len(self.checkpoints):
            return True

        checkpoint_step = self.checkpoints[self.next_checkpoint_idx]

        if self.num_timesteps < checkpoint_step:
            return True

        # --- New, Correct Evaluation Logic ---

        # 1. Get current curriculum phase from the main training environment
        current_phase = self.training_env.env_method("get_phase")[0]

        # 2. Set the dedicated evaluation environment to the same phase
        self.eval_env.set_phase(current_phase, 0)
        self.eval_env.unwrapped.switch_between_modes(is_eval_mode=True)

        print(f"\nOptuna Callback: Checkpoint reached at step {self.num_timesteps}.")
        print(f"Evaluating policy for phase {current_phase} over {self.n_eval_eps} episodes...")

        # 3. Run evaluation on the correctly configured eval_env
        self._success_buffer = []  # Reset buffer before each evaluation
        evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_eps,
            deterministic=True,
            render=False,
            callback=self._collect_successes_callback
        )

        eval_performance = np.mean(self._success_buffer) if self._success_buffer else 0.0
        print(f"Evaluation done. Success Rate: {eval_performance:.2%}")

        # 4. Report performance to Optuna for pruning
        self.history.append((self.num_timesteps, eval_performance))
        self.trial.report(eval_performance, self.num_timesteps)

        if self.trial.should_prune():
            print(f"Trial pruned by Optuna at step {self.num_timesteps} with success rate {eval_performance:.2%}.\n")
            raise optuna.exceptions.TrialPruned()

        self.next_checkpoint_idx += 1
        return True

    def _collect_successes_callback(self, locals_dict: Dict, globals_dict: Dict) -> None:
        """
        Callback to collect success status from each episode during evaluation.
        `evaluate_policy` wraps the env in a DummyVecEnv, so dones/infos are lists of size 1.
        """
        if locals_dict['dones'][0]:
            info = locals_dict['infos'][0]
            self._success_buffer.append(1.0 if info.get("goal_reached", False) else 0.0)

    def compute_aulc(self) -> float:
        """Computes the Area Under the Learning Curve for the final Optuna value."""
        # This method remains unchanged and is correct.
        if not self.history:
            return 0.0
        points = np.array(self.history)
        steps, perfs = points[:, 0], points[:, 1]
        if steps[0] != 0:
            steps = np.insert(steps, 0, 0)
            perfs = np.insert(perfs, 0, 0.0)
        area = np.trapz(y=perfs, x=steps)  # Corrected trapz usage
        duration = steps[-1]
        aulc = area / duration if duration > 0 else 0.0
        print(f"Final AULC for trial: {aulc}")
        return aulc

    def trapezoidal_area(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculates the area under the curve using the trapezoidal rule."""
        if len(x) < 2:
            return 0.0
        return np.trapz(y=y, x=x)


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


    # class WandbLoggingCallback(BaseCallback):
    #     """
    #     Final, comprehensive callback for logging detailed metrics to Weights & Biases.
    #     Includes robust reward composition percentage analysis.
    #     """
    #
    #     def __init__(self, log_freq: int, verbose: int = 0):
    #         super().__init__(verbose)
    #         self.log_freq = log_freq
    #         self.total_episodes, self.total_successes = 0, 0
    #         self.action_history = deque()
    #         self.stagnation_history = deque()
    #         self.initial_distances = deque()
    #         self.episode_lengths_period = deque()
    #         self.success_history = deque(maxlen=1000)
    #         self.progress_made_period = deque()
    #         self.cumulative_rewards_period = defaultdict(float)
    #         self.aulc_history: List[Tuple[int, float]] = []
    #         self._last_logged_step = 0
    #
    #     def _on_step(self) -> bool:
    #         for i, done in enumerate(self.locals.get("dones", [])):
    #             if done:
    #                 is_success = self.locals["infos"][i].get("goal_reached", False)
    #                 self.success_history.append(1.0 if is_success else 0.0)
    #         current_success_rate = np.mean(self.success_history) if self.success_history else 0.0
    #         self.logger.record("periodic/success_rate", current_success_rate)
    #         actions = self.locals.get("actions", [])
    #         action_space_size = self.training_env.get_attr("action_space_size")[0]
    #         n_knockout = action_space_size // 2 if self.training_env.get_attr("allow_gene_activation")[
    #             0] else action_space_size
    #
    #         for i in range(self.training_env.num_envs):
    #             info = self.locals["infos"][i]
    #
    #             # Check for finished episode info
    #             if "episode" in info:
    #                 self.total_episodes += 1
    #                 self.episode_lengths_period.append(info["episode"]["l"])
    #                 if info.get("goal_reached", False):
    #                     self.total_successes += 1
    #
    #             # Check for our custom info keys
    #             if "initial_distance" in info: self.initial_distances.append(info["initial_distance"])
    #             if "progress_made" in info: self.progress_made_period.append(info["progress_made"])
    #
    #             # Accumulate reward components
    #             if "reward_components" in info:
    #                 for key, value in info["reward_components"].items():
    #                     self.cumulative_rewards_period[key] += value
    #
    #             # Accumulate agent behavior stats
    #             self.action_history.append(1 if actions[i] >= n_knockout else 0)
    #             self.stagnation_history.append(1 if info.get("stagnated", False) else 0)
    #
    #         # --- Log aggregated data at the specified frequency ---
    #         if (self.num_timesteps - self._last_logged_step) >= self.log_freq:
    #             nv, log_dict = self.logger.name_to_value, {}
    #             current_success_rate = nv.get("periodic/success_rate", 0.0)
    #
    #             # 2. Add the current point to our AULC history
    #             self.aulc_history.append((self.num_timesteps, current_success_rate))
    #             if len(self.aulc_history) > 1:
    #                 points = np.array(self.aulc_history)
    #                 steps, perfs = points[:, 0], points[:, 1]
    #
    #                 # Add a point at (0, 0) for fair normalization
    #                 if steps[0] != 0:
    #                     steps = np.insert(steps, 0, 0)
    #                     perfs = np.insert(perfs, 0, 0.0)
    #
    #                 if len(steps) < 2:
    #                     area = 0.0
    #                 else:
    #                     area = np.trapz(y=steps, x=perfs)
    #                 duration = steps[-1]
    #
    #                 # The normalized AULC is the average performance so far
    #                 normalized_aulc = area / duration if duration > 0 else 0.0
    #                 log_dict["progress_tracker/aulc_success_rate"] = normalized_aulc
    #             # === Main Rollout & Success Metrics (Top Level) ===
    #             if 'rollout/ep_rew_mean' in nv: log_dict["rollout/ep_reward_mean"] = nv['rollout/ep_rew_mean']
    #             if 'rollout/ep_len_mean' in nv: log_dict["rollout/ep_length_mean"] = nv['rollout/ep_len_mean']
    #             if "periodic/success_rate" in nv: log_dict["rollout/success_rate"] = nv["periodic/success_rate"]
    #             if self.total_episodes > 0: log_dict[
    #                 "episode_overall/success_rate"] = self.total_successes / self.total_episodes
    #
    #             # === PPO Training Internals (train/) ===
    #             if 'train/learning_rate' in nv: log_dict["train/learning_rate"] = nv['train/learning_rate']
    #             if 'train/value_loss' in nv: log_dict["train/value_loss"] = nv['train/value_loss']
    #             if 'train/explained_variance' in nv: log_dict["train/explained_variance"] = nv[
    #                 'train/explained_variance']
    #
    #             # === Progress Tracker Panel ===
    #
    #             # --- Reward Composition (FIXED AND INCLUDED) ---
    #             total_reward_magnitude = sum(abs(v) for v in self.cumulative_rewards_period.values()) + 1e-9
    #             for key, value in self.cumulative_rewards_period.items():
    #                 # Log the percentage contribution of each component's MAGNITUDE to the total MAGNITUDE of rewards
    #                 percentage = (abs(value) / total_reward_magnitude) * 100
    #                 log_dict[f"progress_tracker/reward_composition/{key}_perc"] = percentage
    #             log_dict["progress_tracker/reward_period_total"] = sum(self.cumulative_rewards_period.values())
    #
    #             # --- Agent Behavior & Path Metrics ---
    #             if self.action_history: log_dict["progress_tracker/activation_fraction"] = np.mean(self.action_history)
    #             if self.stagnation_history: log_dict["progress_tracker/stagnation_fraction"] = np.mean(
    #                 self.stagnation_history)
    #             if self.progress_made_period: log_dict["progress_tracker/avg_progress_per_step"] = np.mean(
    #                 self.progress_made_period)
    #             if self.initial_distances and self.episode_lengths_period:
    #                 avg_dist = np.mean(self.initial_distances)
    #                 avg_len = np.mean(self.episode_lengths_period)
    #                 log_dict["progress_tracker/path_efficiency_ratio"] = avg_dist / avg_len if avg_len > 0 else 0
    #
    #             # --- Curriculum & Diversity ---
    #             log_dict["progress_tracker/curriculum_phase"] = self.training_env.get_attr('current_phase')[0]
    #             log_dict["progress_tracker/n_available_start_types"] = \
    #                 self.training_env.get_attr('n_available_start_types')[0]
    #             log_dict["progress_tracker/n_unique_target_types_in_batch"] = len(
    #                 np.unique(self.training_env.get_attr('target_cell_type')))
    #
    #             wandb.log(log_dict, step=self.num_timesteps)
    #
    #             self._last_logged_step = self.num_timesteps
    #             # Clear all periodic deques and dictionaries
    #             for deq in [self.action_history, self.stagnation_history, self.initial_distances,
    #                         self.episode_lengths_period, self.progress_made_period]:
    #                 deq.clear()
    #             self.cumulative_rewards_period.clear()
    #
    #         return True