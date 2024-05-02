import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable


import helpers


@dataclass
class Parameters:
    create_env_func: Callable

    self_play_threads: int

    value_lr: float
    value_batch_size: int

    self_learning_lr: float
    self_learning_batch_size: int

    dirichlet_alpha: float
    exploration_fraction: float

    mem_max_episodes: int
    n_high_level_iterations: int
    n_games_per_iteration: int
    n_max_episode_steps: int

    num_q_s_a_calls: int
    max_depth: int

    evaluate_agent: helpers.BaseAgent
    evaluate_n_games: int

    checkpoints_dir: str


def get_q_a_single_state(self_learning_model: helpers.BaseSelfLearningModel, env: helpers.BaseEnv):
    inputs = env.get_rotated_encoded_state()
    input_tensors = [ torch.as_tensor(np.expand_dims(inp, axis=0)) for inp in inputs ]

    action_values = self_learning_model.forward(*input_tensors)
    action_values = action_values.detach().numpy()[0, :]

    return action_values


class SelfLearningAgent(helpers.BaseAgent):
    def __init__(self, name, model_keeper: helpers.ModelKeeper):
        self.self_learning_model = model_keeper.models['self_learner']
        self.name = name

    def get_action(self, env: helpers.BaseEnv):
        action_values = get_q_a_single_state(self.self_learning_model, env)

        action_mask = env.get_valid_actions_mask()
        assert action_values.shape == action_mask.shape

        action_values[action_mask == 0] = -np.inf

        action = np.argmax(action_values)
        if len(action_mask.shape) > 1:
            action = np.unravel_index(action, action_mask.shape)

        return action

    @torch.no_grad()
    def report_model_performance(self, env: helpers.BaseEnv, enemy_agent: helpers.BaseAgent, n_evaluate_games: int, n_max_steps: int, save_to_tf: bool):
        self.self_learning_model.eval()

        results = helpers.battle(env, self, enemy_agent, n_games=n_evaluate_games, n_max_steps=n_max_steps, randomize_first_turn=False)

        wins = results[0] + results[1]
        losses = results[2] + results[3]
        draws = results[4]

        print(f"  Trained agent total wins {wins}, losses {losses}, draws {draws}. Detailed result: {results} (without randomized first turn)")

        if save_to_tf:
            helpers.TENSORBOARD.append_scalar("wins", (wins + 0.5 * draws)/n_evaluate_games)

    def get_name(self):
        return self.name