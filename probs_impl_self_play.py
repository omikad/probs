import numpy as np
import torch
import time
import multiprocessing

import helpers
import probs_impl_common


SELF_LEARNING_MODEL: helpers.BaseSelfLearningModel = None
PARAMETERS: probs_impl_common.Parameters = None


def multiprocessing_entry_self_play(game_i):
    torch.set_num_threads(1)  # Important for multiprocessing

    SELF_LEARNING_MODEL.eval()

    seed = int(time.time()) + game_i
    np.random.seed(seed)
    torch.manual_seed(seed + 5)

    replay_episode = play_using_self_learned_model()

    return replay_episode


def play_using_self_learned_model():
    env = PARAMETERS.create_env_func()
    replay_episode = helpers.ExperienceReplayEpisode()

    dirichlet_alpha = PARAMETERS.dirichlet_alpha
    exploration_fraction = PARAMETERS.exploration_fraction

    for stepi in range(PARAMETERS.n_max_episode_steps):
        action_mask = env.get_valid_actions_mask()

        action_values = probs_impl_common.get_q_a_single_state(SELF_LEARNING_MODEL, env)   # range (-1, 1)

        greedy_probs = np.exp(action_values - np.max(action_values))
        greedy_probs[action_mask == 0] = 0
        greedy_probs /= np.sum(greedy_probs)

        noise = np.random.dirichlet(action_mask * dirichlet_alpha + 1e-6, size=None)

        probs = greedy_probs * (1 - exploration_fraction) + noise * exploration_fraction
        probs[action_mask == 0] = 0
        probs = np.maximum(0, probs)  # Fix numerical issues
        probs /= np.sum(probs)

        action = env.sample_from_action_probs(probs)

        is_white = env.is_white_to_move()

        reward_mul = 1 if is_white else -1
        reward, done = env.step(action)
        replay_episode.on_action(action, reward * reward_mul, done)
        if done:
            break

    return replay_episode


def go_self_play(self_learning_model: helpers.BaseSelfLearningModel, params: probs_impl_common.Parameters, experience_replay: helpers.ExperienceReplay):
    global SELF_LEARNING_MODEL
    global PARAMETERS
    SELF_LEARNING_MODEL = self_learning_model.to('cpu')
    PARAMETERS = params

    self_learning_model.eval()

    with torch.no_grad():

        if params.self_play_threads == 1:
            for game_i in range(params.n_games_per_iteration):
                replay_episode = play_using_self_learned_model()
                experience_replay.append_replay_episode(replay_episode)

        else:
            with multiprocessing.get_context("fork").Pool(params.self_play_threads) as multiprocessing_pool:
                for replay_episode in multiprocessing_pool.imap_unordered(multiprocessing_entry_self_play, range(params.n_games_per_iteration)):
                    experience_replay.append_replay_episode(replay_episode)