import numpy as np
import torch
import time
import multiprocessing
from collections import Counter, deque

import helpers
import probs_impl_common


VALUE_MODEL: helpers.BaseValueModel = None
SELF_LEARNING_MODEL: helpers.BaseSelfLearningModel = None
PARAMETERS: probs_impl_common.Parameters = None
GET_DATASET_DEVICE: str = None


def multiprocessing_entry_self_play(game_ids):
    # start = time.time()
    torch.set_num_threads(1)  # Important for multiprocessing

    VALUE_MODEL.eval()
    SELF_LEARNING_MODEL.eval()

    seed = int(time.time()) + game_ids[0]
    np.random.seed(seed)
    torch.manual_seed(seed + 5)

    stats = Counter()
    replay_episodes = []

    replay_episodes, stats = play_using_self_learned_model(game_ids)

    # p = multiprocessing.current_process()
    # print(f"Fork {p._identity, os.getpid()}. {game_i} completed. {time.time() - start} seconds")

    return replay_episodes, stats


def self_play__iterative(env: helpers.BaseEnv, replay_episode: helpers.ExperienceReplayEpisode, episode_stats: Counter):
    for stepi in range(PARAMETERS.n_max_episode_steps):
        action_mask = env.get_valid_actions_mask()

        to_eval_env = [env.get_rotated_encoded_state()]
        yield to_eval_env
        action_values = to_eval_env[1]

        action, greedy_action = probs_impl_common.sample_action(env, PARAMETERS, action_values, action_mask, is_v_not_q=True)

        if action == greedy_action:
            episode_stats['greedy_action_sum'] += 1
        episode_stats['greedy_action_cnt'] += 1

        reward_mul = 1 if env.is_white_to_move() else -1
        reward, done = env.step(action)
        replay_episode.on_action(action, reward * reward_mul, done)
        if done:
            break

    yield None


@torch.no_grad()
def play_using_self_learned_model(game_ids):

    tasks_deque = deque()  # [(iterator, to_eval_env)]
    # iterator is self_play__iterative
    # to_eval_env = [env] - need to add evaluated actions as second element

    replay_episodes = []
    stats = Counter()

    next_game_i = 0
    while next_game_i < len(game_ids) or len(tasks_deque) > 0:

        if next_game_i < len(game_ids) and len(tasks_deque) < PARAMETERS.self_learning_batch_size // 4:
            env = PARAMETERS.create_env_func()
            replay_episode = helpers.ExperienceReplayEpisode()
            replay_episodes.append(replay_episode)
            next_game_i += 1

            it = iter(self_play__iterative(env, replay_episode, stats))
            to_eval_q_a = next(it)

            if to_eval_q_a is not None:
                tasks_deque.append((it, to_eval_q_a))

        else:
            inputs_collection = [ to_eval_env[0] for it, to_eval_env in tasks_deque ]
            action_values_batch = probs_impl_common.get_q_a_multi_inputs(VALUE_MODEL, SELF_LEARNING_MODEL, inputs_collection, GET_DATASET_DEVICE)

            new_tasks_deque = deque()

            for (it, to_eval_q_a), action_values in zip(tasks_deque, action_values_batch):
                to_eval_q_a.append(action_values)

                to_eval_q_a = next(it)
                if to_eval_q_a is not None:
                    new_tasks_deque.append((it, to_eval_q_a))

            tasks_deque = new_tasks_deque

    return replay_episodes, stats


def go_self_play(value_model: helpers.BaseValueModel, self_learning_model: helpers.BaseSelfLearningModel, params: probs_impl_common.Parameters, experience_replay: helpers.ExperienceReplay, get_dataset_device: str):
    global VALUE_MODEL
    global SELF_LEARNING_MODEL
    global PARAMETERS
    global GET_DATASET_DEVICE

    GET_DATASET_DEVICE = get_dataset_device
    VALUE_MODEL = value_model
    SELF_LEARNING_MODEL = self_learning_model

    value_model.eval()
    self_learning_model.eval()
    PARAMETERS = params

    stats = Counter()

    with torch.no_grad():

        # ------------ No multiprocessing
        if params.self_play_threads == 1:
            game_ids = list(range(params.v_train_episodes))

            replay_episodes, episodes_stats = play_using_self_learned_model(game_ids)
            for replay_episode in replay_episodes:
                experience_replay.append_replay_episode(replay_episode)
            stats += episodes_stats

        # ------------ Multiprocessing
        else:
            # self_learning_model.share_memory()  - ?

            game_ids = [[] for _ in range(params.self_play_threads)]
            gi = 0
            for game_i in range(params.v_train_episodes):
                game_ids[gi].append(game_i)
                gi = (gi + 1) % len(game_ids)

            with multiprocessing.get_context("fork").Pool(params.self_play_threads) as multiprocessing_pool:
                for replay_episodes, episodes_stats in multiprocessing_pool.imap_unordered(multiprocessing_entry_self_play, game_ids):
                    for replay_episode in replay_episodes:
                        experience_replay.append_replay_episode(replay_episode)
                    stats += episodes_stats

    helpers.TENSORBOARD.append_scalar('greedy_action_freq', stats['greedy_action_sum'] / stats['greedy_action_cnt'])