import numpy as np
import torch
import heapq
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from typing import Optional

import helpers
import probs_impl_common


VALUE_MODEL: helpers.BaseValueModel = None
SELF_LEARNING_MODEL: helpers.BaseSelfLearningModel = None
GET_DATASET_DEVICE: str = None
PARAMETERS: probs_impl_common.Parameters = None
N_ACTIONS: int = None


def multiprocessing_entry_get_self_learning_dataset_row__using_batch_eval(episodes: list[helpers.ExperienceReplayEpisode]):
    global VALUE_MODEL
    global SELF_LEARNING_MODEL

    torch.set_num_threads(1)

    VALUE_MODEL.eval()
    SELF_LEARNING_MODEL.eval()

    dataset = []
    total_stats = Counter()
    for dataset_batch_rows, stats in get_dataset_batches_using_batched_eval(episodes):
        dataset.extend(dataset_batch_rows)
        for key, cnt in stats.items():
            total_stats[key] += cnt

    return dataset, total_stats


@dataclass
class TreeNode:
    state_value: Optional[float]
    env: helpers.BaseEnv
    action_and_kids: list[tuple[int, "TreeNode"]]
    action_values: Optional[np.ndarray]
    terminal: bool

    def bfs(self):
        queue = [self]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            for action, kid in node.action_and_kids:
                queue.append(kid)
            qi += 1
        return queue


def init_beam_search(env: helpers.BaseEnv, prev_tree: TreeNode, prev_action: int):
    MIN_INF = -1e5

    if prev_action < 0:
        tree = TreeNode(
                    state_value=None,
                    env=env.copy(),
                    action_and_kids=[],
                    action_values=None,
                    terminal=False)
        beam = [ (MIN_INF, 0, tree, 0) ]   # heap of options: [(-q_s_a, counter, node, node_depth)]
        stat_depth_sum = 0
        stat_depth_cnt = 1
        expanded_cnt = 0

    else:
        tree = None
        for action, kid in prev_tree.action_and_kids:
            if action == prev_action:
                tree = kid
                break
        assert tree is not None

        # Re-compute beam
        beam = []               # heap of options: [(-q_s_a, counter, node, node_depth)]
        stat_depth_sum = 0
        stat_depth_cnt = 0
        expanded_cnt = 0
        queue = [(MIN_INF, tree, 0)]   # array of (priority, node, node_depth)
        qi = 0
        while qi < len(queue):
            priority, node, depth = queue[qi]

            stat_depth_sum += depth
            stat_depth_cnt += 1

            if node.terminal:
                pass

            elif len(node.action_and_kids) == 0:
                heapq.heappush(beam, (priority, len(beam), node, depth))

            else:
                expanded_cnt += 1

                for action, kid in node.action_and_kids:
                    kid_priority = MIN_INF if depth == 0 else -node.action_values[action]
                    queue.append((kid_priority, kid, depth + 1))

            qi += 1

    return tree, beam, stat_depth_sum, stat_depth_cnt, expanded_cnt


def expand_env_to_tree_data__using_q_s_a(env: helpers.BaseEnv, prev_tree: TreeNode, prev_action: int, out_tree_container: list, out_stats: Counter):
    """
    Params:
        * env: current environment
        * prev_tree: previous env beam tree result
        * prev_action: action which led to the current env
    Yields items to evaluate
    Returns tree, stats
        * out_tree_container[0] array of tree nodes
        * out_stats: dict of stats
    """

    num_q_s_a_calls = PARAMETERS.num_q_s_a_calls
    max_depth = PARAMETERS.max_depth
    MIN_INF = -1e5

    tree, beam, stat_depth_sum, stat_depth_cnt, expanded_cnt = init_beam_search(env, prev_tree, prev_action)
    heap_counter = len(beam)
    stat_reused_tree_size_sum = stat_depth_cnt
    # print("expanded_cnt, len(tree), len(beam)", expanded_cnt, len(tree), len(beam))

    while len(beam) > 0:
        if expanded_cnt >= num_q_s_a_calls and beam[0][0] != MIN_INF:   # Always expand root to get its V(ns), otherwise Q will learn its own predictions
            break
        expanded_cnt += 1

        __neg_q_s_a, __counter, node, depth = heapq.heappop(beam)
        node_env = node.env

        to_eval_q_a = [node_env.get_rotated_encoded_state()]
        yield to_eval_q_a
        action_values = to_eval_q_a[1]

        node.action_values = action_values

        for action in node_env.get_valid_actions_iter():
            kid_env = node_env.copy()

            is_white_before = kid_env.is_white_to_move()

            reward, done = kid_env.step(action)

            if done:
                reward_mul = -1 if kid_env.is_white_to_move() != is_white_before else 1

                kid = TreeNode(
                        state_value=reward_mul * reward,
                        env=kid_env,
                        action_and_kids=[],
                        action_values=None,
                        terminal=True)
                node.action_and_kids.append((action, kid))

            else:
                kid = TreeNode(
                        state_value=None,
                        env=kid_env,
                        action_and_kids=[],
                        action_values=None,
                        terminal=False)
                node.action_and_kids.append((action, kid))

                if depth < max_depth:
                    kid_priority = MIN_INF if depth == 0 else -action_values[action]
                    heapq.heappush(beam, (kid_priority, heap_counter, kid, depth + 1))
                    heap_counter += 1

                    stat_depth_sum += depth + 1
                    stat_depth_cnt += 1

    out_tree_container.append(tree)
    out_stats['depth_sum'] += stat_depth_sum
    out_stats['depth_cnt'] += stat_depth_cnt
    out_stats['reused_tree_size_sum'] += stat_reused_tree_size_sum
    out_stats['trees_cnt'] += 1
    out_stats['tree_size_sum'] += len(tree.bfs())


def get_state_values(to_eval):
    # num_workers=1, pin_memory=True
    dataloader = helpers.torch_create_dataloader(to_eval, GET_DATASET_DEVICE, batch_size=PARAMETERS.value_batch_size, shuffle=False, drop_last=False)

    state_values = []
    for batch_input in dataloader:
        eval_result = VALUE_MODEL.forward(*batch_input)
        eval_result = eval_result.detach().cpu().numpy()[:, 0]
        state_values.extend(eval_result)

    return state_values


def evaluate_tree_data(tree: TreeNode):
    to_eval_data = []
    to_eval_nodes = []

    tree_bfs = tree.bfs()

    for node in tree_bfs:
        if node.state_value is None and len(node.action_and_kids) == 0:
            to_eval_data.append(node.env.get_rotated_encoded_state())
            to_eval_nodes.append(node)

    state_values = get_state_values(to_eval_data)
    assert len(state_values) == len(to_eval_nodes)
    for node, computed_vs in zip(to_eval_nodes, state_values):
        node.state_value = computed_vs

    for node in tree_bfs[::-1]:
        if len(node.action_and_kids) > 0:
            node_white_to_move = node.env.is_white_to_move()
            computed_vs = max((-1 if (node_white_to_move != kid.env.is_white_to_move()) else 1) * kid.state_value for action, kid in node.action_and_kids)
            node.state_value = computed_vs
        assert node.state_value is not None

    action_values = np.zeros(N_ACTIONS, dtype=np.float32)
    for action, kid in tree.action_and_kids:
        rew_mul = -1 if (tree.env.is_white_to_move() != kid.env.is_white_to_move()) else 1
        action_values[action] = rew_mul * kid.state_value

    return action_values


def get_self_learning_model_dataset_rows__using_batch_eval(out_episode_rows: list, out_stats: Counter):
    tree = None

    env = PARAMETERS.create_env_func()
    action = -1
    for stepi in range(PARAMETERS.n_max_episode_steps):

        action_mask = env.get_valid_actions_mask()

        tree_container = []

        for to_eval_q_a in expand_env_to_tree_data__using_q_s_a(env, tree, action, tree_container, out_stats):
            yield to_eval_q_a

        tree = tree_container[0]

        action_values = evaluate_tree_data(tree)

        action, greedy_action = probs_impl_common.sample_action(env, PARAMETERS, action_values, action_mask, is_v_not_q=False)

        for dataset_row in env.get_rotated_encoded_states_with_symmetry__q_value_model(action_values):
            if PARAMETERS.dataset_drop_ratio > 1e-5 and np.random.rand() < PARAMETERS.dataset_drop_ratio:
                continue
            out_episode_rows.append(dataset_row)

        if action == greedy_action:
            out_stats['greedy_action_sum'] += 1
        out_stats['greedy_action_cnt'] += 1

        reward_mul = 1 if env.is_white_to_move() else -1
        reward, done = env.step(action)
        # replay_episode.on_action(action, reward * reward_mul, done)
        if done:
            break

    yield None


@torch.no_grad()
def get_dataset_batches_using_batched_eval(n_games: int):
    tasks_deque = deque()  # [(iterator, to_eval_qa)]
    # iterator is get_self_learning_model_dataset_rows__using_batch_eval
    # to_eval_qa = [env.get_rotated_encoded_state()] - need to add evaluated actions as second element

    result_rows = []
    stats = Counter()

    next_episode_i = 0
    while next_episode_i < n_games or len(tasks_deque) > 0:

        if next_episode_i < n_games and len(tasks_deque) < PARAMETERS.get_q_dataset_batch_size:
            next_episode_i += 1

            it = iter(get_self_learning_model_dataset_rows__using_batch_eval(result_rows, stats))
            to_eval_q_a = next(it)

            if to_eval_q_a is not None:
                tasks_deque.append((it, to_eval_q_a))

        else:
            inputs_collection = [ to_eval_qa[0] for it, to_eval_qa in tasks_deque ]
            action_values_batch = probs_impl_common.get_q_a_multi_inputs(VALUE_MODEL, SELF_LEARNING_MODEL, inputs_collection, GET_DATASET_DEVICE)

            new_tasks_deque = deque()

            for (it, to_eval_q_a), action_values in zip(tasks_deque, action_values_batch):
                to_eval_q_a.append(action_values)

                to_eval_q_a = next(it)
                if to_eval_q_a is not None:
                    new_tasks_deque.append((it, to_eval_q_a))

            tasks_deque = new_tasks_deque

            yield result_rows, stats
            result_rows.clear()
            stats.clear()


# @torch.inference_mode()
def get_dataset(
        n_games: int,
        value_model: helpers.BaseValueModel,
        self_learning_model: helpers.BaseSelfLearningModel,
        params: probs_impl_common.Parameters,
        device):
    global SELF_LEARNING_MODEL
    global VALUE_MODEL
    global GET_DATASET_DEVICE
    global PARAMETERS
    global N_ACTIONS

    GET_DATASET_DEVICE = device

    VALUE_MODEL = value_model
    SELF_LEARNING_MODEL = self_learning_model

    PARAMETERS = params
    N_ACTIONS = len(params.create_env_func().get_valid_actions_mask())

    value_model.eval()
    self_learning_model.eval()

    def __yield_dataset_batch_rows():
        # ------------ No multiprocessing, eval by batches
        if params.self_play_threads == 1:
            for dataset_batch_rows, stats in get_dataset_batches_using_batched_eval(n_games):
                yield dataset_batch_rows, stats

        # ------------ Multiprocessing, eval by batches
        elif params.self_play_threads > 1:
            raise 1  # not implemented
            # episode_batches = [[] for _ in range(params.self_play_threads)]
            # idx = 0
            # for episode in experience_replay.episodes:
            #     episode_batches[idx].append(episode)
            #     idx = (idx + 1) % len(episode_batches)

            # with multiprocessing.get_context("fork").Pool(params.self_play_threads) as multiprocessing_pool:
            #     for sub_dataset, stats in multiprocessing_pool.imap_unordered(multiprocessing_entry_get_self_learning_dataset_row__using_batch_eval, episode_batches):
            #         yield sub_dataset, stats

    stats = Counter()
    dataset = []
    for dataset_batch_rows, curr_stats in __yield_dataset_batch_rows():
        dataset.extend(dataset_batch_rows)
        stats += curr_stats

    return dataset, stats


def train_q_model(
        dataset,
        value_model: helpers.BaseValueModel,
        self_learning_model: helpers.BaseSelfLearningModel,
        params: probs_impl_common.Parameters,
        device,
        optimizer):
    value_model.eval()
    self_learning_model.train()

    dataloader = helpers.torch_create_dataloader(dataset, device, params.self_learning_batch_size, shuffle=True, drop_last=True)

    for batch_input in dataloader:
        # for inp_tensor in batch_input: print(f"[train_self_learning_model] inp_tensor {inp_tensor.shape} {inp_tensor.dtype}")

        inputs = batch_input[:-2]
        actual_action_values = batch_input[-2]   # (B, N_ACTIONS)
        actions_mask = batch_input[-1]         # (B, N_ACTIONS)

        pred_action_values = self_learning_model.forward(*inputs)

        loss = torch.nn.functional.mse_loss(pred_action_values, actual_action_values, reduction='none')

        loss = (loss * actions_mask).sum(axis=1)

        action_mask_norm = actions_mask.sum(axis=1)

        if action_mask_norm.min() == 0:
            print(action_mask_norm.detach().cpu().numpy())
            raise 1

        loss = (loss / action_mask_norm).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        helpers.TENSORBOARD.append_scalar('self_learning_loss', float(loss))
