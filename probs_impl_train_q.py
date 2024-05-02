import numpy as np
import torch
import multiprocessing
import heapq
from dataclasses import dataclass
from typing import Optional

import helpers
import probs_impl_common


VALUE_MODEL: helpers.BaseValueModel = None
SELF_LEARNING_MODEL: helpers.BaseSelfLearningModel = None
PARAMETERS: probs_impl_common.Parameters = None
N_ACTIONS: int = None


def multiprocessing_entry_get_self_learning_dataset_row(episode: helpers.ExperienceReplayEpisode):
    torch.set_num_threads(1)  # Important for multiprocessing

    VALUE_MODEL.eval()

    episode_dataset = get_self_learning_model_dataset_rows(episode)
    return episode_dataset


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
                    env=env,
                    action_and_kids=[],
                    action_values=None,
                    terminal=False)
        beam = [ (MIN_INF, 0, tree, 0) ]   # heap of options: [(-q_s_a, counter, node, node_depth)]
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
        expanded_cnt = 0
        queue = [(MIN_INF, tree, 0)]   # array of (priority, node, node_depth)
        qi = 0
        while qi < len(queue):
            priority, node, depth = queue[qi]

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

    return tree, beam, expanded_cnt


def expand_env_to_tree_data__using_q_s_a(env: helpers.BaseEnv, prev_tree: TreeNode, prev_action: int):
    """
    Params:
        * env: current environment
        * prev_tree: previous env beam tree result
        * prev_action: action which led to the current env
    Returns tree
    """

    num_q_s_a_calls = PARAMETERS.num_q_s_a_calls
    max_depth = PARAMETERS.max_depth
    MIN_INF = -1e5

    tree, beam, expanded_cnt = init_beam_search(env, prev_tree, prev_action)
    heap_counter = len(beam)

    while len(beam) > 0:
        if expanded_cnt >= num_q_s_a_calls and beam[0][0] != MIN_INF:   # Always expand root to get its V(ns), otherwise Q will learn its own predictions
            break
        expanded_cnt += 1

        __neg_q_s_a, __counter, node, depth = heapq.heappop(beam)
        node_env = node.env

        action_values = get_q_a_single_state(node_env)
        node.action_values = action_values

        for action in node_env.get_valid_actions_iter():
            kid_env = node_env.copy()

            reward, done = kid_env.step(action)

            if done:
                kid = TreeNode(
                        state_value=-reward,
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

    return tree


def get_state_values(to_eval):
    dataloader = torch.utils.data.DataLoader(to_eval, batch_size=PARAMETERS.value_batch_size, shuffle=False, drop_last=False)

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
            computed_vs = -min(kid.state_value for action, kid in node.action_and_kids)
            node.state_value = computed_vs
        assert node.state_value is not None

    action_values = np.zeros(N_ACTIONS, dtype=np.float32)
    for action, kid in tree.action_and_kids:
        action_values[action] = -kid.state_value

    return action_values


def get_self_learning_model_dataset_rows(episode: helpers.ExperienceReplayEpisode):
    tree = []

    episode_dataset = []
    for action, env, reward in helpers.ExperienceReplay.yield_training_tuples_from_episode(PARAMETERS.create_env_func(), episode):
        if not env.done:
            tree = expand_env_to_tree_data__using_q_s_a(env, tree, action)

            action_values = evaluate_tree_data(tree)

            dataset_row = env.get_rotated_encoded_state()
            dataset_row.append(action_values)
            dataset_row.append(np.copy(env.get_valid_actions_mask()))

            episode_dataset.append(dataset_row)

    return episode_dataset


def get_q_a_single_state(env):
    state_enc = env.get_rotated_encoded_state()
    input = [torch.as_tensor(inp).unsqueeze(dim=0) for inp in state_enc]
    action_values = SELF_LEARNING_MODEL.forward(*input)
    return action_values.detach().numpy()[0, :]


def go_train_self_learning_model(
                experience_replay: helpers.ExperienceReplay,
                self_learning_model: helpers.BaseSelfLearningModel,
                params: probs_impl_common.Parameters,
                value_model: helpers.BaseValueModel,
                optimizer):
    global SELF_LEARNING_MODEL
    global VALUE_MODEL
    global PARAMETERS
    global N_ACTIONS
    VALUE_MODEL = value_model.to('cpu')
    SELF_LEARNING_MODEL = self_learning_model.to('cpu')
    PARAMETERS = params
    N_ACTIONS = len(params.create_env_func().get_valid_actions_mask())

    value_model.eval()
    self_learning_model.eval()

    def __yield_episode_dataset():
        if params.self_play_threads == 1:
            with torch.no_grad():
                for episode in experience_replay.episodes:
                    episode_dataset = get_self_learning_model_dataset_rows(episode)
                    yield episode_dataset
        else:
            with multiprocessing.get_context("fork").Pool(params.self_play_threads) as multiprocessing_pool:
                for episode_dataset in multiprocessing_pool.imap_unordered(multiprocessing_entry_get_self_learning_dataset_row, experience_replay.episodes):
                    yield episode_dataset

    dataset = []
    for episode_dataset in __yield_episode_dataset():
        dataset.extend(episode_dataset)

    helpers.TENSORBOARD.append_scalar('self_learner_data_len', len(dataset))

    self_learning_model.train()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.self_learning_batch_size, shuffle=True, drop_last=True)

    for batch_input in dataloader:

        inputs = batch_input[:-2]
        actual_action_values = batch_input[-2]   # (B, N_ACTIONS)
        actions_mask = batch_input[-1]         # (B, N_ACTIONS)

        pred_action_values = self_learning_model.forward(*inputs)

        loss = torch.nn.functional.mse_loss(pred_action_values, actual_action_values, reduction='none')

        loss = (loss * actions_mask).sum(axis=1)

        action_mask_norm = actions_mask.sum(axis=1)

        loss = (loss / action_mask_norm).mean()

        helpers.TENSORBOARD.append_scalar('self_learning_loss', loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
