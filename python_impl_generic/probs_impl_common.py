import heapq
import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable


import helpers
import environments


@dataclass
class Parameters:
    env_name: str
    create_env_func: Callable

    sub_processes_cnt: int
    self_play_threads: int

    value_lr: float
    value_batch_size: int

    self_learning_lr: float
    self_learning_batch_size: int
    get_q_dataset_batch_size: int

    alphazero_move_num_sampling_moves: float
    v_pick_second_best_prob: float
    greedy_action_freq: float
    dirichlet_alpha: float
    exploration_fraction: float

    mem_max_episodes: int
    n_high_level_iterations: int
    n_max_episode_steps: int
    q_dataset_episodes_sub_iter: int
    v_train_episodes: int
    q_train_episodes: int
    dataset_drop_ratio: float

    num_q_s_a_calls: int
    max_depth: int

    evaluate_agent: helpers.BaseAgent
    evaluate_n_games: int

    checkpoints_dir: str


def sample_action(env: helpers.BaseEnv, params: Parameters, action_values, action_mask, is_v_not_q: bool):
    assert action_values.shape == action_mask.shape

    alphazero_move_num_sampling_moves = params.alphazero_move_num_sampling_moves
    greedy_action_freq = params.greedy_action_freq
    dirichlet_alpha = params.dirichlet_alpha
    exploration_fraction = params.exploration_fraction

    if alphazero_move_num_sampling_moves is not None:

        I = np.where(action_mask == 1)[0]
        probs = action_values[I]

        greedy_i = np.argmax(probs)
        greedy_action = I[greedy_i]

        if env.move_counter < alphazero_move_num_sampling_moves:
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)
            action = np.random.choice(I, p=probs)
        elif is_v_not_q and params.v_pick_second_best_prob > 0 and np.random.rand() <= params.v_pick_second_best_prob:
            probs[greedy_i] = 1e-5
            action = I[np.argmax(probs)]
        else:
            action = greedy_action

    elif greedy_action_freq is not None and np.random.rand() <= greedy_action_freq:
        action_values[action_mask == 0] = -1e9
        greedy_action = np.argmax(action_values)
        action = greedy_action

    else:
        greedy_probs = np.exp(action_values - np.max(action_values))
        greedy_probs[action_mask == 0] = 0
        greedy_probs /= np.sum(greedy_probs)

        greedy_action = np.argmax(greedy_probs)

        noise = np.random.dirichlet(action_mask * dirichlet_alpha + 1e-6, size=None)

        probs = greedy_probs * (1 - exploration_fraction) + noise * exploration_fraction
        probs[action_mask == 0] = 0
        probs = np.maximum(0, probs)  # Fix numerical issues
        probs /= np.sum(probs)

        action = env.sample_from_action_probs(probs)

    return action, greedy_action


def get_q_a_single_state(value_model: helpers.BaseValueModel, self_learning_model: helpers.BaseSelfLearningModel, env: helpers.BaseEnv, device: str):
    inputs = env.get_rotated_encoded_state()
    input_tensors = [ torch.as_tensor(np.expand_dims(inp, axis=0)).to(device) for inp in inputs ]

    action_values = self_learning_model.forward(*input_tensors)
    action_values = action_values.detach().cpu().numpy()[0, :]

    return action_values


def get_q_a_single_inputs(value_model: helpers.BaseValueModel, self_learning_model: helpers.BaseSelfLearningModel, inputs):
    input_tensors = [ torch.as_tensor(np.expand_dims(inp, axis=0)) for inp in inputs ]

    action_values = self_learning_model.forward(*input_tensors)
    action_values = action_values.detach().cpu().numpy()[0, :]

    return action_values


def get_q_a_multi_inputs(value_model: helpers.BaseValueModel, self_learning_model: helpers.BaseSelfLearningModel, inputs_collection, device: str):
    # inputs_collection - list where each row is result of env.get_rotated_encoded_state()
    input_tensors = []
    for i in range(len(inputs_collection[0])):
        inps = [ torch.Tensor(inp[i]) for inp in inputs_collection ]
        inps = torch.stack(inps).to(device)
        input_tensors.append(inps)

    action_values = self_learning_model.forward(*input_tensors)
    action_values = action_values.detach().cpu().numpy()

    return action_values


class SelfLearningAgent(helpers.BaseAgent):
    def __init__(self, name, model_keeper: helpers.ModelKeeper, device: str):
        self.value_model = model_keeper.models['value']
        self.self_learning_model = model_keeper.models['self_learner']
        self.name = name
        self.device = device

    def get_action(self, env: helpers.BaseEnv):
        action_values = get_q_a_single_state(self.value_model, self.self_learning_model, env, self.device)

        action_mask = env.get_valid_actions_mask()
        assert action_values.shape == action_mask.shape

        action_values[action_mask == 0] = -np.inf

        action = np.argmax(action_values)
        if len(action_mask.shape) > 1:
            action = np.unravel_index(action, action_mask.shape)

        return action

    def eval(self):
        self.value_model.eval()
        self.self_learning_model.eval()

    def get_name(self):
        return self.name


def parse_model_str(model_str):
    res = dict()
    for part in model_str.split(','):
        key, val = part.split('=')
        res[key] = val
    return res


def create_model_keeper(args, env, train_params, model_str) -> helpers.ModelKeeper:
    parsed_model_str = parse_model_str(model_str)

    model_keeper = helpers.ModelKeeper()

    model_keeper.models['value'] = environments.create_value_model(env, parsed_model_str['V'])
    model_keeper.models['self_learner'] = environments.create_self_learning_model(env, parsed_model_str['SL'])

    if train_params is not None:
        model_keeper.optimizers['value'] = torch.optim.AdamW( model_keeper.models['value'].parameters(), lr=train_params.value_lr, weight_decay=1e-5)
        model_keeper.optimizers['self_learner'] = torch.optim.AdamW(model_keeper.models['self_learner'].parameters(), lr=train_params.self_learning_lr, weight_decay=1e-5)

    if 'CKPT' in parsed_model_str:
        checkpoint = parsed_model_str['CKPT']
        model_keeper.load_from_checkpoint(checkpoint)
        print(f"Models loaded from `{checkpoint}`:")

    for name, model in model_keeper.models.items():
        total_parameters = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {model.__class__.__name__} Params cnt: {total_parameters}")

    return model_keeper


def create_agent(args, env, train_params, model_str, device: str) -> helpers.BaseAgent:
    if model_str == 'random':
        return helpers.RandomAgent()
    if model_str == 'one_step_lookahead':
        return helpers.OneStepLookaheadAgent()
    if model_str == 'two_step_lookahead':
        return helpers.TwoStepLookaheadAgent()
    if model_str == 'three_step_lookahead':
        return helpers.ThreeStepLookaheadAgent()
    elif 'budget_lookahead' in model_str:
        parsed_model_str = parse_model_str(model_str)
        return helpers.BudgetLookahead(time_budget=float(parsed_model_str['time_budget']))
    else:
        parsed_model_str = parse_model_str(model_str)
        model_keeper = create_model_keeper(args, env, train_params, model_str=model_str)
        model_keeper.eval()

        if 'agent_type' not in parsed_model_str or parsed_model_str['agent_type'] == 'qval':
            return SelfLearningAgent(model_keeper.models['self_learner'].__class__.__name__, model_keeper=model_keeper, device=device)

        if parsed_model_str['agent_type'] == 'qvalts':
            return SelfLearningAgent_TreeScan(model_str, model_keeper=model_keeper, device=device)

    raise Exception(f"Unknown agent type `{model_str}`")


class SelfLearningAgent_TreeScan(helpers.BaseAgent):
    def __init__(self, name, model_keeper: helpers.ModelKeeper, device: str):
        self.self_learning_model = model_keeper.models['self_learner']
        self.name = name
        self.device = device
        self.action_time_budget = 3.0
        self.expand_tree_budget = 20000
        self.batch_size = 10
        self.last_search_time = 0
        self.last_search_nodes_cnt = 0

    def get_action(self, env: helpers.BaseEnv):
        action_time_budget = self.action_time_budget
        expand_tree_budget = self.expand_tree_budget
        batch_size = self.batch_size
        device = self.device
        assert not env.done

        tree_kids = dict()             # node_i -> action -> kid_node_i
        tree_qvalues = { 0: None }    # node_i -> action -> q-value
        tree_colors = { 0: env.is_white_to_move() }   # node_i -> {true or false}

        start_time = time.time()
        end_time = start_time + action_time_budget
        nodes_cnt = 1
        heap = [(float('-inf'), 0, env)]
        while len(heap) > 0 and time.time() < end_time and nodes_cnt < expand_tree_budget:
            input_tensors = []
            envs = []
            nodes = []
            while len(heap) > 0 and len(input_tensors) < batch_size:
                _, node_i, curr_env = heapq.heappop(heap)
                encs = curr_env.get_rotated_encoded_state()
                if len(input_tensors) == 0:
                    input_tensors = [[] for _ in range(len(encs))]
                for i, enc in enumerate(encs):
                    input_tensors[i].append(torch.Tensor(enc))
                envs.append(curr_env)
                nodes.append(node_i)

            input_tensors = [torch.stack(inps).to(device) for inps in input_tensors]
            predicted_action_values = self.self_learning_model.forward(*input_tensors).detach().cpu().numpy()

            for row_i, curr_env, node_i in zip(range(len(envs)), envs, nodes):
                tree_qvalues[node_i] = predicted_action_values[row_i, :]
                tree_kids[node_i] = dict()

                for action in curr_env.get_valid_actions_iter():
                    copy_env = curr_env.copy()
                    reward, done = copy_env.step(action)
                    if done:
                        tree_qvalues[node_i][action] = reward
                        tree_kids[node_i][action] = nodes_cnt
                        nodes_cnt += 1
                    else:
                        tree_kids[node_i][action] = nodes_cnt
                        tree_colors[nodes_cnt] = copy_env.is_white_to_move()
                        heapq.heappush(heap, (-predicted_action_values[row_i, action], nodes_cnt, copy_env))
                        nodes_cnt += 1

                if time.time() >= end_time:
                    break

        def __get_best_action(node_i):
            node_qvals = tree_qvalues[node_i]
            color = tree_colors[node_i]

            if node_i in tree_kids:
                best_action, best_val = None, 0
                for action, kid_node_i in tree_kids[node_i].items():

                    if kid_node_i in tree_qvalues:
                        kid_action, kid_val, kid_color = __get_best_action(kid_node_i)
                        if color != kid_color:
                            kid_val = -kid_val
                    else:
                        kid_val = node_qvals[action]

                    if best_action is None or kid_val > best_val:
                        best_action, best_val = action, kid_val

                return best_action, best_val, color

            else:
                best_action = np.argmax(node_qvals)
                return best_action, node_qvals[best_action], color

        action = __get_best_action(0)[0]

        self.last_search_nodes_cnt = nodes_cnt
        self.last_search_time = time.time() - start_time

        return action


    def eval(self):
        self.self_learning_model.eval()

    def get_name(self):
        return self.name