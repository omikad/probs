import os
import numpy as np
import time
import copy
import torch
import resource
from collections import defaultdict, Counter, deque
from torch.utils.tensorboard import SummaryWriter


TENSORBOARD = None


class BaseEnv:
    def get_rotated_encoded_state(self):
        raise Exception("Need to be implemented in child, return list of arrays that will be passed to NNs forward")

    def step(self, action: int):
        raise Exception("Need to be implemented in child, return tuple (reward, done)")

    def reset(self):
        raise Exception("Need to be implemented in child")

    def copy(self):
        raise Exception("Need to be implemented in child")

    def get_valid_actions_mask(self) -> np.ndarray:
        raise Exception("Need to be implemented in child, return 1d array of 0 or 1")

    def sample_from_action_probs(self, action_probs: np.ndarray) -> int:
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def get_random_action(self) -> int:
        I = np.where(self.get_valid_actions_mask() == 1)[0]
        action = np.random.choice(I)
        return action

    def get_valid_actions_iter(self):
        return np.where(self.get_valid_actions_mask() == 1)[0]

    def render_ascii(self):
        raise Exception("Need to be implemented in child")

    def is_white_to_move(self) -> bool:
        raise Exception("Need to be implemented in child")


class BaseAgent:
    def get_action(self, env: BaseEnv) -> int:
        raise Exception("Need to be implemented in child")

    def get_name(self) -> str:
        return self.__class__.__name__


class BaseValueModel(torch.nn.Module):
    def get_v_single_state(self, env: BaseEnv) -> float:
        raise Exception("Need to be implemented in child")


class BaseSelfLearningModel(torch.nn.Module):
    pass


def get_model_memory_sizes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size, buffer_size

def show_models_memory_sizes(models):
    for model in models:
        param_size, buffer_size = get_model_memory_sizes(model)
        print(f"model {model.__class__.__name__} KB sizes: params {param_size/1024.0}, buffer {buffer_size/1024.0}")


def show_usage():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    print(f"Memory usage: usertime={usage[0]} systime={usage[1]} mem={usage[2]/1024.0} mb")


def get_mem_size():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return usage[2]


class UsageCounter:
    def __init__(self):
        self.names = []
        self.times = []
        self.mems = []
        self.checkpoint("")

    def checkpoint(self, name):
        self.names.append(name)
        self.times.append(time.time())
        self.mems.append(get_mem_size())

    def print_stats(self):
        content_tabs = [[] for _ in range(3)]
        for i in range(1, len(self.names)):
            name = self.names[i]
            timedelta = self.times[i] - self.times[i - 1]
            memdelta = self.mems[i] - self.mems[i - 1]
            content_tabs[0].append(name)
            content_tabs[1].append(f"{timedelta:>30} sec")
            content_tabs[2].append(f"{memdelta:>10} kb")

        lens = [max(len(line) for line in tab) for tab in content_tabs]
        for i in range(len(content_tabs[0])):
            printrow = []
            for j in range(3):
                printrow.append(content_tabs[j][i] + ' ' * (lens[j] - len(content_tabs[j][i])))
            print('--- ' + ' '.join(printrow))


class ModelKeeper:
    """
    Save/load models and optimizers from checkpoints
    """

    def __init__(self) -> None:
        self.models = dict()
        self.optimizers = dict()

    def save_checkpoint(self, checkpoints_dir, log_prefix):
        data = dict()

        for name, model in self.models.items():
            data[name] = { 'model': model.state_dict() }

        for name, opt in self.optimizers.items():
            data[name]['opt'] = opt.state_dict()

        checkpoint_path = f"checkpoint_{ time.strftime('%Y%m%d-%H%M%S') }.ckpt"
        if checkpoints_dir is not None:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_path)
        torch.save(data, checkpoint_path)
        print(f"{log_prefix}. Checkpoint saved to `{checkpoint_path}`")

    def eval(self):
        for _, model in self.models.items():
            model.eval()

    def to(self, device):
        for _, model in self.models.items():
            model.to(device)
        for _, optimizer in self.optimizers.items():
            optimizer_to(optimizer, device)

    def load_from_checkpoint(self, checkpoint_path: str, device='cpu'):
        data = torch.load(checkpoint_path, map_location=torch.device(device))

        for name, subdata in data.items():
            if name in self.models:
                self.models[name].load_state_dict(subdata['model'])

            if name in self.optimizers and 'opt' in subdata:
                self.optimizers[name].load_state_dict(subdata['opt'])


class ExperienceReplayEpisode:
    def __init__(self):
        self.actions = []
        self.terminal_rewards = [0, 0]

    def on_action(self, action, white_player_reward: int, done: bool):
        if done:
            self.terminal_rewards[0] = white_player_reward
            self.terminal_rewards[1] = -white_player_reward
        self.actions.append(action)

    def yield_training_tuples_from_episode(self, env):
        terminal_rewards = self.terminal_rewards
        yield -1, env, terminal_rewards[0]

        for action in self.actions:
            env.step(action)
            player_idx = 0 if env.is_white_to_move() else 1
            reward = terminal_rewards[player_idx]
            yield action, env, reward
            if env.done:
                break


class ExperienceReplay:
    """
    Store played games history
    """
    def __init__(self, max_episodes, create_env_func):
        self.episodes = deque()
        self.max_episodes = max_episodes
        self.create_env_func = create_env_func

    def split(self, chunks: int):
        ers = [ExperienceReplay(self.max_episodes, self.create_env_func) for _ in range(chunks)]
        for i, episode in enumerate(self.episodes):
            ers[i % chunks].append_replay_episode(episode)
        return ers

    def sample(self, keep_fraction: float):
        res = ExperienceReplay(self.max_episodes, self.create_env_func)
        for er in np.random.choice(list(self.episodes), size=int(len(self.episodes) * keep_fraction), replace=False):
            res.append_replay_episode(er)
        return res

    def append_replay_episode(self, replay_episode: ExperienceReplayEpisode):
        self.episodes.append(replay_episode)
        while len(self.episodes) > self.max_episodes:
            self.episodes.popleft()

    def clear(self):
        self.episodes.clear()

    def yield_training_tuples(self):
        create_env_func = self.create_env_func
        for episode in self.episodes:
            yield from episode.yield_training_tuples_from_episode(create_env_func())

    def get_cnt_actions(self):
        return sum(len(ep.actions) for ep in self.episodes)

    def print_stats(self):
        eps = len(self.episodes)
        ep_lens = np.array([ len(ep.actions) + 1 for ep in self.episodes ])
        states = sum(ep_lens)
        m0 = min(ep_lens)
        m1 = np.mean(ep_lens)
        m2 = max(ep_lens)
        wr = Counter([ ep.terminal_rewards[0] for ep in self.episodes ])
        print(f"Episodes {eps}, states {states}: episode lengths min {m0}, mean {m1}, max {m2}. White rewards: {wr}")


class TensorboardSummaryWriter(SummaryWriter):
    """
    Wrapper around tensorboard to add points one by one
    """
    def __init__(self):
        super().__init__()
        self.points_cnt = Counter()
        self.figure_cnt = Counter()

    def append_scalar(self, name, value):
        step = self.points_cnt[name]
        self.points_cnt[name] += 1
        self.add_scalar(name, value, step)


class MemorySummaryWriter:
    def __init__(self):
        self.points = defaultdict(list)

    def append_scalar(self, name, value):
        self.points[name].append(value)


def split_uniformly(cnt: int, chunks: int):
    split = [cnt // chunks] * chunks
    for i in range(cnt % chunks):
        split[i] += 1
    while len(split) > 0 and split[-1] == 0:
        split.pop()
    return split


def clone_model(old_model, new_model):
    new_model.load_state_dict(copy.deepcopy(old_model.state_dict()))


# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def battle(env: BaseEnv, agent0: BaseAgent, agent1: BaseAgent, n_games: int, n_max_steps: int, randomize_first_turn: bool=False, show_game_stats: bool = False):
    """
    Play `n_games` where each game is at most `n_max_steps` turns.
    Return counter of game results as array `[
        agent0  first turn and agent0 wins,
        agent0 second turn and agent0 wins,
        agent1  first turn and agent1 wins,
        agent1 second turn and agent1 wins,
        draws
    ]`
    """

    start_time = time.time()
    results = [0, 0, 0, 0, 0]
    agents = [agent0, agent1]

    for game_i in range(n_games):
        env.reset()

        agent_id_first_turn = game_i % 2  # Switch sides every other game

        done = False

        for step in range(n_max_steps):
            env_player_idx = 0 if env.is_white_to_move() else 1
            agent_idx = (agent_id_first_turn + env_player_idx) % 2

            if randomize_first_turn and step == 0:
                action = env.get_random_action()
            else:
                action = agents[agent_idx].get_action(env)

            reward, done = env.step(action)
            white_player_reward = reward if env_player_idx == 0 else -reward
            # white_player_reward = reward if agent_idx == agent_id_first_turn else -reward

            if done:
                if white_player_reward == 1:
                    results[2 * agent_id_first_turn] += 1
                else:
                    # white loses
                    # => agent_id_first_turn loses
                    # => (1 - agent_id_first_turn) plays second turn and wins
                    results[2 * (1 - agent_id_first_turn) + 1] += 1
                break

        if not done:   # draw
            results[4] += 1

        if show_game_stats:
            print(f"Game {game_i}/{n_games}, {step} steps, results: {results}. Time passed {time.time() - start_time}")

    return results


def show_battle_results(name0, name1, battle_results):
    print(f"{name0} vs {name1}:")
    print(f"  {name0} white wins: {battle_results[0]}")
    print(f"  {name0} black wins: {battle_results[1]}")
    print(f"  {name1} white wins: {battle_results[2]}")
    print(f"  {name1} black wins: {battle_results[3]}")
    print(f"               draws: {battle_results[4]}")

    wins = battle_results[0] + battle_results[1]
    losses = battle_results[2] + battle_results[3]
    games = np.sum(battle_results)

    print(f"  {name0} wins {wins / games:.5f}, losses {losses / games:.5f}")


def print_encoding(name: str, inp: np.ndarray):
    print(f"Input {name} shape {inp.shape}:")

    if len(inp.shape) == 3:
        binary_channels = [chi for chi in range(inp.shape[0]) if np.sum(inp[chi, :, :] == 0) + np.sum(inp[chi, :, :] == 1) == inp.shape[1] * inp.shape[2]]

        for chi in range(inp.shape[0]):
            if chi not in binary_channels:
                print(f"Not binary channel {chi}:")
                print(inp[chi, :, :])

        if len(binary_channels) > 0:
            print("Binary channels content:")
            print_tabs = [[] for _ in range(inp.shape[2])]
            for row in range(inp.shape[1]):
                for col in range(inp.shape[2]):
                    if np.sum(inp[binary_channels, row, col]) == 0:
                        print_tabs[col].append('---')
                    else:
                        print_tabs[col].append(','.join([ str(chi) for chi in range(inp.shape[0]) if chi in binary_channels and inp[chi, row, col] == 1 ]))

            print_tabs_content(print_tabs)

    else:
        print(inp)


def print_tabs_content(print_tabs):
    print_tab_sizes = [max([len(line) for line in tab] + [0]) for tab in print_tabs]
    for li in range(max([ len(tab) for tab in print_tabs ])):
        content = []
        for ti, tab in enumerate(print_tabs):
            line = tab[li] if li < len(tab) else ''
            content.append(line + ' ' * (print_tab_sizes[ti] - len(line)))

        print('   '.join(content))


class RandomAgent(BaseAgent):
    def get_action(self, env: BaseEnv):
        return env.get_random_action()


class XStepLookaheadAgent(BaseAgent):
    def __init__(self, lookahead_steps_cnt: int) -> None:
        assert lookahead_steps_cnt > 0
        self.lookahead_steps_cnt = lookahead_steps_cnt

    def _dfs_get_best_action(self, curr_env, depth):
        values_and_actions = []

        for action in curr_env.get_valid_actions_iter():
            sub_env = curr_env.copy()

            reward, done = sub_env.step(action)

            if done:
                value = reward

            elif depth == self.lookahead_steps_cnt - 1:
                value = 0

            else:
                sub_values_and_actions = self._dfs_get_best_action(sub_env, depth + 1)
                value = max((v for v,a in sub_values_and_actions))
                if curr_env.is_white_to_move() != sub_env.is_white_to_move():
                    value = -value

            values_and_actions.append((value, action))

        return values_and_actions

    def get_action(self, env: BaseEnv) -> int:
        top_values_and_actions = self._dfs_get_best_action(env, 0)

        if len(top_values_and_actions) == 0:
            return env.get_random_action()

        grouped_by_value = defaultdict(list)
        for value, action in top_values_and_actions:
            grouped_by_value[value].append(action)

        if 1 in grouped_by_value:
            # Pick winning move
            return np.random.choice(grouped_by_value[1])

        if 0 in grouped_by_value:
            # Pick move without clear outcome
            return np.random.choice(grouped_by_value[0])

        # Pick losing move
        return np.random.choice(grouped_by_value[-1])


class OneStepLookaheadAgent(XStepLookaheadAgent):
    def __init__(self) -> None:
        super().__init__(lookahead_steps_cnt=1)


class TwoStepLookaheadAgent(XStepLookaheadAgent):
    def __init__(self) -> None:
        super().__init__(lookahead_steps_cnt=2)


class ThreeStepLookaheadAgent(XStepLookaheadAgent):
    def __init__(self) -> None:
        super().__init__(lookahead_steps_cnt=3)


def torch_create_dataloader(dataset: list, device: str, batch_size: int, shuffle: bool, drop_last: bool):
    def __tuple_to_device(tpl):
        return tuple(x.to(device) for x in torch.utils.data.dataloader.default_collate(tpl))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=__tuple_to_device)
    return dataloader


def torch_batch_map_to_dataset(dataloader, fmap):
    dataset = []
    for batch_input in dataloader:
        batch_output = fmap(batch_input)
        for row in zip(*batch_output):
            dataset.append(row)
    return dataset
