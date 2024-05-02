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


def show_usage():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    print(f"Memory usage: usertime={usage[0]} systime={usage[1]} mem={usage[2]/1024.0} mb")


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

    def load_from_checkpoint(self, checkpoint_path: str):
        data = torch.load(checkpoint_path)

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


class ExperienceReplay:
    """
    Store played games history
    """
    def __init__(self, max_episodes, create_env_func):
        self.episodes = deque()
        self.max_episodes = max_episodes
        self.create_env_func = create_env_func

    def append_replay_episode(self, replay_episode: ExperienceReplayEpisode):
        self.episodes.append(replay_episode)
        while len(self.episodes) > self.max_episodes:
            self.episodes.popleft()

    def clear(self):
        self.episodes.clear()

    def yield_training_tuples(self):
        create_env_func = self.create_env_func
        for episode in self.episodes:
            yield from ExperienceReplay.yield_training_tuples_from_episode(create_env_func(), episode)

    @staticmethod
    def yield_training_tuples_from_episode(env, episode: ExperienceReplayEpisode):
        terminal_rewards = episode.terminal_rewards
        yield -1, env, terminal_rewards[0]

        for i, action in enumerate(episode.actions):
            env.step(action)
            reward = terminal_rewards[(i + 1) % 2]
            if not env.done:
                yield action, env, reward


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

    def append_scalar(self, name, value):
        step = self.points_cnt[name]
        self.points_cnt[name] += 1
        self.add_scalar(name, value, step)


class MemorySummaryWriter:
    def __init__(self):
        self.points = defaultdict(list)

    def append_scalar(self, name, value):
        self.points[name].append(value)


def clone_model(old_model, new_model):
    new_model.load_state_dict(copy.deepcopy(old_model.state_dict()))


def battle(env: BaseEnv, agent0: BaseAgent, agent1: BaseAgent, n_games: int, n_max_steps: int, randomize_first_turn: bool=False):
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

    results = [0, 0, 0, 0, 0]
    agents = [agent0, agent1]

    for game in range(n_games):
        env.reset()

        agent_id_first_turn = game % 2  # Switch sides every other game

        done = False

        for step in range(n_max_steps):
            curr_player_idx = (agent_id_first_turn + step) % 2

            if randomize_first_turn and step == 0:
                action = env.get_random_action()
            else:
                action = agents[curr_player_idx].get_action(env)

            reward, done = env.step(action)
            white_player_reward = reward if curr_player_idx == agent_id_first_turn else -reward

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

    return results


def show_battle_results(name0, name1, battle_results, randomize_first_turn=False):
    print(f"{name0} vs {name1} {'with' if randomize_first_turn else 'without'} first turn randomized:")
    print(f"  {name0} white wins: {battle_results[0]}")
    print(f"  {name0} black wins: {battle_results[1]}")
    print(f"  {name1} white wins: {battle_results[2]}")
    print(f"  {name1} black wins: {battle_results[3]}")
    print(f"               draws: {battle_results[4]}")

    wins = battle_results[0] + battle_results[1]
    losses = battle_results[2] + battle_results[3]
    games = np.sum(battle_results)

    print(f"  {name0} wins {wins / games:.5f}, losses {losses / games:.5f}")


class RandomAgent(BaseAgent):
    def get_action(self, env: BaseEnv):
        return env.get_random_action()


class OneStepLookaheadAgent(BaseAgent):
    """
    Check all possible actions outcome (full scan depth of 1).
    If there is a winning move - play it immediately.
    Ignore all losing moves and play random safe action.
    """
    def get_action(self, env: BaseEnv) -> int:
        ok_moves = []
        for action in env.get_valid_actions_iter():
            curr_env = env.copy()
            reward, done = curr_env.step(action)
            if done and reward == 1:
                return action
            elif not done:
                ok_moves.append(action)

        if len(ok_moves) == 0:
            return env.get_random_action()

        return np.random.choice(ok_moves)


class XStepLookaheadAgent(BaseAgent):
    def __init__(self, lookahead_steps_cnt: int) -> None:
        assert lookahead_steps_cnt > 0
        self.lookahead_steps_cnt = lookahead_steps_cnt

    def __dfs_get_values_and_actions(self, curr_env, depth):
        values_and_actions = []

        for action in curr_env.get_valid_actions_iter():
            sub_env = curr_env.copy()

            reward, done = sub_env.step(action)

            if done:
                value = reward

            elif depth == self.lookahead_steps_cnt - 1:
                value = 0

            else:
                sub_values_and_actions = self.__dfs_get_values_and_actions(sub_env, depth + 1)
                value = -max((v for v,a in sub_values_and_actions))

            values_and_actions.append((value, action))

        return values_and_actions

    def get_action(self, env: BaseEnv) -> int:
        top_values_and_actions = self.__dfs_get_values_and_actions(env, 0)

        if len(top_values_and_actions) == 0:
            return env.get_random_action()

        grouped_by_value = defaultdict(list)
        for value, action in top_values_and_actions:
            grouped_by_value[value].append(action)

        if 1 in grouped_by_value:
            # Pick winning move
            return np.random.choice(grouped_by_value[1])

        if 0 in grouped_by_value:
            # Pick move with unknown outcome
            return np.random.choice(grouped_by_value[0])

        # Pick losing move
        return np.random.choice(grouped_by_value[-1])


class TwoStepLookaheadAgent(XStepLookaheadAgent):
    def __init__(self) -> None:
        super().__init__(lookahead_steps_cnt=2)


class ThreeStepLookaheadAgent(XStepLookaheadAgent):
    def __init__(self) -> None:
        super().__init__(lookahead_steps_cnt=3)


class BaseSpielEnv(BaseEnv):
    def __init__(self, game, prev_state = None, prev_done: bool = False):
        self.game = game
        self.n_actions = game.num_distinct_actions()
        self.state = prev_state
        self.done = prev_done
        if prev_state is None:
            self.reset()

    def reset(self):
        self.state = self.game.new_initial_state()
        self.done = False

    def render_ascii(self):
        if self.done:
            print("Game finished")
        else:
            print(self.state.observation_string())
            print(self.get_rotated_encoded_state())

    def get_valid_actions_mask(self) -> np.ndarray:
        state = self.state

        legal_actions = state.legal_actions(state.current_player())
        action_mask = np.zeros(self.n_actions, dtype=np.int32)
        action_mask[legal_actions] = 1
        return action_mask

    def copy(self):
        state_copy = self.state.clone()
        return self.__class__(self.game, state_copy, self.done)

    def step(self, action: int):
        state = self.state

        player_idx = state.current_player()
        state.apply_action(action)
        self.done = state.is_terminal()

        if not self.done:
            return 0, False

        return state.rewards()[player_idx], True

    def is_white_to_move(self):
        return self.state.current_player() == 0

    def get_rotated_encoded_state(self):
        raise Exception("Need to be implemented in child")
