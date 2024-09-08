from collections import Counter
import numpy as np
import multiprocessing
import threading
import torch
import time

import probs_impl_common
import helpers


ENV = None
AGENTS_DATA = None
N_MAX_STEPS = 0
RANDOMIZE_N_TURNS = 1
VERBOSE = False


def battle(env: helpers.BaseEnv, agent0: helpers.BaseAgent, agent1: helpers.BaseAgent, n_games: int, n_max_steps: int, randomize_n_turns: int, show_game_stats: bool = False, stats: Counter = None):
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
    agent_turn_times = [[], []]
    results = [0, 0, 0, 0, 0]
    agents = [agent0, agent1]

    for game_i in range(n_games):
        env.reset()

        agent_id_first_turn = game_i % 2  # Switch sides every other game

        done = False

        for step in range(n_max_steps):
            env_player_idx = 0 if env.is_white_to_move() else 1
            agent_idx = (agent_id_first_turn + env_player_idx) % 2

            if step < randomize_n_turns:
                action = env.get_random_action()
            else:
                turn_start_time = time.time()
                action = agents[agent_idx].get_action(env)
                agent_turn_times[agent_idx].append(time.time() - turn_start_time)

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
            print(f"Game {game_i}/{n_games}, {step} steps. first {randomize_n_turns} turns randomized, results: {results}. Time passed total {time.time() - start_time}")
        
        if stats is not None:
            for agent_idx in range(2):
                times = agent_turn_times[agent_idx]
                stats[(agent_idx, "turn_time_max")] = max(stats[(agent_idx, "turn_time_max")], np.max(times))
                stats[(agent_idx, "turn_time_sum")] += np.sum(times)
                stats[(agent_idx, "turn_time_cnt")] += len(times)

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


def _multiprocessing_battle_p0_p1(args):
    torch.set_num_threads(1)

    seed = int(time.time() * threading.get_native_id()) % 100000007
    np.random.seed(seed)
    torch.manual_seed(seed + 5)

    p0, p1, n_games = args
    p0_agent = AGENTS_DATA[p0]
    p1_agent = AGENTS_DATA[p1]

    if hasattr(p0_agent.__class__, 'eval'): p0_agent.eval()
    if hasattr(p1_agent.__class__, 'eval'): p1_agent.eval()

    stats = Counter()

    with torch.no_grad():
        curr_battle_results = battle(ENV, p0_agent, p1_agent, n_games=n_games, n_max_steps=N_MAX_STEPS, randomize_n_turns=RANDOMIZE_N_TURNS, stats=stats)

    if VERBOSE:
        print(f"Thread {threading.get_native_id()}. Battle {p0_agent.get_name()} vs {p1_agent.get_name()} (games={n_games}, max_steps={N_MAX_STEPS}, randomize_n_turns={RANDOMIZE_N_TURNS}): {curr_battle_results}")

    return p0, p1, curr_battle_results, stats


def multiprocessing_battle(env, agent0, agent1, n_games: int, n_max_steps: int, randomize_n_turns: int, n_threads: int = 0):
    battle_stats = Counter()

    if n_threads <= 1:
        battle_results = battle(env, agent0, agent1, n_games=n_games, n_max_steps=n_max_steps, randomize_n_turns=randomize_n_turns, stats=battle_stats)
    else:
        global ENV
        global AGENTS_DATA
        global N_MAX_STEPS
        global RANDOMIZE_N_TURNS
        global VERBOSE

        ENV = env
        AGENTS_DATA = [agent0, agent1]
        N_MAX_STEPS = n_max_steps
        RANDOMIZE_N_TURNS = randomize_n_turns
        VERBOSE = True

        rounds = helpers.split_uniformly(n_games, chunks=n_threads)

        battle_results = np.zeros(5, dtype=np.int32)

        with multiprocessing.get_context("fork").Pool(n_threads) as multiprocessing_pool:
            for p0, p1, sub_battle_results, sub_battle_stats in multiprocessing_pool.imap_unordered(_multiprocessing_battle_p0_p1, [(0, 1, curr_n_rounds) for curr_n_rounds in rounds]):
                battle_results += sub_battle_results
                for agent_idx in range(2):
                    battle_stats[(agent_idx, "turn_time_max")] = max(battle_stats[(agent_idx, "turn_time_max")], sub_battle_stats[(agent_idx, "turn_time_max")])
                    battle_stats[(agent_idx, "turn_time_sum")] += sub_battle_stats[(agent_idx, "turn_time_sum")]
                    battle_stats[(agent_idx, "turn_time_cnt")] += sub_battle_stats[(agent_idx, "turn_time_cnt")]

        battle_results = battle_results.tolist()

    score = (battle_results[0] + battle_results[1] + 0.5 * battle_results[4]) / sum(battle_results)
    print(f"Args: games={n_games}, max_steps={n_max_steps}, randomize_n_turns={randomize_n_turns})")

    for agent_idx in range(2):
        turn_time_max = battle_stats[(agent_idx, "turn_time_max")]
        turn_time_sum = battle_stats[(agent_idx, "turn_time_sum")]
        turn_time_cnt = battle_stats[(agent_idx, "turn_time_cnt")]
        print(f"Agent {(agent0 if agent_idx == 0 else agent1).get_name()}, turn timings: max {turn_time_max}, average {turn_time_sum / turn_time_cnt}")

    wins = battle_results[0] + battle_results[1]
    losses = battle_results[2] + battle_results[3]
    games = np.sum(battle_results)

    print(f"{agent0.get_name()} vs {agent1.get_name()}: wins {wins / games:.5f}, losses {losses / games:.5f}. Battle {battle_results}, score = {score}")

    return battle_results