import argparse
import numpy as np
import torch
import pprint
import dataclasses
import numpy as np
import traceback
from PyQt5.QtWidgets import QApplication
import sys

import environments
import helpers
import probs_impl_common
import probs_impl_main
import play_chess_qt
import battle


ARGS = None
TENSORBOARD = None
TRAIN_DEVICE = None
TRAIN_PARAMS = None
MODEL_TRAIN_PARAMS = None


@torch.no_grad()
def cmd_test():
    from environments.my_chess_env import go_test_chess_correctness, go_test_chess_encoding_correctness
    from environments.py_chess_env import go_test_env_works_as_mychess
    go_test_chess_encoding_correctness()
    go_test_chess_correctness()
    go_test_env_works_as_mychess()


@torch.no_grad()
def cmd_battle():
    agent0 = probs_impl_common.create_agent(ARGS, ARGS.env, TRAIN_PARAMS, ARGS.model, TRAIN_DEVICE)
    agent1 = probs_impl_common.create_agent(ARGS, ARGS.env, TRAIN_PARAMS, ARGS.enemy, TRAIN_DEVICE)
    env = TRAIN_PARAMS.create_env_func()
    battle.multiprocessing_battle(env, agent0, agent1, n_games=TRAIN_PARAMS.evaluate_n_games, n_max_steps=TRAIN_PARAMS.n_max_episode_steps, randomize_n_turns=2, n_threads=TRAIN_PARAMS.self_play_threads)


@torch.no_grad()
def cmd_play_chess():
    app = QApplication(sys.argv)
    window = play_chess_qt.Window()
    sys.exit(app.exec_())


@torch.no_grad()
def cmd_play_vs_human():
    env = TRAIN_PARAMS.create_env_func()
    agent = probs_impl_common.create_agent(ARGS, ARGS.env, TRAIN_PARAMS, ARGS.model, TRAIN_DEVICE)
    step_i = 0

    while True:
        print()
        env.render_ascii()
        pl = "0" if env.is_white_to_move() else "1"
        print(f"* Step {step_i}. Player {pl}. Valid actions: {' '.join(map(str, np.where(env.get_valid_actions_mask() == 1)[0]))}")
        print(f"Type your action, or type 'r' for random move, or type 'a' to make AI move")

        try:
            inp = input().strip().lower()
            if inp == 'a':
                action = agent.get_action(env)
                print(f"{agent.get_name()} move {action}")
            elif inp == 'r':
                action = env.get_random_action()
                print(f"Random move {action}")
            elif inp == 'd':
                inputs = env.get_rotated_encoded_state()
                for ii, inp in enumerate(inputs):
                    helpers.print_encoding(str(ii), inp)
                continue
            elif inp == 'aa':
                inputs = env.get_rotated_encoded_state()
                vs = agent.value_model.forward(*[torch.unsqueeze(torch.as_tensor(inp), dim=0) for inp in inputs]).detach().cpu().numpy()[0, 0]
                qa = probs_impl_common.get_q_a_single_state(agent.value_model, agent.self_learning_model, env, 'cpu')
                print(f"Value(state) = {vs}")
                print(f"Q(state, *) = {qa}")
                for action in env.get_valid_actions_iter():
                    print(f"Q(state, action={action}) = {qa[action]}")
                continue

            else:
                action = int(inp)

            reward, done = env.step(action)
            print(f"Reward {reward}, done {done}")

        except Exception:
            traceback.print_exc()
            continue

        step_i += 1
        if done:
            env.render_ascii()
            break


def cmd_train():
    if ARGS.enemy is None:
        TRAIN_PARAMS.evaluate_agent = helpers.RandomAgent()
    else:
        TRAIN_PARAMS.evaluate_agent = probs_impl_common.create_agent(ARGS, ARGS.env, TRAIN_PARAMS, ARGS.enemy, TRAIN_DEVICE)

    model_keeper = probs_impl_common.create_model_keeper(ARGS, ARGS.env, TRAIN_PARAMS, model_str=ARGS.model)
    pprint.pprint(dataclasses.asdict(TRAIN_PARAMS), sort_dicts=False)

    probs_impl_main.go_train(TRAIN_PARAMS, model_keeper, TRAIN_DEVICE)

    if isinstance(TENSORBOARD, helpers.MemorySummaryWriter):
        for key, val in TENSORBOARD.points.items():
            m0 = np.min(val) if len(val) > 0 else '-'
            m1 = np.mean(val) if len(val) > 0 else '-'
            m2 = np.max(val) if len(val) > 0 else '-'
            print(f"Memory tensorboard: `{key}` has {len(val)} points (min, mean, max) = {m0, m1, m2}")

    return model_keeper


def main():
    global ARGS
    global TENSORBOARD
    global TRAIN_DEVICE
    global TRAIN_PARAMS
    global MODEL_TRAIN_PARAMS

    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', required=True)
    parser.add_argument('--env', required=True)
    parser.add_argument('--model')
    parser.add_argument('--enemy')
    parser.add_argument('--log')
    parser.add_argument('--checkpoints_dir')
    parser.add_argument('--sub_processes_cnt', type=int)
    parser.add_argument('--self_play_threads', type=int)
    parser.add_argument('--n_high_level_iterations', type=int)
    parser.add_argument('--n_max_episode_steps', type=int)
    parser.add_argument('--q_dataset_episodes_sub_iter', type=int)
    parser.add_argument('--device')
    parser.add_argument('--num_q_s_a_calls', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--evaluate_n_games', type=int, required=False)
    parser.add_argument('--models', action='append', required=False)
    parser.add_argument('--alphazero_move_num_sampling_moves', type=float)
    parser.add_argument('--v_pick_second_best_prob', type=float)
    parser.add_argument('--greedy_freq', type=float)
    parser.add_argument('--dir_alpha', type=float)
    parser.add_argument('--expl_frac', type=float)
    parser.add_argument('--value_lr', type=float)
    parser.add_argument('--value_batch_size', type=int)
    parser.add_argument('--self_learning_lr', type=float)
    parser.add_argument('--self_learning_batch_size', type=int)
    parser.add_argument('--get_q_dataset_batch_size', type=int)
    parser.add_argument('--v_train_episodes', type=int)
    parser.add_argument('--q_train_episodes', type=int)
    parser.add_argument('--dataset_drop_ratio', type=float)
    ARGS = parser.parse_args()

    if ARGS.log == 'tf':
        TENSORBOARD = helpers.TensorboardSummaryWriter()
    else:
        TENSORBOARD = helpers.MemorySummaryWriter()
    helpers.TENSORBOARD = TENSORBOARD
    print("Tensorboard:", TENSORBOARD.__class__.__name__)

    if ARGS.device is None or ARGS.device == 'cpu':
        TRAIN_DEVICE = 'cpu'
    else:
        TRAIN_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Train device {TRAIN_DEVICE}")

    TRAIN_PARAMS = probs_impl_common.Parameters(
        env_name = ARGS.env,
        create_env_func = environments.get_create_env_func(ARGS, ARGS.env),

        sub_processes_cnt = (ARGS.sub_processes_cnt if ARGS.sub_processes_cnt is not None else 0),
        self_play_threads = (ARGS.self_play_threads if ARGS.self_play_threads is not None else 1),

        value_lr                 = (ARGS.value_lr if ARGS.value_lr is not None else 0.001),
        value_batch_size         = (ARGS.value_batch_size if ARGS.value_batch_size is not None else 256),
        self_learning_lr         = (ARGS.self_learning_lr if ARGS.self_learning_lr is not None else 0.0003),
        self_learning_batch_size = (ARGS.self_learning_batch_size if ARGS.self_learning_batch_size is not None else 256),
        get_q_dataset_batch_size = (ARGS.get_q_dataset_batch_size if ARGS.get_q_dataset_batch_size is not None else 256),
        v_train_episodes         = (ARGS.v_train_episodes if ARGS.v_train_episodes is not None else 100),
        q_train_episodes         = (ARGS.q_train_episodes if ARGS.q_train_episodes is not None else 100),
        dataset_drop_ratio       = (ARGS.dataset_drop_ratio if ARGS.dataset_drop_ratio is not None else 0),

        alphazero_move_num_sampling_moves  = ARGS.alphazero_move_num_sampling_moves,
        v_pick_second_best_prob            = (ARGS.v_pick_second_best_prob if ARGS.v_pick_second_best_prob is not None else 0.0),
        greedy_action_freq                 = ARGS.greedy_freq,
        dirichlet_alpha                    = (ARGS.dir_alpha if ARGS.dir_alpha is not None else 0.2),
        exploration_fraction               = (ARGS.expl_frac if ARGS.expl_frac is not None else 0.1),

        mem_max_episodes            = 100000,
        n_high_level_iterations     = (ARGS.n_high_level_iterations if ARGS.n_high_level_iterations is not None else 2),
        n_max_episode_steps         = (ARGS.n_max_episode_steps if ARGS.n_max_episode_steps is not None else 200),
        num_q_s_a_calls             = (ARGS.num_q_s_a_calls if ARGS.num_q_s_a_calls is not None else 50),
        max_depth                   = (ARGS.max_depth if ARGS.max_depth is not None else 1e5),
        q_dataset_episodes_sub_iter = (ARGS.q_dataset_episodes_sub_iter if ARGS.q_dataset_episodes_sub_iter is not None else 500),

        evaluate_agent   = None,
        evaluate_n_games = (ARGS.evaluate_n_games if ARGS.evaluate_n_games is not None else 200),

        checkpoints_dir = ARGS.checkpoints_dir,
    )

    helpers.show_usage()

    if ARGS.cmd == 'test':
        cmd_test()
    elif ARGS.cmd == 'battle':
        cmd_battle()
    elif ARGS.cmd == 'train':
        cmd_train()
    elif ARGS.cmd == 'play_chess':
        cmd_play_chess()
    elif ARGS.cmd == 'play_vs_human':
        cmd_play_vs_human()
    else:
        parser.print_help()
        return

    helpers.show_usage()


if __name__ == "__main__":
    main()


