import numpy as np
import torch
import pprint
import dataclasses
import numpy as np
import traceback
from PyQt5.QtWidgets import QApplication
import sys
import yaml

import environments
import helpers
from probs_impl import probs_impl_common, probs_impl_main
import play_chess_qt
import battle


@torch.no_grad()
def cmd_test():
    from environments.my_chess_env import go_test_chess_correctness, go_test_chess_encoding_correctness
    from environments.py_chess_env import go_test_env_works_as_mychess
    go_test_chess_encoding_correctness()
    go_test_chess_correctness()
    go_test_env_works_as_mychess()


@torch.no_grad()
def cmd_battle(config, device):
    agent1 = probs_impl_common.create_agent(config['player1'], config['env']['name'], device)
    agent2 = probs_impl_common.create_agent(config['player2'], config['env']['name'], device)
    env = environments.get_create_env_func(config['env']['name'])()
    battle.multiprocessing_battle(
        env,
        agent1,
        agent2,
        n_games=config['evaluate']['evaluate_n_games'],
        n_max_steps=config['env']['n_max_episode_steps'],
        randomize_n_turns=config['evaluate']['randomize_n_turns'],
        n_threads=config['infra']['threads_cnt'])


@torch.no_grad()
def cmd_play_chess():
    app = QApplication(sys.argv)
    window = play_chess_qt.Window()
    sys.exit(app.exec_())


@torch.no_grad()
def cmd_interactive_play(config: dict, device):
    env = environments.get_create_env_func(config['env']['name'])()
    enemy = probs_impl_common.create_agent(config['enemy'], config['env']['name'], device)
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
                action = enemy.get_action(env)
                print(f"{enemy.get_name()} move {action}")
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
                vs = enemy.value_model.forward(*[torch.unsqueeze(torch.as_tensor(inp), dim=0) for inp in inputs]).detach().cpu().numpy()[0, 0]
                qa = probs_impl_common.get_q_a_single_state(enemy.value_model, enemy.self_learning_model, env, 'cpu')
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


def cmd_train(config, device):
    evaluage_enemy = probs_impl_common.create_agent(config['evaluate']['enemy'], config['env']['name'], device)
    model_keeper = probs_impl_common.create_model_keeper(config['model'], config['env']['name'])
    probs_impl_main.go_train(config, device, model_keeper, evaluage_enemy)
    return model_keeper


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python go_probs.py <config_file>")
        return

    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(False)

    with open(sys.argv[1]) as fin:
        config = yaml.safe_load(fin)
    print(config['name'])

    if config['infra'].get('log', 'none') == 'tf':
        helpers.TENSORBOARD = helpers.TensorboardSummaryWriter()
    elif config['infra'].get('log', 'none') == 'mem':
        helpers.TENSORBOARD = helpers.MemorySummaryWriter()
    print("Tensorboard: ", 'none' if helpers.TENSORBOARD is None else helpers.TENSORBOARD.__class__.__name__)

    if config['infra']['device'] is None or config['infra']['device'] == 'cpu':
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pprint.pprint(config, width=120)

    if config['cmd'] == 'test':
        cmd_test()
    elif config['cmd'] == 'battle':
        cmd_battle(config, device)
    elif config['cmd'] == 'train':
        cmd_train(config, device)
    elif config['cmd'] == 'play_chess':
        cmd_play_chess()
    elif config['cmd'] == 'interactive_play':
        cmd_interactive_play(config, device)

    if isinstance(helpers.TENSORBOARD, helpers.MemorySummaryWriter):
        for key, val in helpers.TENSORBOARD.points.items():
            m0 = np.min(val) if len(val) > 0 else '-'
            m1 = np.mean(val) if len(val) > 0 else '-'
            m2 = np.max(val) if len(val) > 0 else '-'
            l = val[-1] if len(val) > 0 else '-'
            print(f"Memory tensorboard: `{key}` has {len(val)} points (min, mean, max, last) = {m0, m1, m2, l}")


if __name__ == "__main__":
    main()


