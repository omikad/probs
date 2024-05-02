import argparse
import numpy as np
import torch
import pprint
import dataclasses
import numpy as np
import traceback
import time

import environments
import helpers
import probs_impl_common
import probs_impl_main


torch.set_num_threads(1)
print("Torch number of threads:", torch.get_num_threads())

parser = argparse.ArgumentParser()
parser.add_argument('--cmd', required=True)
parser.add_argument('--env', required=True)
parser.add_argument('--model')
parser.add_argument('--enemy')
parser.add_argument('--log')
parser.add_argument('--checkpoints_dir')
parser.add_argument('--self_play_threads', type=int)
parser.add_argument('--n_high_level_iterations', type=int)
parser.add_argument('--n_games_per_iteration', type=int)
parser.add_argument('--n_max_episode_steps', type=int)
parser.add_argument('--num_q_s_a_calls', type=int)
parser.add_argument('--max_depth', type=int)
parser.add_argument('--evaluate_n_games', type=int)
parser.add_argument('--randomize_first_turn', action=argparse.BooleanOptionalAction)
ARGS = parser.parse_args()

if ARGS.log == 'tf':
    TENSORBOARD = helpers.TensorboardSummaryWriter()
else:
    TENSORBOARD = helpers.MemorySummaryWriter()
helpers.TENSORBOARD = TENSORBOARD
print("Tensorboard:", TENSORBOARD.__class__.__name__)

MODEL_TRAIN_PARAMS = environments.get_default_train_params(ARGS, ARGS.env)

TRAIN_PARAMS = probs_impl_common.Parameters(
    create_env_func = environments.get_create_env_func(ARGS.env),

    self_play_threads=(ARGS.self_play_threads if ARGS.self_play_threads is not None else 1),

    value_lr                =MODEL_TRAIN_PARAMS['value_lr'],
    value_batch_size        =MODEL_TRAIN_PARAMS['value_batch_size'],
    self_learning_lr        =MODEL_TRAIN_PARAMS['self_learning_lr'],
    self_learning_batch_size=MODEL_TRAIN_PARAMS['self_learning_batch_size'],
    dirichlet_alpha         =MODEL_TRAIN_PARAMS['dirichlet_alpha'],
    exploration_fraction    =MODEL_TRAIN_PARAMS['exploration_fraction'],

    mem_max_episodes=100000,
    n_high_level_iterations=(ARGS.n_high_level_iterations if ARGS.n_high_level_iterations is not None else 2),
    n_games_per_iteration=(ARGS.n_games_per_iteration if ARGS.n_games_per_iteration is not None else 100),
    n_max_episode_steps=(ARGS.n_max_episode_steps if ARGS.n_max_episode_steps is not None else 100),
    num_q_s_a_calls=(ARGS.num_q_s_a_calls if ARGS.num_q_s_a_calls is not None else 50),
    max_depth=(ARGS.max_depth if ARGS.max_depth is not None else 1e5),

    evaluate_agent=None,
    evaluate_n_games=(ARGS.evaluate_n_games if ARGS.evaluate_n_games is not None else 100),

    checkpoints_dir=ARGS.checkpoints_dir,
)


def parse_model_str(model_str):
    res = dict()
    for part in model_str.split(','):
        key, val = part.split('=')
        res[key] = val
    return res


def create_model_keeper(model_str) -> helpers.ModelKeeper:
    parsed_model_str = parse_model_str(model_str)

    model_keeper = helpers.ModelKeeper()

    model_keeper.models['value'] = environments.create_value_model(ARGS, ARGS.env, parsed_model_str['V'])
    model_keeper.models['self_learner'] = environments.create_self_learning_model(ARGS, ARGS.env, parsed_model_str['SL'])

    model_keeper.optimizers['value'] = torch.optim.AdamW( model_keeper.models['value'].parameters(), lr=TRAIN_PARAMS.value_lr, weight_decay=1e-5)
    model_keeper.optimizers['self_learner'] = torch.optim.AdamW(model_keeper.models['self_learner'].parameters(), lr=TRAIN_PARAMS.self_learning_lr, weight_decay=1e-5)

    if 'CKPT' in parsed_model_str:
        checkpoint = parsed_model_str['CKPT']
        model_keeper.load_from_checkpoint(checkpoint)
        print(f"Models loaded from `{checkpoint}`:")

    for name, model in model_keeper.models.items():
        total_parameters = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {model.__class__.__name__} Params cnt: {total_parameters}")
        for name, par in model.named_parameters():
            print("         ", name, par.numel())

    return model_keeper


def create_agent(model_str) -> helpers.BaseAgent:
    if model_str == 'random':
        return helpers.RandomAgent()
    elif model_str == 'one_step_lookahead':
        return helpers.OneStepLookaheadAgent()
    elif model_str == 'two_step_lookahead':
        return helpers.TwoStepLookaheadAgent()
    elif model_str == 'three_step_lookahead':
        return helpers.ThreeStepLookaheadAgent()
    else:
        model_keeper = create_model_keeper(model_str=model_str)
        model_keeper.eval()
        return probs_impl_common.SelfLearningAgent(model_keeper.models['self_learner'].__class__.__name__, model_keeper=model_keeper)


@torch.no_grad()
def cmd_battle():
    env = TRAIN_PARAMS.create_env_func()
    agent0 = create_agent(ARGS.model)
    agent1 = create_agent(ARGS.enemy)

    n_games = (ARGS.evaluate_n_games if ARGS.evaluate_n_games is not None else 1000)
    randomize_first_turn = (ARGS.randomize_first_turn if ARGS.randomize_first_turn is not None else False)
    n_max_episode_steps = (ARGS.n_max_episode_steps if ARGS.n_max_episode_steps is not None else 200)

    battle_results = helpers.battle(env, agent0, agent1, n_games=n_games, n_max_steps=n_max_episode_steps, randomize_first_turn=randomize_first_turn)
    helpers.show_battle_results(agent0.get_name(), agent1.get_name(), battle_results, randomize_first_turn=randomize_first_turn)


@torch.no_grad()
def cmd_play_vs_human():
    env = TRAIN_PARAMS.create_env_func()
    replay_episode = helpers.ExperienceReplayEpisode()
    agent = create_agent(ARGS.enemy)
    step_i = 0

    while True:
        print()
        env.render_ascii()
        pl = "0" if env.is_white_to_move() else "1"
        print(f"* Step {step_i}. Player {pl}. Valid actions: {' '.join(map(str, np.where(env.get_valid_actions_mask() == 1)[0]))}")
        print(f"Type your action, or type 'a' to make AI move")

        try:
            inp = input().strip().lower()
            if inp == 'a':
                action = agent.get_action(env)
                print(f"{agent.get_name()} move {action}")
            else:
                action = int(inp)

            is_white = env.is_white_to_move()
            reward_mul = 1 if is_white else -1

            reward, done = env.step(action)
            replay_episode.on_action(action, reward * reward_mul, done)
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
        TRAIN_PARAMS.evaluate_agent = create_agent(ARGS.enemy)

    model_keeper = create_model_keeper(model_str=ARGS.model)
    pprint.pprint(dataclasses.asdict(TRAIN_PARAMS), sort_dicts=False)

    impl = probs_impl_main.ProbsAlgorithmImpl(
            TRAIN_PARAMS,
            helpers.ExperienceReplay(max_episodes=TRAIN_PARAMS.mem_max_episodes, create_env_func=TRAIN_PARAMS.create_env_func))
    impl.go_train(model_keeper)

    if isinstance(TENSORBOARD, helpers.MemorySummaryWriter):
        for key, val in TENSORBOARD.points.items():
            m0 = np.min(val) if len(val) > 0 else '-'
            m1 = np.mean(val) if len(val) > 0 else '-'
            m2 = np.max(val) if len(val) > 0 else '-'
            l = val[-1] if len(val) > 0 else '-'
            print(f"Memory tensorboard: `{key}` has {len(val)} points (min, mean, max) = {m0, m1, m2}, last = {l}")

    return model_keeper


def main():
    helpers.show_usage()

    if ARGS.cmd == 'battle':
        cmd_battle()
    elif ARGS.cmd == 'train':
        cmd_train()
    elif ARGS.cmd == 'play_vs_human':
        cmd_play_vs_human()
    else:
        parser.print_help()
        return

    helpers.show_usage()


if __name__ == "__main__":
    main()


