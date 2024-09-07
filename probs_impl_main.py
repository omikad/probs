import torch
import threading
import os
import multiprocessing
from collections import Counter
import pickle

import environments
import battle
import helpers
import probs_impl_common
import probs_impl_self_play
import probs_impl_train_q
import probs_impl_train_v


def worker(tasks_queue, results_queue, params, device, v_class_name, q_class_name):
    torch.set_num_threads(1)  # Important for multiprocessing

    proc = multiprocessing.current_process()
    print(f"Process started: identity {proc._identity}, pid {os.getpid()}, thread native id {threading.get_native_id()}")

    value_model = environments.create_value_model(params.env_name, v_class_name)
    self_learning_model = environments.create_self_learning_model(params.env_name, q_class_name)
    value_model.to(device)
    self_learning_model.to(device)

    while True:
        pi, task, package = tasks_queue.get(True, None)
        # print(f"Process {proc._identity}, pi={pi}, got task type `{task}`")

        if task == "load_v_model":
            state_dict = pickle.loads(package)
            # print(f"Process {proc._identity}, pi={pi}, task `load_v_model`, pickle size is {len(package)}")
            value_model.load_state_dict(state_dict)

        elif task == "load_q_model":
            state_dict = pickle.loads(package)
            # print(f"Process {proc._identity}, pi={pi}, task `load_q_model`, pickle size is {len(package)}")
            self_learning_model.load_state_dict(state_dict)

        elif task == "get_q_dataset":
            n_subprocess_games = package
            print(f"Process {proc._identity}, pi={pi}, task is compute {n_subprocess_games} episodes")
            dataset, stats = probs_impl_train_q.get_dataset(n_subprocess_games, value_model, self_learning_model, params, device)
            print(f"Process {proc._identity}, pi={pi}, computed {len(dataset)} rows, sending result")
            results_queue.put_nowait((pi, "got_q_dataset", (dataset, stats)))

        elif task == "stop":
            break


@torch.no_grad()
def report_model_performance(agent, env: helpers.BaseEnv, enemy_agent: helpers.BaseAgent, n_evaluate_games: int, n_max_steps: int, save_to_tf: bool):
    agent.value_model.eval()
    agent.self_learning_model.eval()

    results = battle.battle(env, agent, enemy_agent, n_games=n_evaluate_games, n_max_steps=n_max_steps, randomize_n_turns=1)

    wins = results[0] + results[1]
    losses = results[2] + results[3]
    draws = results[4]

    print(f"  Trained agent total wins {wins}, losses {losses}, draws {draws}. Detailed result: {results} (with randomized first turn)")

    if save_to_tf:
        helpers.TENSORBOARD.append_scalar("wins", (wins + 0.5 * draws)/n_evaluate_games)


def go_train_self_learning_model(
        model_keeper: helpers.ModelKeeper,
        tasks_queues: list[multiprocessing.Queue],
        results_queue: multiprocessing.Queue,
        self_learning_model: helpers.BaseSelfLearningModel,
        params: probs_impl_common.Parameters,
        value_model: helpers.BaseValueModel,
        agent: probs_impl_common.SelfLearningAgent,
        device: str,
        optimizer):

    value_model.eval()
    self_learning_model.eval()

    if params.sub_processes_cnt == 0:
        dataset, stats = probs_impl_train_q.get_dataset(params.q_train_episodes, value_model, self_learning_model, params, device)
    else:
        for pi in range(params.sub_processes_cnt):
            tasks_queues[pi].put_nowait((pi, "load_v_model", pickle.dumps(value_model.state_dict())))
            tasks_queues[pi].put_nowait((pi, "load_q_model", pickle.dumps(self_learning_model.state_dict())))

        for n_games in helpers.split_uniformly(cnt=params.q_train_episodes, chunks=(params.q_train_episodes + params.q_dataset_episodes_sub_iter - 1) // params.q_dataset_episodes_sub_iter):

            task_episode_splits = helpers.split_uniformly(cnt=n_games, chunks=params.sub_processes_cnt)
            print(f"Main thread send task `get_q_dataset` with {task_episode_splits} episodes")
            for pi, n_subprocess_games in enumerate(task_episode_splits):
                tasks_queues[pi].put_nowait((pi, "get_q_dataset", n_subprocess_games))

            dataset, stats = [], Counter()

            for task_i in range(params.sub_processes_cnt):
                pi, itemtype, package = results_queue.get(True, None)

                if itemtype == "got_q_dataset":
                    sub_dataset, sub_stats = package
                    # print(f"Main thread got result `{itemtype}`: dataset with {len(sub_dataset)} rows and stats {sub_stats}")
                    dataset.extend(sub_dataset)
                    stats += sub_stats

                else:
                    raise Exception(f"Got unexpected item from results queue {pi, itemtype}")

            helpers.TENSORBOARD.append_scalar('beam_search_avg_depth', stats['depth_sum'] / stats['depth_cnt'])
            helpers.TENSORBOARD.append_scalar('reused_tree_size', stats['reused_tree_size_sum'] / stats['trees_cnt'])
            helpers.TENSORBOARD.append_scalar('tree_size', stats['tree_size_sum'] / stats['trees_cnt'])
            helpers.TENSORBOARD.append_scalar('greedy_action_freq', stats['greedy_action_sum'] / stats['greedy_action_cnt'])

            print(f"Self learner model, processed {n_games} episodes. Go train with dataset length {len(dataset)}")
            probs_impl_train_q.train_q_model(dataset, value_model, self_learning_model, params, device, optimizer)

            value_model.eval()
            self_learning_model.eval()
            model_keeper.save_checkpoint(params.checkpoints_dir, "  ")

            report_model_performance(
                agent,
                env=params.create_env_func(),
                enemy_agent=params.evaluate_agent,
                n_evaluate_games=params.evaluate_n_games,
                n_max_steps=params.n_max_episode_steps,
                save_to_tf=True)


def go_train(params: probs_impl_common.Parameters, model_keeper: helpers.ModelKeeper, train_device: str):
    if params.self_play_threads > 1:
        assert params.train_device == 'cpu'

    value_model = model_keeper.models['value']
    value_optimizer = model_keeper.optimizers['value']
    self_learning_model = model_keeper.models['self_learner']
    self_learning_optimizer = model_keeper.optimizers['self_learner']

    if params.sub_processes_cnt > 0:
        assert params.self_play_threads == 1
        multiprocessing.set_start_method('spawn')
        tasks_queues = [multiprocessing.Queue() for _ in range(params.sub_processes_cnt)]
        results_queue = multiprocessing.Queue()
        for pi in range(params.sub_processes_cnt):
            v_class_name = value_model.__class__.__name__
            q_class_name = self_learning_model.__class__.__name__
            p = multiprocessing.Process(target=worker, args=(tasks_queues[pi], results_queue, params, train_device, v_class_name, q_class_name))
            p.start()
    else:
        tasks_queues, results_queue = None, None

    model_keeper.to(train_device)

    experience_replay = helpers.ExperienceReplay(max_episodes=params.mem_max_episodes, create_env_func=params.create_env_func)

    agent = probs_impl_common.SelfLearningAgent(self_learning_model.__class__.__name__, model_keeper, train_device)

    for high_level_i in range(params.n_high_level_iterations):
        print(f"High level iteration {high_level_i}/{params.n_high_level_iterations}")
        usage = helpers.UsageCounter()
        experience_replay.clear()

        probs_impl_self_play.go_self_play(value_model, self_learning_model, params, experience_replay, train_device)

        experience_replay.print_stats()

        usage.checkpoint("Self play")

        probs_impl_train_v.train_value_model(value_model, train_device, value_optimizer, experience_replay, params.value_batch_size, params.dataset_drop_ratio)

        usage.checkpoint("Train value")

        go_train_self_learning_model(
            model_keeper,
            tasks_queues,
            results_queue,
            self_learning_model,
            params,
            value_model,
            agent,
            train_device,
            self_learning_optimizer
        )

        usage.checkpoint("Train selfL and report wins")

        helpers.show_usage()
        helpers.show_models_memory_sizes([value_model, self_learning_model])
        usage.print_stats()

    if params.sub_processes_cnt > 0:
        for pi in range(params.sub_processes_cnt):
            tasks_queues[pi].put_nowait((pi, "stop", None))
