#include <ATen/Device.h>
#include <time.h>

#include "training/train.h"

using namespace std;


namespace probs {


void worker(ProbsImpl& impl, SafeQueue<shared_ptr<QueueItem>>& taskQueue, SafeQueue<shared_ptr<QueueItem>>& resultsQueue, int worker_idx) {
    srand(time(NULL) + worker_idx);
    // torch::set_num_threads(1);   // seems like this here don't improve things

    auto thread_id = this_thread::get_id();
    cout << "[WORKER " << thread_id << "] started." << endl;

    try {
        while (true) {
            auto command = taskQueue.dequeue();
            if (!command) break; // Exit signal

            if (auto command_self_play = dynamic_pointer_cast<QueueCommand_SelfPlay>(command)) {
                // cout << "[WORKER " << thread_id << "] Got command self play " << command_self_play->n_games << " games. Device = " << impl.device << endl;
                auto rows = SelfPlay(impl.model_keeper.q_model, impl.device, impl.config_parser, command_self_play->n_games);
                resultsQueue.enqueue(make_shared<QueueResponse_SelfPlay>(rows));
            } else if (auto command_get_q = dynamic_pointer_cast<QueueCommand_GetQDataset>(command)) {
                cout << "[WORKER " << thread_id << "] Got command get q dataset " << command_get_q->n_games << " games. Device = " << impl.device << endl;
                auto rows = GetQDataset(impl.model_keeper.v_model, impl.model_keeper.q_model, impl.device, impl.config_parser, command_get_q->n_games);
                resultsQueue.enqueue(make_shared<QueueResponse_QDataset>(rows));
            } else {
                cout << "[WORKER " << thread_id << "] Unknown command type!" << endl;
            }
        }
        cout << "[WORKER " << thread_id << "] stopped." << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
    }
}


ProbsImpl::ProbsImpl(const ConfigParser& config_parser)
        : config_parser(config_parser),
        model_keeper(config_parser, "model.v", "model.q", "training"),
        device(torch::kCPU) {
    int n_threads = config_parser.GetInt("infra.n_threads");
    taskQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);
    resultQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);

    for (int wi = 0; wi < n_threads; wi++)
        workers.push_back(thread(worker, ref(*this), ref(taskQueues[wi]), ref(resultQueues[wi]), wi));
}


void ProbsImpl::SelfPlayAndTrainV(UsageCounter& usage, const int v_train_episodes) {
    int wcnt = workers.size();

    vector<int> worker_games(wcnt, v_train_episodes / wcnt);
    for (int wi = 0; wi < v_train_episodes % wcnt; wi++)
        worker_games[wi]++;

    for (int wi = 0; wi < wcnt; wi++) {
        int curr_games = worker_games[wi];
        if (curr_games > 0)
            taskQueues[wi].enqueue(make_shared<QueueCommand_SelfPlay>(curr_games));
    }

    VDataset v_dataset;

    for (int wi = 0; wi < wcnt; wi++) {
        if (worker_games[wi] == 0) continue;

        auto response = resultQueues[wi].dequeue();

        if (auto response_self_play = dynamic_pointer_cast<QueueResponse_SelfPlay>(response)) {
            cout << "Got self play response from worker " << wi << " with " << response_self_play->v_dataset.size() << " rows" << endl;
            for (auto& item: response_self_play->v_dataset)
                v_dataset.push_back(item);
        }
    }
    usage.MarkCheckpoint("get V dataset");

    TrainV(config_parser, model_keeper.v_model, device, model_keeper.v_optimizer, v_dataset);
    usage.MarkCheckpoint("train V");
}


void ProbsImpl::GetQDatasetAndTrain(UsageCounter& usage, const int q_train_episodes) {
    int wcnt = workers.size();

    vector<int> worker_games(wcnt, q_train_episodes / wcnt);
    for (int wi = 0; wi < q_train_episodes % wcnt; wi++)
        worker_games[wi]++;

    for (int wi = 0; wi < wcnt; wi++) {
        int curr_games = worker_games[wi];
        if (curr_games > 0)
            taskQueues[wi].enqueue(make_shared<QueueCommand_GetQDataset>(curr_games));
    }

    QDataset q_dataset;

    for (int wi = 0; wi < wcnt; wi++) {
        auto response = resultQueues[wi].dequeue();

        if (auto response_q_dataset = dynamic_pointer_cast<QueueResponse_QDataset>(response)) {
            cout << "Got self play response from worker " << wi << " with " << response_q_dataset->q_dataset.size() << " rows" << endl;
            for (auto& item: response_q_dataset->q_dataset)
                q_dataset.push_back(item);
        }
    }
    usage.MarkCheckpoint("get Q dataset");

    TrainQ(config_parser, model_keeper.q_model, device, model_keeper.q_optimizer, q_dataset);
    usage.MarkCheckpoint("train Q");
}


void ProbsImpl::GoTrain() {
    int batch_size = config_parser.GetInt("training.batch_size");
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio");
    bool exploration_full_random = config_parser.KeyExist("training.exploration_full_random");
    int exploration_num_first_moves = config_parser.GetInt("training.exploration_num_first_moves");
    int n_high_level_iterations = config_parser.GetInt("training.n_high_level_iterations");
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    int v_train_episodes = config_parser.GetInt("training.v_train_episodes");
    int q_train_episodes = config_parser.GetInt("training.q_train_episodes");
    int tree_num_q_s_a_calls = config_parser.GetInt("training.tree_num_q_s_a_calls");
    int tree_max_depth = config_parser.GetInt("training.tree_max_depth");

    cout << "[TRAIN] Start training:" << endl;
    cout << "  batch_size = " << batch_size << endl;
    cout << "  exploration_full_random = " << (exploration_full_random ? "true" : "false") << endl;
    cout << "  exploration_num_first_moves = " << exploration_num_first_moves << endl;
    cout << "  n_high_level_iterations = " << n_high_level_iterations << endl;
    cout << "  n_max_episode_steps = " << n_max_episode_steps << endl;
    cout << "  v_train_episodes = " << v_train_episodes << endl;
    cout << "  q_train_episodes = " << q_train_episodes << endl;
    cout << "  tree_num_q_s_a_calls = " << tree_num_q_s_a_calls << endl;
    cout << "  tree_max_depth = " << tree_max_depth << endl;

    torch::set_num_threads(1);

    int gpu_num = config_parser.GetInt("infra.gpu");
    cout << "[TRAIN] GPU: " << gpu_num << endl;
    if (gpu_num >= 0) {
        if (torch::cuda::is_available())
            device = at::Device("cuda:" + to_string(gpu_num));
        else
            throw Exception("Config points to GPU which is not available (config parameter infra.gpu)");
        model_keeper.v_model->to(device);
        model_keeper.q_model->to(device);
    }

    for (int high_level_i = 0; high_level_i < n_high_level_iterations; high_level_i++) {
        UsageCounter usage;

        SelfPlayAndTrainV(usage, v_train_episodes);

        model_keeper.SetEvalMode();
        GetQDatasetAndTrain(usage, q_train_episodes);

        model_keeper.SetEvalMode();
        model_keeper.SaveCheckpoint();
        usage.MarkCheckpoint("save models");

        usage.PrintStats();
    }

    // Shutdown workers
    for (int wi = 0; wi < workers.size(); wi++) taskQueues[wi].enqueue(nullptr);
    for (auto& worker : workers) worker.join();
}

}  // namespace probs