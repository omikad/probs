#include <ATen/Device.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <time.h>

#include "training/train.h"

using namespace std;


namespace probs {


void worker(ProbsImpl& impl, SafeQueue<shared_ptr<QueueItem>>& taskQueue, SafeQueue<shared_ptr<QueueItem>>& resultsQueue, int worker_idx) {
    srand(time(NULL) + worker_idx);
    // torch::set_num_threads(1);   // seems like this here don't improve things

    auto thread_id = this_thread::get_id();
    // cout << "[WORKER " << thread_id << "] started." << endl;

    try {
        while (true) {
            auto command = taskQueue.dequeue();
            if (!command) break; // Exit signal

            if (auto command_self_play = dynamic_pointer_cast<QueueCommand_SelfPlay>(command)) {
                // cout << "[WORKER " << thread_id << "] Got command self play " << command_self_play->n_games << " games. Device = " << impl.device << endl;
                auto rows = SelfPlay(impl.model_keeper.q_model, impl.device, impl.config_parser, command_self_play->n_games);
                resultsQueue.enqueue(make_shared<QueueResponse_SelfPlay>(rows));
            } else if (auto command_get_q = dynamic_pointer_cast<QueueCommand_GetQDataset>(command)) {
                // cout << "[WORKER " << thread_id << "] Got command get q dataset " << command_get_q->n_games << " games. Device = " << impl.device << endl;
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
    int n_threads = config_parser.GetInt("infra.n_threads", false, 1);
    taskQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);
    resultQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);

    for (int wi = 0; wi < n_threads; wi++)
        workers.push_back(thread(worker, ref(*this), ref(taskQueues[wi]), ref(resultQueues[wi]), wi));
}


void ProbsImpl::SelfPlayAndTrainV(UsageCounter& usage, ofstream& losses_file, const int v_train_episodes) {
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

    TrainV(config_parser, losses_file, model_keeper.v_model, device, model_keeper.v_optimizer, v_dataset);
    usage.MarkCheckpoint("train V");
}


void ProbsImpl::GetQDatasetAndTrain(UsageCounter& usage, ofstream& losses_file, const int q_train_episodes) {
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
        if (worker_games[wi] == 0) continue;
        auto response = resultQueues[wi].dequeue();

        if (auto response_q_dataset = dynamic_pointer_cast<QueueResponse_QDataset>(response)) {
            cout << "Got self play response from worker " << wi << " with " << response_q_dataset->q_dataset.size() << " rows" << endl;
            for (auto& item: response_q_dataset->q_dataset)
                q_dataset.push_back(item);
        }
    }
    usage.MarkCheckpoint("get Q dataset");

    TrainQ(config_parser, losses_file, model_keeper.q_model, device, model_keeper.q_optimizer, q_dataset);
    usage.MarkCheckpoint("train Q");
}


void ProbsImpl::GoTrain() {
    torch::set_num_threads(1);

    int batch_size = config_parser.GetInt("training.batch_size", true, 0);
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio", false, 0);
    bool exploration_full_random = config_parser.GetInt("training.exploration_full_random", false, 0) > 0;
    int exploration_num_first_moves = config_parser.GetInt("training.exploration_num_first_moves", true, 0);
    int n_high_level_iterations = config_parser.GetInt("training.n_high_level_iterations", true, 0);
    int q_train_sub_iterations = config_parser.GetInt("training.q_train_sub_iterations", true, 0);
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps", true, 0);
    int v_train_episodes = config_parser.GetInt("training.v_train_episodes", true, 0);
    int q_train_episodes = config_parser.GetInt("training.q_train_episodes", true, 0);
    int tree_num_q_s_a_calls = config_parser.GetInt("training.tree_num_q_s_a_calls", true, 0);
    int tree_max_depth = config_parser.GetInt("training.tree_max_depth", true, 0);
    int evaluate_n_games = config_parser.GetInt("infra.evaluate_n_games", true, 0);

    cout << "[TRAIN] Start training:" << endl;
    cout << "  batch_size = " << batch_size << endl;
    cout << "  exploration_full_random = " << (exploration_full_random ? "true" : "false") << endl;
    cout << "  exploration_num_first_moves = " << exploration_num_first_moves << endl;
    cout << "  n_high_level_iterations = " << n_high_level_iterations << endl;
    cout << "  q_train_sub_iterations = " << q_train_sub_iterations << endl;
    cout << "  n_max_episode_steps = " << n_max_episode_steps << endl;
    cout << "  v_train_episodes = " << v_train_episodes << endl;
    cout << "  q_train_episodes = " << q_train_episodes << endl;
    cout << "  tree_num_q_s_a_calls = " << tree_num_q_s_a_calls << endl;
    cout << "  tree_max_depth = " << tree_max_depth << endl;

    std::time_t t =  std::time(NULL);
    std::tm tm    = *std::localtime(&t);
    std::ostringstream losses_log_filename;
    losses_log_filename << config_parser.GetString("infra.losses_log_dir") << "/losses_" << std::put_time(&tm, "%Y%m%d-%H%M%S") << ".log";
    cout << "  losses log file name = " << losses_log_filename.str() << endl;
    
    ofstream losses_file;
    losses_file.open(losses_log_filename.str());

    device = GetDeviceFromConfig(config_parser);
    cout << "  device = " << device << endl;
    model_keeper.To(device);
    model_keeper.SetEvalMode();

    IPlayer* player1;
    IPlayer* player2;
    player1 = new QResnetPlayer(model_keeper.q_model, device, "QResnetPlayer");
    player2 = new NStepLookaheadPlayer("OneStepLookahead", 1);
    cout << "  evaluate on " << evaluate_n_games << " games " << player1->GetName() << " vs " << player2->GetName() << endl;

    for (int high_level_i = 0; high_level_i < n_high_level_iterations; high_level_i++) {
        UsageCounter usage;

        model_keeper.SetEvalMode();
        SelfPlayAndTrainV(usage, losses_file, v_train_episodes);

        for (int q_train_sub_i = 0; q_train_sub_i < q_train_sub_iterations; q_train_sub_i++) {
            model_keeper.SetEvalMode();
            GetQDatasetAndTrain(usage, losses_file, q_train_episodes);
        }

        model_keeper.SetEvalMode();
        model_keeper.SaveCheckpoint();
        cout << "[TRAIN] V model score on starting fen: " << GetVScoreOnStartingBoard(model_keeper.v_model, device) << endl;
        usage.MarkCheckpoint("save models");

        BattleInfo battle_info = ComparePlayers(*player1, *player2, evaluate_n_games, n_max_episode_steps);
        int w = battle_info.results[0][0] + battle_info.results[0][1];
        int d = battle_info.results[1][0] + battle_info.results[1][1];
        int l = battle_info.results[2][0] + battle_info.results[2][1];
        int games = w + d + l;
        double q_player_score = (w + (double)d / 2) / (double)games;
        cout << "[TRAIN] High level iteration " << high_level_i << " completed. ";
        cout << "Q player wins=" << battle_info.results[0][0] << "," << battle_info.results[0][1];
        cout << "; draws=" << battle_info.results[1][0] << "," << battle_info.results[1][1];
        cout << "; losses=" << battle_info.results[2][0] << "," << battle_info.results[2][1];
        cout << "; score=" << q_player_score << endl;
        losses_file << "WinScore: " << q_player_score << endl;
        usage.MarkCheckpoint("report wins");

        usage.PrintStats();
    }

    // Shutdown workers
    for (int wi = 0; wi < workers.size(); wi++) taskQueues[wi].enqueue(nullptr);
    for (auto& worker : workers) worker.join();

    delete player1;
    delete player2;
    losses_file.close();
}

}  // namespace probs