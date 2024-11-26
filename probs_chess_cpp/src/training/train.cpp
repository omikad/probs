#include "training/train.h"

using namespace std;


namespace probs {


void worker(ProbsImpl& impl, SafeQueue<shared_ptr<QueueItem>>& taskQueue, SafeQueue<shared_ptr<QueueItem>>& resultsQueue) {
    auto thread_id = this_thread::get_id();
    cout << "[Worker " << thread_id << "] started." << endl;

    try {
        while (true) {
            auto command = taskQueue.dequeue();
            if (!command) break; // Exit signal

            if (auto command_self_play = dynamic_pointer_cast<QueueCommand_SelfPlay>(command)) {
                // cout << "[Worker " << thread_id << "] Got command self play " << command_self_play->n_games << " games" << endl;
                auto rows = SelfPlay(impl.config_parser, command_self_play->n_games);
                resultsQueue.enqueue(make_shared<QueueResponse_SelfPlay>(rows));
            } else {
                cout << "[Worker " << thread_id << "] Unknown command type!" << endl;
            }
        }
        cout << "[Worker " << thread_id << "] stopped." << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
    }
}


ProbsImpl::ProbsImpl(const ConfigParser& config_parser) : config_parser(config_parser) {
    int n_threads = config_parser.GetInt("infra.n_threads");
    taskQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);
    resultQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);

    for (int ti = 0; ti < n_threads; ti++)
        workers.push_back(thread(worker, ref(*this), ref(taskQueues[ti]), ref(resultQueues[ti])));
}


void ProbsImpl::SelfPlayAndTrainV(const int v_train_episodes, const double dataset_drop_ratio) {
    int wcnt = workers.size();

    vector<int> worker_games(wcnt, v_train_episodes / wcnt);
    for (int wi = 0; wi < v_train_episodes % wcnt; wi++)
        worker_games[wi]++;

    for (int wi = 0; wi < wcnt; wi++) {
        int curr_games = worker_games[wi];
        if (curr_games > 0)
            taskQueues[wi].enqueue(make_shared<QueueCommand_SelfPlay>(curr_games));
    }

    vector<pair<torch::Tensor, float>> v_dataset;

    for (int wi = 0; wi < wcnt; wi++) {
        auto response = resultQueues[wi].dequeue();

        if (auto response_self_play = dynamic_pointer_cast<QueueResponse_SelfPlay>(response)) {
            cout << "Got self play response from worker " << wi << " with " << response_self_play->v_dataset.size() << " rows" << endl;
            for (auto& item: response_self_play->v_dataset)
                v_dataset.push_back(item);
        }
    }

    ResNet v_model(config_parser, "model.v", true);             // TODO: load V
    TrainV(config_parser, v_model, v_dataset);
}


void ProbsImpl::GoTrain() {
    int n_high_level_iterations = config_parser.GetInt("training.n_high_level_iterations");
    int v_train_episodes = config_parser.GetInt("training.v_train_episodes");
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio");
    cout << "[Train] Start training:" << endl;
    cout << "  n_high_level_iterations = " << n_high_level_iterations << endl;
    cout << "  v_train_episodes = " << v_train_episodes << endl;
    cout << "  dataset_drop_ratio = " << dataset_drop_ratio << endl;

    for (int high_level_i = 0; high_level_i < n_high_level_iterations; high_level_i++) {
        SelfPlayAndTrainV(v_train_episodes, dataset_drop_ratio);

    }

    // Shutdown workers
    for (int wi = 0; wi < workers.size(); wi++) taskQueues[wi].enqueue(nullptr);
    for (auto& worker : workers) worker.join();
}

}  // namespace probs