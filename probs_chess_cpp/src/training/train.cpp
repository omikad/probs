#include "training/train.h"

using namespace std;


namespace probs {


void worker(SafeQueue<shared_ptr<QueueItem>>& taskQueue, SafeQueue<shared_ptr<QueueItem>>& resultsQueue) {
    auto thread_id = this_thread::get_id();
    cout << "[Worker " << thread_id << "] started." << endl;

    while (true) {
        auto command = taskQueue.dequeue();
        if (!command) break; // Exit signal

        if (auto command_self_play = dynamic_pointer_cast<QueueCommand_SelfPlay>(command)) {
            cout << "[Worker " << thread_id << "] Got command self play " << command_self_play->n_games << " games" << endl;
        } else {
            cout << "[Worker " << thread_id << "] Unknown command type!" << endl;
        }
    }
    cout << "[Worker " << thread_id << "] stopped." << endl;
}


ProbsImpl::ProbsImpl(const ConfigParser& config_parser) : config_parser(config_parser) {
    int n_threads = config_parser.GetInt("infra.n_threads");
    taskQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);
    resultQueues = vector<SafeQueue<shared_ptr<QueueItem>>>(n_threads);

    for (int ti = 0; ti < n_threads; ti++)
        workers.push_back(thread(worker, ref(taskQueues[ti]), ref(resultQueues[ti])));
}


void ProbsImpl::GoTrain() {
    int n_high_level_iterations = config_parser.GetInt("training.n_high_level_iterations");
    cout << "[Train] Start training with " << n_high_level_iterations << "high level iterations" << endl;

    // for ()
}


}  // namespace probs