#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <torch/torch.h>

#include "infra/config_parser.h"
#include "utils/ts_queue.h"
#include "neural/encoder.h"

using namespace std;


namespace probs {


class QueueItem {
public:
    virtual ~QueueItem() = default;
};


class QueueCommand_SelfPlay : public QueueItem {
public:
    int n_games;
    explicit QueueCommand_SelfPlay(const int n_games) : n_games(n_games) {}
};


class QueueResponse_SelfPlay : public QueueItem {
public:
    vector<pair<lczero::InputPlanes, float>> v_dataset;
    explicit QueueResponse_SelfPlay(const vector<pair<lczero::InputPlanes, float>>& rows) : v_dataset(rows) {}
};


class ProbsImpl {
    public:
        ProbsImpl(const ConfigParser& config_parser);
        void GoTrain();

    private:
        const ConfigParser& config_parser;
        vector<SafeQueue<shared_ptr<QueueItem>>> taskQueues;
        vector<SafeQueue<shared_ptr<QueueItem>>> resultQueues;
        vector<thread> workers;
};

}  // namespace probs
