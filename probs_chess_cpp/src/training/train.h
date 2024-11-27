#pragma once
#include <vector>
#include <utility>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <torch/torch.h>

#include "infra/config_parser.h"
#include "neural/encoder.h"
#include "utils/ts_queue.h"
#include "training/model_keeper.h"
#include "training/v_train.h"


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
    std::vector<std::pair<torch::Tensor, float>> v_dataset;
    explicit QueueResponse_SelfPlay(const std::vector<std::pair<torch::Tensor, float>>& rows) : v_dataset(rows) {}
};


class ProbsImpl {
    public:
        ProbsImpl(const ConfigParser& config_parser);
        void GoTrain();
        void SelfPlayAndTrainV(const int v_train_episodes, const double dataset_drop_ratio);
        const ConfigParser& config_parser;
        ModelKeeper model_keeper;

    private:
        std::vector<SafeQueue<std::shared_ptr<QueueItem>>> taskQueues;
        std::vector<SafeQueue<std::shared_ptr<QueueItem>>> resultQueues;
        std::vector<std::thread> workers;
};

}  // namespace probs
