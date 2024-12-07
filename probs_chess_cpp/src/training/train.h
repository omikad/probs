#pragma once
#include <vector>
#include <utility>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <ATen/Device.h>

#include "infra/config_parser.h"
#include "infra/battle.h"
#include "infra/player.h"
#include "neural/encoder.h"
#include "utils/ts_queue.h"
#include "utils/usage_counter.h"
#include "training/model_keeper.h"
#include "training/training_helpers.h"
#include "training/v_train.h"
#include "training/q_train.h"
#include "neural/torch_encoder.h"


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
    VDataset v_dataset;
    explicit QueueResponse_SelfPlay(const VDataset& rows) : v_dataset(rows) {}
};


class QueueCommand_GetQDataset : public QueueItem {
public:
    int n_games;
    explicit QueueCommand_GetQDataset(const int n_games) : n_games(n_games) {}
};


class QueueResponse_QDataset : public QueueItem {
public:
    QDataset q_dataset;
    explicit QueueResponse_QDataset(const QDataset& rows) : q_dataset(rows) {}
};


class ProbsImpl {
    public:
        ProbsImpl(const ConfigParser& config_parser);
        void GoTrain();
        void SelfPlayAndTrainV(UsageCounter& usage, std::ofstream& losses_file, const int v_train_episodes);
        void GetQDatasetAndTrain(UsageCounter& usage, std::ofstream& losses_file, const int q_train_episodes);
        const ConfigParser& config_parser;
        ModelKeeper model_keeper;
        at::Device device;

    private:
        std::vector<SafeQueue<std::shared_ptr<QueueItem>>> taskQueues;
        std::vector<SafeQueue<std::shared_ptr<QueueItem>>> resultQueues;
        std::vector<std::thread> workers;
};

}  // namespace probs
