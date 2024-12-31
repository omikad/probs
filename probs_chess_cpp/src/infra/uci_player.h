#pragma once

#include <future>
#include <atomic>
#include <vector>
#include <string>
#include <optional>
#include <chrono>
#include <ATen/Device.h>

#include "infra/config_parser.h"
#include "chess/game_tree.h"
#include "chess/board.h"
#include "chess/bitboard.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "neural/torch_encoder.h"
#include "training/model_keeper.h"
#include "training/training_helpers.h"

namespace probs {


struct KidInfo {
    int kid_node;
    lczero::Move move;
    float q_nn_score;
    float q_tree_score;
};


struct NodeInfo {
    bool is_terminal;
    bool q_model_called;
    std::vector<KidInfo> kids;
    float v_tree_score;
};


struct SearchConstraintsInfo {
    std::optional<std::chrono::milliseconds> wtime;
    std::optional<std::chrono::milliseconds> btime;
    std::optional<std::chrono::milliseconds> winc;
    std::optional<std::chrono::milliseconds> binc;
    std::optional<int> moves_to_go;
    std::optional<int> depth;
    std::optional<uint64_t> nodes;
    std::optional<int> mate;
    std::optional<std::chrono::milliseconds> fixed_time;
    bool infinite;
    std::vector<std::string> moves;
};


class UciPlayer {
    public:
        UciPlayer(ConfigParser& config);
        void OnNewGame();
        void SetDebug(bool turn_on);
        void SetPosition(const std::string& starting_fen, const std::vector<std::string>& moves);
        void WaitForReadyState();
        void Stop();
        void StartSearch(SearchConstraintsInfo& search_info);

    private:
        int AppendLastTreeNode();
        int FindKidIndex(const int node, const lczero::Move move) const;
        void EstimateNodesActions(const std::vector<int>& nodes);

        lczero::Move GoSearch__Random();
        lczero::Move GoSearch__SingleQCall();
        lczero::Move GoSearch__FullSearch();
        int search_mode;
        SearchConstraintsInfo search_info;

        bool debug_on;
        std::atomic<bool> is_in_search;
        std::atomic<bool> stop_search_flag;
        std::future<void> search_future;

        int n_max_episode_steps;
        PositionHistoryTree tree;
        std::string last_pos_fen;
        std::vector<std::string> last_pos_moves;
        int top_node;
        std::vector<NodeInfo> nodes;

        ResNet q_model;
        at::Device device;
        int q_agent_batch_size;
};

}  // namespace probs
