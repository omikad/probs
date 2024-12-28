#pragma once

#include <future>
#include <atomic>
#include <vector>
#include <string>
#include <optional>
#include <chrono>

#include "infra/config_parser.h"
#include "chess/game_tree.h"
#include "chess/board.h"
#include "chess/bitboard.h"
#include "neural/encoder.h"

namespace probs {


struct KidInfo {
    int kid_node;
    lczero::Move move;
    float q_nn_score;
    float q_tree_score;
};


struct NodeInfo {
    bool is_terminal;
    std::vector<KidInfo> kids;
    int transform;
    lczero::InputPlanes input_planes;
    float v_tree_score;
};


class UciPlayer {
    public:
        UciPlayer(ConfigParser& config);
        void OnNewGame();
        void SetDebug(bool turn_on);
        void SetPosition(const std::string& starting_fen, const std::vector<std::string>& moves);
        void WaitForReadyState();
        void Stop();

        void StartSearch(
            std::optional<std::chrono::milliseconds> search_wtime,
            std::optional<std::chrono::milliseconds> search_btime,
            std::optional<std::chrono::milliseconds> search_winc,
            std::optional<std::chrono::milliseconds> search_binc,
            std::optional<int> search_moves_to_go,
            std::optional<int> search_depth,
            std::optional<uint64_t> search_nodes,
            std::optional<int> search_mate,
            std::optional<std::chrono::milliseconds> search_fixed_time,
            bool search_infinite,
            std::vector<std::string>& search_moves
        );

    private:
        int AppendLastTreeNode();
        int FindKidIndex(const int node, const lczero::Move move) const;
        int n_max_episode_steps;
        PositionHistoryTree tree;
        bool debug_on;
        std::atomic<bool> is_in_search;
        std::atomic<bool> stop_search_flag;
        std::future<void> search_future;
        std::string last_pos_fen;
        std::vector<std::string> last_pos_moves;
        int top_node;
        std::vector<NodeInfo> nodes;
};

}  // namespace probs
