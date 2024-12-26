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

namespace probs {

class UciPlayer {
    public:
        UciPlayer(ConfigParser& config);
        void onNewGame();
        void setDebug(bool turn_on);
        void setPosition(const std::string& starting_fen, const std::vector<std::string>& moves);
        void waitForReadyState();
        void stop();

        void startSearch(
            bool is_reading_moves,
            std::optional<std::chrono::milliseconds> wtime,
            std::optional<std::chrono::milliseconds> btime,
            std::optional<std::chrono::milliseconds> winc,
            std::optional<std::chrono::milliseconds> binc,
            std::optional<int> moves_to_go,
            std::optional<int> depth,
            std::optional<uint64_t> nodes,
            std::optional<int> mate,
            std::optional<std::chrono::milliseconds> fixed_time,
            bool infinite,
            std::vector<std::string>& search_moves
        );

    private:
        int n_max_episode_steps;
        PositionHistoryTree tree;
        bool debug_on;
        std::atomic<bool> is_in_search;
        std::atomic<bool> stop_search_flag;
        std::future<void> search_future;
};

}  // namespace probs
