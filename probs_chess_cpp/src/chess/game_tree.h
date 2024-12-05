#pragma once
#include <iostream>
#include <vector>
#include <optional>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"
#include "utils/exception.h"


namespace probs {

class PositionHistoryTree {
    public:
        PositionHistoryTree(const std::string& starting_fen, const int n_max_episode_steps);
        // PositionHistoryTree(const lczero::PositionHistory& lchistory, const int n_max_episode_steps);

        lczero::GameResult GetGameResult(const int node) const {
            assert(node < (int)game_results.size());
            return game_results[node >= 0 ? node : (game_results.size() - 1)];
        }

        const lczero::Position& LastPosition() const {
            assert(positions.size() > 0);
            return positions.back();
        }

        const int LastIndex() const {
            assert(positions.size() > 0);
            return positions.size() - 1;
        }

        /// @brief create new node based on `node` with applied move `move`. Return new node index
        int Move(const int node, const lczero::Move move);

        /// @brief remove last node from the tree
        void PopLast();

        std::vector<int> GetHistoryPathNodes(const int node) const;

        // Note: tree doesn't have kid nodes, thats why this method runs through the whole tree
        std::vector<int> BFS(const int start_node) const;

        std::optional<int> GetRelativePositionScore(const int node) const;

        /// @brief node -> board position
        std::vector<lczero::Position> positions;

        /// @brief node -> vector of node valid moves
        std::vector<std::vector<lczero::Move>> node_valid_moves;

        /// @brief node -> parent node (or -1 for root)
        std::vector<int> parents;

    private:
        /// @brief node -> board hash
        std::vector<uint64_t> hashes;

        /// @brief node -> game result
        std::vector<lczero::GameResult> game_results;

        int n_max_episode_steps;

        int ComputeLastMoveRepetitions(const int node, int* cycle_length) const;

        void ComputeNodeGameResult(const int node);
};

}   // namespace probs