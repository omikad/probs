#pragma once
#include <iostream>
#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"
#include "utils/exception.h"

using namespace std;


namespace probs {

class PositionHistoryTree {
    public:
        PositionHistoryTree(const string& starting_fen);
        PositionHistoryTree(const lczero::PositionHistory& lchistory);

        lczero::GameResult ComputeGameResult(const int node) const;

        const lczero::Position& Last() const {return positions.back();}
        const int LastIndex() {
            assert(positions.size() > 0);
            return positions.size() - 1;
        }

        lczero::PositionHistory ToLczeroHistory(const int node) const;

        /// @brief create new node based on `node` with applied move `move`. Return new node index
        int Append(const int node, lczero::Move move);

        /// @brief remove last node from the tree
        void PopLast();

        /// @brief node -> board position
        vector<lczero::Position> positions;

        /// @brief node -> board hash
        vector<uint64_t> hashes;

        /// @brief rnode -> list of kids
        vector<vector<int>> kids;

        /// @brief node -> parent node (or -1 for root)
        vector<int> parents;

    private:
        int ComputeLastMoveRepetitions(const int node, int* cycle_length) const;
};

}   // namespace probs