#include "chess/game_tree.h"

using namespace std;


namespace probs {

PositionHistoryTree::PositionHistoryTree(const string& starting_fen) {
    lczero::ChessBoard starting_board;
    int no_capture_ply;
    int full_moves;
    starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);

    int game_ply = full_moves * 2 - (starting_board.flipped() ? 1 : 2);
    lczero::Position root_position(starting_board, no_capture_ply, game_ply);

    positions.push_back(root_position);
    hashes.push_back(starting_board.Hash());
    kids.push_back({});
    parents.push_back(-1);
}


PositionHistoryTree::PositionHistoryTree(const lczero::PositionHistory& lchistory) {
    for (int pi = 0; pi < lchistory.GetLength(); pi++) {
        auto& position = lchistory.GetPositionAt(pi);

        positions.push_back(position);
        hashes.push_back(position.GetBoard().Hash());
        kids.push_back({});
        parents.push_back(pi - 1);

        if (pi > 0)
            kids[pi - 1].push_back(pi);
    }
}


lczero::GameResult PositionHistoryTree::ComputeGameResult(const int node) const {
    auto& position = positions[node];
    const auto& board = position.GetBoard();
    auto legal_moves = board.GenerateLegalMoves();
    if (legal_moves.empty()) {
        if (board.IsUnderCheck()) {
            return position.IsBlackToMove() ? lczero::GameResult::WHITE_WON : lczero::GameResult::BLACK_WON;
        }
        return lczero::GameResult::DRAW;
    }

    if (!board.HasMatingMaterial()) return lczero::GameResult::DRAW;
    if (position.GetRule50Ply() >= 100) return lczero::GameResult::DRAW;
    if (position.GetRepetitions() >= 2) return lczero::GameResult::DRAW;

    return lczero::GameResult::UNDECIDED;
};


int PositionHistoryTree::Append(const int node, lczero::Move move) {
    int new_node = (int)positions.size();

    positions.push_back(lczero::Position(positions[node], move));
    hashes.push_back(positions.back().GetBoard().Hash());
    kids.push_back({});
    parents.push_back(node);

    kids[node].push_back(new_node);

    int cycle_length;
    int repetitions = ComputeLastMoveRepetitions(new_node, &cycle_length);
    positions.back().SetRepetitions(repetitions, cycle_length);

    return new_node;
}


int PositionHistoryTree::ComputeLastMoveRepetitions(const int node, int* cycle_length) const {
    int distance = 1;
    int runnode = parents[node];
    while (runnode >= 0) {
        if (hashes[runnode] == hashes[node]) {
            *cycle_length = distance;
            return 1 + positions[runnode].GetRepetitions();
        }
        distance++;
        runnode = parents[runnode];
    }

    *cycle_length = 0;
    return 0;
}

}   // namespace probs