#include "chess/game_tree.h"

using namespace std;


namespace probs {

PositionHistoryTree::PositionHistoryTree(const string& starting_fen, const int n_max_episode_steps) : n_max_episode_steps(n_max_episode_steps) {
    lczero::ChessBoard starting_board;
    int no_capture_ply;
    int full_moves;
    starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);

    int game_ply = full_moves * 2 - (starting_board.flipped() ? 1 : 2);
    lczero::Position root_position(starting_board, no_capture_ply, game_ply);

    positions.push_back(root_position);
    hashes.push_back(starting_board.Hash());
    parents.push_back(-1);
    node_valid_moves.push_back({});
    game_results.push_back(lczero::GameResult::UNDECIDED);
    ComputeNodeGameResult(0);
}


vector<int> PositionHistoryTree::GetHistoryPathNodes(const int node) const {
    vector<int> result;
    int runnode = node >= 0 ? node : LastIndex();
    while (runnode >= 0) {
        result.push_back(runnode);
        runnode = parents[runnode];
    }
    reverse(begin(result), end(result));
    return result;
}


void PositionHistoryTree::ComputeNodeGameResult(const int node) {
    auto& position = positions[node];
    const auto& board = position.GetBoard();

    node_valid_moves[node] = board.GenerateLegalMoves();

    if (node_valid_moves[node].empty()) {
        if (board.IsUnderCheck())
            game_results[node] = position.IsBlackToMove() ? lczero::GameResult::WHITE_WON : lczero::GameResult::BLACK_WON;
        else
            game_results[node] = lczero::GameResult::DRAW;
    }
    else {
        if (position.GetGamePly() >= n_max_episode_steps) game_results[node] = lczero::GameResult::DRAW;
        else if (!board.HasMatingMaterial()) game_results[node] = lczero::GameResult::DRAW;
        else if (position.GetRule50Ply() >= 100) game_results[node] = lczero::GameResult::DRAW;
        else if (position.GetRepetitions() >= 2) game_results[node] = lczero::GameResult::DRAW;
    }

    // Make game simpler to test NN training:
    // if (game_results[node] == lczero::GameResult::UNDECIDED && position.GetGamePly() >= 48) {
    //     int ours = board.ours().count();
    //     int theirs = board.theirs().count();

    //     if (ours == theirs) game_results[node] = lczero::GameResult::UNDECIDED;
    //     else if (position.IsBlackToMove())
    //         game_results[node] = ours > theirs ? lczero::GameResult::BLACK_WON : lczero::GameResult::WHITE_WON;
    //     else
    //         game_results[node] = ours > theirs ? lczero::GameResult::WHITE_WON : lczero::GameResult::BLACK_WON;
    // }
};


int PositionHistoryTree::Move(const int node_, const lczero::Move move_) {
    assert(node_ < (int)positions.size());
    int node = node_ >= 0 ? node_ : (positions.size() - 1);

    auto move = positions[node].GetBoard().GetModernMove(move_);

    int new_node = (int)positions.size();

    positions.push_back(lczero::Position(positions[node], move));
    hashes.push_back(positions.back().GetBoard().Hash());
    parents.push_back(node);
    node_valid_moves.push_back({});
    game_results.push_back(lczero::GameResult::UNDECIDED);

    int cycle_length;
    int repetitions = ComputeLastMoveRepetitions(new_node, &cycle_length);
    positions.back().SetRepetitions(repetitions, cycle_length);

    ComputeNodeGameResult(new_node);

    return new_node;
}


void PositionHistoryTree::PopLast() {
    assert(positions.size() > 0);
    int removed = positions.size() - 1;
    int parent = parents[removed];

    positions.pop_back();
    hashes.pop_back();
    parents.pop_back();
    node_valid_moves.pop_back();
    game_results.pop_back();
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