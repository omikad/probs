#include "infra/uci_player.h"

using namespace std;

namespace probs {


int UciPlayer::AppendLastTreeNode() {
    int tree_node_idx = (int)tree.positions.size() - 1;
    NodeInfo node;
    for (auto move : tree.node_valid_moves[tree_node_idx]) {
        KidInfo kid_info;
        kid_info.kid_node = -1;
        kid_info.move = move;
        kid_info.q_nn_score = 0;
        kid_info.q_tree_score = 0;
        node.kids.push_back(kid_info);
    }
    if (node.kids.size() == 0) {
        node.is_terminal = true;
        auto score = tree.GetRelativePositionScore(tree_node_idx);
        node.v_tree_score = score.has_value() ? 0 : score.value();
    }
    else {
        node.is_terminal = false;
        node.v_tree_score = 0;
    }
    nodes.push_back(node);
    assert(nodes.size() == tree.positions.size());
    return tree_node_idx;
}


UciPlayer::UciPlayer(ConfigParser& config)
        : n_max_episode_steps(config.GetInt("env.n_max_episode_steps", true, 0))
        , tree(lczero::ChessBoard::kStartposFen, n_max_episode_steps)
        , is_in_search(false)
        , stop_search_flag(false)
        , debug_on(false) {
    top_node = AppendLastTreeNode();
}


void UciPlayer::OnNewGame() {
    assert(is_in_search == false);
}


int UciPlayer::FindKidIndex(const int node, const lczero::Move move) const {
    for (int ki = 0; ki < (int)nodes[node].kids.size(); ki++)
        if (nodes[node].kids[ki].move == move)
            return ki;
    return -1;
}


void UciPlayer::SetPosition(const string& starting_fen, const vector<string>& moves) {
    assert(is_in_search == false);

    bool continuation = false;
    if (last_pos_fen == starting_fen && last_pos_moves.size() < moves.size()) {
        bool eq = true;
        for (int i = 0; i < (int)last_pos_moves.size(); i++)
            if (moves[i] != last_pos_moves[i]) {
                eq = false;
                break;
            }

        if (eq)
            continuation = true;
    }

    if (!continuation) {
        last_pos_fen = starting_fen;
        last_pos_moves.clear();
        nodes.clear();
        tree = PositionHistoryTree(starting_fen, n_max_episode_steps);
        top_node = AppendLastTreeNode();
    }

    for (int i = last_pos_moves.size(); i < (int)moves.size(); i++) {
        assert(nodes[top_node].is_terminal == false);

        auto& move_str = moves[i];
        auto move = lczero::Move(move_str, tree.positions[top_node].IsBlackToMove());
        move = tree.positions[top_node].GetBoard().GetModernMove(move);

        int ki = FindKidIndex(top_node, move);
        // if (ki < 0) {
        //     cerr << move_str << endl;
        //     cerr << "tree:"; for (auto& kid_move : tree.node_valid_moves[top_node]) cerr << kid_move.as_string(); cerr << endl;
        //     cerr << "kids:"; for (auto& kid : nodes[top_node].kids) cerr << kid.move.as_string(); cerr << endl;
        // }
        assert(ki >= 0);

        int new_top_node;
        if (nodes[top_node].kids[ki].kid_node < 0) {
            tree.Move(top_node, move);
            new_top_node = AppendLastTreeNode();
            nodes[top_node].kids[ki].kid_node = new_top_node;
        }
        else
            new_top_node = nodes[top_node].kids[ki].kid_node;

        top_node = new_top_node;
        last_pos_moves.push_back(move_str);
    }

    if (debug_on)
        cerr << "[DEBUG] Last position:\n" << tree.positions[top_node].DebugString() << endl;
}


void UciPlayer::Stop() {
    stop_search_flag = true;
}


void UciPlayer::StartSearch(
        optional<chrono::milliseconds> search_wtime,
        optional<chrono::milliseconds> search_btime,
        optional<chrono::milliseconds> search_winc,
        optional<chrono::milliseconds> search_binc,
        optional<int> search_moves_to_go,
        optional<int> search_depth,
        optional<uint64_t> search_nodes,
        optional<int> search_mate,
        optional<chrono::milliseconds> search_fixed_time,
        bool search_infinite,
        vector<string>& search_moves
    ) {
    is_in_search = true;
    stop_search_flag = false;

    if (debug_on) {
        cerr << "Start search with params:" << endl;
        cerr << "search_wtime=" << (search_wtime.has_value() ? to_string(search_wtime.value().count()) : "null") << endl;
        cerr << "search_btime=" << (search_btime.has_value() ? to_string(search_btime.value().count()) : "null") << endl;
        cerr << "search_winc=" << (search_winc.has_value() ? to_string(search_winc.value().count()) : "null") << endl;
        cerr << "search_binc=" << (search_binc.has_value() ? to_string(search_binc.value().count()) : "null") << endl;
        cerr << "search_moves_to_go=" << (search_moves_to_go.has_value() ? to_string(search_moves_to_go.value()) : "null") << endl;
        cerr << "search_depth=" << (search_depth.has_value() ? to_string(search_depth.value()) : "null") << endl;
        cerr << "search_nodes=" << (search_nodes.has_value() ? to_string(search_nodes.value()) : "null") << endl;
        cerr << "search_mate=" << (search_mate.has_value() ? to_string(search_mate.value()) : "null") << endl;
        cerr << "search_fixed_time=" << (search_fixed_time.has_value() ? to_string(search_fixed_time.value().count()) : "null") << endl;
        cerr << "search_infinite=" << (search_infinite ? "true" : "false") << endl;
        cerr << "search_moves=["; for (auto& move : search_moves) cerr << " " << move; cerr << " ]" << endl;
    }

    assert(!nodes[top_node].is_terminal);

    int ki = rand() % nodes[top_node].kids.size();

    auto move = nodes[top_node].kids[ki].move;

    auto& top_position = tree.positions[top_node];
    move = top_position.GetBoard().GetLegacyMove(move);
    if (top_position.IsBlackToMove())
        move.Mirror();

    cout << "bestmove " << move.as_string() << "\n";
    cout << flush;

    is_in_search = false;
}


void UciPlayer::WaitForReadyState() {
    if (search_future.valid()) {
        search_future.get();
    }
}


void UciPlayer::SetDebug(bool turn_on) {
    debug_on = turn_on;
}


}  // namespace probs
