#include "infra/uci_player.h"

using namespace std;

namespace probs {

UciPlayer::UciPlayer(ConfigParser& config)
    : n_max_episode_steps(config.GetInt("env.n_max_episode_steps", true, 0))
    , tree(lczero::ChessBoard::kStartposFen, n_max_episode_steps)
    , is_in_search(false)
    , stop_search_flag(false)
    , debug_on(false) {
}


void UciPlayer::onNewGame() {
    assert(is_in_search == false);
}


void UciPlayer::setPosition(const string& starting_fen, const vector<string>& moves) {
    assert(is_in_search == false);
    tree = PositionHistoryTree(starting_fen, n_max_episode_steps);
    for (auto& move : moves)
        tree.Move(-1, lczero::Move(move, tree.LastPosition().IsBlackToMove()));
    if (debug_on)
        cerr << "current position debug:\n" << tree.LastPosition().DebugString() << endl;
}


void UciPlayer::stop() {
    stop_search_flag = true;
}


void UciPlayer::startSearch(
        bool is_reading_moves,
        optional<chrono::milliseconds> wtime,
        optional<chrono::milliseconds> btime,
        optional<chrono::milliseconds> winc,
        optional<chrono::milliseconds> binc,
        optional<int> moves_to_go,
        optional<int> depth,
        optional<uint64_t> nodes,
        optional<int> mate,
        optional<chrono::milliseconds> fixed_time,
        bool infinite,
        vector<string>& search_moves
    ) {
    is_in_search = true;
    stop_search_flag = false;

    if (debug_on) {
        cerr << "Start search with params:" << endl;
        cerr << "is_reading_moves=" << (is_reading_moves ? "true" : "false") << endl;
        cerr << "wtime=" << (wtime.has_value() ? to_string(wtime.value().count()) : "null") << endl;
        cerr << "btime=" << (btime.has_value() ? to_string(btime.value().count()) : "null") << endl;
        cerr << "winc=" << (winc.has_value() ? to_string(winc.value().count()) : "null") << endl;
        cerr << "binc=" << (binc.has_value() ? to_string(binc.value().count()) : "null") << endl;
        cerr << "moves_to_go=" << (moves_to_go.has_value() ? to_string(moves_to_go.value()) : "null") << endl;
        cerr << "depth=" << (depth.has_value() ? to_string(depth.value()) : "null") << endl;
        cerr << "nodes=" << (nodes.has_value() ? to_string(nodes.value()) : "null") << endl;
        cerr << "mate=" << (mate.has_value() ? to_string(mate.value()) : "null") << endl;
        cerr << "fixed_time=" << (fixed_time.has_value() ? to_string(fixed_time.value().count()) : "null") << endl;
        cerr << "infinite=" << (infinite ? "true" : "false") << endl;
        cerr << "nodes=" << (nodes.has_value() ? to_string(nodes.value()) : "null") << endl;
        cerr << "search_moves=["; for (auto& move : search_moves) cerr << " " << move; cerr << " ]" << endl;
    }

    is_in_search = false;
}


void UciPlayer::waitForReadyState() {
    if (search_future.valid()) {
        search_future.get();
    }
}


void UciPlayer::setDebug(bool turn_on) {
    debug_on = turn_on;
}


}  // namespace probs
