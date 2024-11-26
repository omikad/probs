#include "infra/player.h"

using namespace std;


namespace probs {

vector<lczero::Move> RandomPlayer::GetActions(vector<PositionHistoryTree*>& history) {
    vector<lczero::Move> picked_moves(history.size());

    for(int hi = 0; hi < history.size(); hi++) {
        const auto& board = history[hi]->Last().GetBoard();

        auto legal_moves = board.GenerateLegalMoves();
        if (legal_moves.size() == 0)
            throw Exception("No legal moves found");

        int mi = rand() % legal_moves.size();

        picked_moves[hi] = legal_moves[mi];
    }

    return picked_moves;
}


vector<lczero::Move> NStepLookaheadPlayer::GetActions(vector<PositionHistoryTree*>& history) {
    int max_depth = this->depth;

    auto dfs = [&](auto& self, PositionHistoryTree& tree, const int node, const int depth)->vector<pair<lczero::Move, int>> {
        vector<pair<lczero::Move, int>> result;

        bool is_black = tree.positions[node].IsBlackToMove();

        for (auto& move : tree.positions[node].GetBoard().GenerateLegalMoves()) {

            int kid_node = tree.Append(node, move);

            lczero::GameResult game_result = tree.ComputeGameResult(kid_node);

            if (game_result != lczero::GameResult::UNDECIDED) {
                int score = (game_result == lczero::GameResult::DRAW) ? 0
                    : (is_black == (game_result == lczero::GameResult::BLACK_WON)) ? 1
                    : -1;

                result.push_back({move, score});
            }
            else if (depth == max_depth) {
                result.push_back({move, 0});
            }
            else {
                int maxval = -10;
                for (auto& move_and_score : self(self, tree, kid_node, depth + 1))
                    maxval = max(maxval, move_and_score.second);
                result.push_back({move, -maxval});
            }
        }

        return result;
    };

    vector<lczero::Move> picked_moves(history.size());

    for (int hi = 0; hi < history.size(); hi++) {
        auto tree = history[hi];
        int tree_size_orig = tree->positions.size();

        auto top_values_and_actions = dfs(dfs, *tree, tree->LastIndex(), 1);

        vector<lczero::Move> best_moves;
        int best_score = -10;
        for (auto& item : top_values_and_actions) {
            int score = item.second;
            if (score > best_score) {
                best_moves.clear();
                best_moves.push_back(item.first);
                best_score = score;
            }
            else if (score == best_score)
                best_moves.push_back(item.first);
        }

        picked_moves[hi] = best_moves[rand() % best_moves.size()];

        while (tree->positions.size() > tree_size_orig)
            tree->PopLast();
    }

    return picked_moves;
}


VQResnetPlayer::VQResnetPlayer(const ConfigParser& config_parser, const string& config_key_prefix, const string& name):
        name(name),
        device(torch::kCPU),
        v_model(config_parser, config_key_prefix + ".model.v", true),
        q_model(config_parser, config_key_prefix + ".model.q", false) {

    int gpu_num = config_parser.GetInt("infra.gpu");
    cout << "VQResnetPlayer GPU: " << gpu_num << endl;
    if (gpu_num >= 0) {
        if (torch::cuda::is_available())
            device = at::Device("cuda:" + to_string(gpu_num));
        else
            throw Exception("Config points to GPU which is not available (config parameter infra.gpu)");
        v_model.to(device);
        q_model.to(device);
    }

    cout << DebugString(v_model) << endl;
    cout << DebugString(q_model) << endl;
}


vector<lczero::Move> VQResnetPlayer::GetActions(vector<PositionHistoryTree*>& history) {
    int batch_size = history.size();

    vector<int> nodes(batch_size);
    for (int bi = 0; bi < batch_size; bi++) nodes[bi] = history[bi]->LastIndex();

    auto encoded_batch = GetQModelEstimation(history, nodes, q_model, device);

    return encoded_batch->FindBestMoves();
}


}  // namespace probs
