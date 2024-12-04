#include "infra/player.h"

using namespace std;


namespace probs {

vector<lczero::Move> RandomPlayer::GetActions(vector<PositionHistoryTree*>& history) {
    vector<lczero::Move> picked_moves(history.size());

    for(int hi = 0; hi < history.size(); hi++) {
        const auto& board = history[hi]->LastPosition().GetBoard();

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

        for (auto move : tree.node_valid_moves[node]) {

            int kid_node = tree.Move(node, move);

            lczero::GameResult game_result = tree.GetGameResult(kid_node);

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


VResnetPlayer::VResnetPlayer(ModelKeeper& model_keeper, const ConfigParser& config_parser, const string& config_key_prefix, const string& name):
        name(name),
        device(torch::kCPU),
        v_model(config_parser, config_key_prefix + ".model.v", true) {

    int gpu_num = config_parser.GetInt("infra.gpu");
    cout << "VResnetPlayer GPU: " << gpu_num << endl;
    if (gpu_num >= 0) {
        if (torch::cuda::is_available())
            device = at::Device("cuda:" + to_string(gpu_num));
        else
            throw Exception("Config points to GPU which is not available (config parameter infra.gpu)");
        v_model->to(device);
    }

    cout << DebugString(*v_model) << endl;
}


vector<lczero::Move> VResnetPlayer::GetActions(vector<PositionHistoryTree*>& history) {
    vector<lczero::Move> picked_moves;

    for (int bi = 0; bi < history.size(); bi++) {
        vector<lczero::Move> moves;
        vector<lczero::InputPlanes> input_planes;

        int last_node_idx = history[bi]->LastIndex();

        for (auto move: history[bi]->node_valid_moves[last_node_idx]) {
            moves.push_back(move);

            int new_node = history[bi]->Move(last_node_idx, move);

            int transform_out;
            input_planes.push_back(Encode(*history[bi], new_node, &transform_out));

            history[bi]->PopLast();
        }
        assert(moves.size() > 0);
        assert(history[bi]->LastIndex() == last_node_idx);

        torch::Tensor input = torch::zeros({(int)moves.size(), lczero::kInputPlanes, 8, 8});
        for (int mi = 0; mi < moves.size(); mi++)
            for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                const auto& plane = input_planes[mi][pi];
                for (auto bit : lczero::IterateBits(plane.mask)) {
                    input[mi][pi][bit / 8][bit % 8] = plane.value;
                }
            }

        input = input.to(device);

        torch::Tensor predictions = v_model->forward(input);
        predictions = predictions.contiguous();
        vector<float> values(predictions.data_ptr<float>(), predictions.data_ptr<float>() + predictions.numel());
        assert(values.size() == moves.size());

        // My best move leads to the lowest evaluation of the next state for the next player
        int best_i = 0;
        for (int i = 1; i < values.size(); i++)
            if (values[i] < values[best_i])
                best_i = i;
        
        picked_moves.push_back(moves[best_i]);
    }

    return picked_moves;
}


QResnetPlayer::QResnetPlayer(ModelKeeper& model_keeper, const ConfigParser& config_parser, const string& config_key_prefix, const string& name):
        name(name),
        device(torch::kCPU),
        q_model(config_parser, config_key_prefix + ".model.q", false) {

    int gpu_num = config_parser.GetInt("infra.gpu");
    cout << "QResnetPlayer GPU: " << gpu_num << endl;
    if (gpu_num >= 0) {
        if (torch::cuda::is_available())
            device = at::Device("cuda:" + to_string(gpu_num));
        else
            throw Exception("Config points to GPU which is not available (config parameter infra.gpu)");
        q_model->to(device);
    }

    cout << DebugString(*q_model) << endl;
}


vector<lczero::Move> QResnetPlayer::GetActions(vector<PositionHistoryTree*>& history) {
    int batch_size = history.size();

    vector<int> nodes(batch_size);
    for (int bi = 0; bi < batch_size; bi++) nodes[bi] = history[bi]->LastIndex();

    auto encoded_batch = GetQModelEstimation(history, nodes, q_model, device);

    return encoded_batch->FindBestMoves();
}


}  // namespace probs
