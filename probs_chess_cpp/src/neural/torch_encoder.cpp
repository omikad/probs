#include "neural/torch_encoder.h"

using namespace std;


namespace probs {


lczero::Move EncodedPositionBatch::FindBestMove(const int bi) const {
    float best_score = -1000000;
    lczero::Move best_move;

    for (auto& move_and_score : moves_estimation[bi]) {
        float score = move_and_score.score;
        if (score > best_score) {
            best_score = score;
            best_move = move_and_score.move;
        }
    }
    return best_move;
}


vector<lczero::Move> const EncodedPositionBatch::FindBestMoves() const {
    vector<lczero::Move> result(transforms.size());
    for (int bi = 0; bi < transforms.size(); bi++)
        result[bi] = FindBestMove(bi);
    return result;
}


lczero::InputPlanes Encode(const PositionHistoryTree& history_tree, const vector<int>& history_nodes, int* transform_out) {
    return lczero::EncodePositionForNN(
        lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
        history_tree,
        history_nodes,
        8,
        lczero::FillEmptyHistory::FEN_ONLY,
        transform_out);
}


lczero::InputPlanes Encode(const PositionHistoryTree& history_tree, const int last_node, int* transform_out) {
    vector<int> history_nodes = history_tree.GetHistoryPathNodes(last_node);

    return lczero::EncodePositionForNN(
        lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
        history_tree,
        history_nodes,
        8,
        lczero::FillEmptyHistory::FEN_ONLY,
        transform_out);
}


void FillInputTensor(torch::TensorAccessor<float, 4>& input_accessor, const int batch_index, const lczero::InputPlanes& input_planes) {
    for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
        const auto& plane = input_planes[pi];
        for (auto bit : lczero::IterateBits(plane.mask))
            input_accessor[batch_index][pi][bit / 8][bit % 8] = plane.value;
    }
}


shared_ptr<EncodedPositionBatch> GetQModelEstimation(const vector<PositionHistoryTree*>& trees, const vector<int>& nodes, ResNet q_model, const at::Device& device) {
    assert(trees.size() == nodes.size());
    int batch_size = nodes.size();

    EncodedPositionBatch result;
    torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});
    auto input_accessor = input.accessor<float, 4>();

    for (int bi = 0; bi < batch_size; bi++) {
        int transform_out;

        result.planes.push_back(Encode(*trees[bi], nodes[bi], &transform_out));

        result.transforms.push_back(transform_out);

        FillInputTensor(input_accessor, bi, result.planes.back());
    }

    input = input.to(device);
    torch::Tensor q_values = q_model->forward(input);
    q_values = q_values.to(torch::kCPU);

    result.moves_estimation.resize(batch_size);

    for (int bi = 0; bi < batch_size; bi++) {
        for (auto move: trees[bi]->positions[nodes[bi]].GetBoard().GenerateLegalMoves()) {
            int move_idx = move.as_nn_index(result.transforms[bi]);
            int policy_idx = move_to_policy_idx_map[move_idx];
            int displacement = policy_idx / 64;
            int square = policy_idx % 64;
            int row = square / 8;
            int col = square % 8;
            float score = q_values[bi][displacement][row][col].item<float>();

            result.moves_estimation[bi].push_back({move, score});
        }
    }

    return make_shared<EncodedPositionBatch>(result);
}


float GetVScoreOnStartingBoard(ResNet v_model, const at::Device& device) {
    auto tree = PositionHistoryTree(lczero::ChessBoard::kStartposFen, 100);

    vector<int> nodes{0};
    int transform_out;
    auto input_planes = Encode(tree, nodes, &transform_out);

    torch::Tensor input = torch::zeros({1, lczero::kInputPlanes, 8, 8});
    auto input_accessor = input.accessor<float, 4>();

    FillInputTensor(input_accessor, 0, input_planes);

    input = input.to(device);
    torch::Tensor prediction = v_model->forward(input);
    prediction = prediction.to(torch::kCPU);

    float score = prediction[0].item<float>();
    return score;
}


}  // namespace probs
