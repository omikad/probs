#include "neural/torch_encoder.h"

using namespace std;


namespace probs {


lczero::Move EncodedPositionBatch::FindBestMove(const int bi) const {
    float best_score = -1000000;
    lczero::Move best_move;

    for (auto& move_and_score : moves_estimation[bi]) {
        float score = move_and_score.second;
        if (score > best_score) {
            best_score = score;
            best_move = move_and_score.first;
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


shared_ptr<EncodedPositionBatch> GetQModelEstimation(
        const vector<PositionHistoryTree*>& trees,
        const vector<int>& nodes,
        ResNet q_model,
        at::Device& device) {
    assert(trees.size() == nodes.size());
    int batch_size = nodes.size();

    EncodedPositionBatch result;
    result.tensor = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});

    for (int bi = 0; bi < batch_size; bi++) {
        lczero::PositionHistory lchistory = trees[bi]->ToLczeroHistory(nodes[bi]);

        int transform_out;

        result.planes.push_back(
            lczero::EncodePositionForNN(
                lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
                lchistory,
                8,
                lczero::FillEmptyHistory::FEN_ONLY,
                &transform_out)
        );

        result.transforms.push_back(transform_out);

        for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
            const auto& plane = result.planes.back()[pi];
            for (auto bit : lczero::IterateBits(plane.mask)) {
                result.tensor[bi][pi][bit / 8][bit % 8] = plane.value;
            }
        }
    }

    auto inp = result.tensor.to(device);
    torch::Tensor q_values = q_model->forward(inp);

    result.moves_estimation.resize(batch_size);

    for (int bi = 0; bi < batch_size; bi++) {
        for (auto& move: trees[bi]->positions[nodes[bi]].GetBoard().GenerateLegalMoves()) {
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

}  // namespace probs
