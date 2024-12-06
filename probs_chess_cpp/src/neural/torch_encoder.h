#pragma once
#include <ATen/Device.h>
#include <torch/torch.h>
#include <vector>
#include <utility>
#include <memory>

#include "chess/bitboard.h"
#include "chess/position.h"
#include "chess/game_tree.h"
#include "chess/policy_map.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "training/training_helpers.h"


namespace probs {

struct EncodedPositionBatch {
    std::vector<int> transforms;
    std::vector<lczero::InputPlanes> planes;
    std::vector<std::vector<MoveEstimation>> moves_estimation;

    lczero::Move FindBestMove(const int bi) const;
    std::vector<lczero::Move> const FindBestMoves() const;
};

lczero::InputPlanes Encode(const PositionHistoryTree& history_tree, const std::vector<int>& history_nodes, int* transform_out);

lczero::InputPlanes Encode(const PositionHistoryTree& history_tree, const int last_node, int* transform_out);

void FillInputTensor(torch::TensorAccessor<float, 4>& input_accessor, const int batch_index, const lczero::InputPlanes& input_planes);

std::shared_ptr<EncodedPositionBatch> GetQModelEstimation(const std::vector<PositionHistoryTree*>& trees, const std::vector<int>& nodes, ResNet q_model, const at::Device& device);

template<typename TNode>
void GetQModelEstimation_Nodes(std::vector<TNode*>& result, ResNet q_model, const at::Device& device) {
    int batch_size = result.size();

    torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});
    auto input_accessor = input.accessor<float, 4>();

    for (int bi = 0; bi < batch_size; bi++)
        FillInputTensor(input_accessor, bi, result[bi]->input_planes);

    input = input.to(device);
    torch::Tensor q_values = q_model->forward(input);
    q_values = q_values.to(torch::kCPU);
    auto q_values_accessor = q_values.accessor<float, 4>();


    for (int bi = 0; bi < batch_size; bi++) {
        for (int mi = 0; mi < result[bi]->moves_estimation.size(); mi++) {
            auto move = result[bi]->moves_estimation[mi].move;
            int move_idx = move.as_nn_index(result[bi]->transform);
            int policy_idx = move_to_policy_idx_map[move_idx];
            int displacement = policy_idx / 64;
            int square = policy_idx % 64;
            int row = square / 8;
            int col = square % 8;
            float score = q_values_accessor[bi][displacement][row][col];

            result[bi]->moves_estimation[mi].score = score;
        }
    }

    // Using gather - a bit worse:
    // torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});
    // auto input_accessor = input.accessor<float, 4>();
    // for (int bi = 0; bi < batch_size; bi++)
    //     FillInputTensor(input_accessor, bi, result[bi]->input_planes);

    // const int MAX_NUM_POSSIBLE_MOVES = 128;

    // torch::Tensor moves_idx = torch::zeros({batch_size, MAX_NUM_POSSIBLE_MOVES}, torch::kInt64);
    // auto moves_idx_accessor = moves_idx.accessor<long, 2>();
    // for (int bi = 0; bi < batch_size; bi++) {
    //     for (int mi = 0; mi < std::min((int)result[bi]->moves_estimation.size(), MAX_NUM_POSSIBLE_MOVES); mi++) {
    //         auto move = result[bi]->moves_estimation[mi].move;
    //         int move_idx = move.as_nn_index(result[bi]->transform);
    //         int policy_idx = move_to_policy_idx_map[move_idx];

    //         moves_idx_accessor[bi][mi] = policy_idx;
    //     }
    // }

    // input = input.to(device);
    // moves_idx = moves_idx.to(device);
    // torch::Tensor q_values = q_model->forward(input);
    // q_values = q_values.view({batch_size, -1});
    // q_values = torch::gather(q_values, 1, moves_idx);

    // q_values = q_values.to(torch::kCPU);

    // auto q_values_accessor = q_values.accessor<float, 2>();
    // for (int bi = 0; bi < batch_size; bi++)
    //     for (int mi = 0; mi < std::min((int)result[bi]->moves_estimation.size(), MAX_NUM_POSSIBLE_MOVES); mi++) {
    //         float q_val = q_values_accessor[bi][mi];
    //         result[bi]->moves_estimation[mi].score = q_val;
    //     }
}

}  // namespace probs
