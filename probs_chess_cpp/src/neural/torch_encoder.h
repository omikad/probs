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

template<typename TNode>
void GetQModelEstimation_Nodes(std::vector<TNode>& result, ResNet q_model, const at::Device& device);

lczero::InputPlanes Encode(const lczero::PositionHistory& lchistory, int* transform_out);

std::shared_ptr<EncodedPositionBatch> GetQModelEstimation(const std::vector<PositionHistoryTree*>& trees, const std::vector<int>& nodes, ResNet q_model, const at::Device& device);

std::shared_ptr<EncodedPositionBatch> GetQModelEstimation_OneNode(PositionHistoryTree& tree, const int node, ResNet q_model, const at::Device& device);

}  // namespace probs
