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


namespace probs {

struct EncodedPositionBatch {
    std::vector<int> transforms;
    std::vector<lczero::InputPlanes> planes;
    std::vector<std::vector<std::pair<lczero::Move, float>>> moves_estimation;

    lczero::Move FindBestMove(const int bi) const;
    std::vector<lczero::Move> const FindBestMoves() const;
};


lczero::InputPlanes Encode(const lczero::PositionHistory& lchistory, int* transform_out);


std::shared_ptr<EncodedPositionBatch> GetQModelEstimation(
    const std::vector<PositionHistoryTree*>& trees,
    const std::vector<int>& nodes,
    ResNet q_model,
    const at::Device& device);

}  // namespace probs
