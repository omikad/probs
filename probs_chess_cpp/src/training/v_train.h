#pragma once
#include <vector>
#include <utility>
#include <ATen/Device.h>

#include "chess/position.h"
#include "infra/config_parser.h"
#include "infra/env_player.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "neural/torch_encoder.h"


namespace probs {

std::vector<std::pair<lczero::InputPlanes, float>> SelfPlay(const ConfigParser& config_parser, const int n_games);

}  // namespace probs
