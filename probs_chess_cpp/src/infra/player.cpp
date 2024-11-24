#include <iostream>
#include <vector>
#include <torch/torch.h>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/player.h"
#include "utils/exception.h"
#include "neural/encoder.h"
#include "utils/torch_utils.h"

using namespace std;


namespace probs {

vector<lczero::Move> RandomPlayer::GetActions(const vector<lczero::PositionHistory>& history) {
    vector<lczero::Move> picked_moves(history.size());

    for(int hi = 0; hi < history.size(); hi++) {
        const auto& board = history[hi].Last().GetBoard();

        auto legal_moves = board.GenerateLegalMoves();
        if (legal_moves.size() == 0)
            throw Exception("No legal moves found");

        int mi = rand() % (legal_moves.size());

        picked_moves[hi] = legal_moves[mi];
    }

    return picked_moves;
}


VQResnetPlayer::VQResnetPlayer(const ConfigParser& config_parser, const string& config_key_prefix, const string& name):
        name(name),
        v_model(config_parser, config_key_prefix + ".model.v"),
        q_model(config_parser, config_key_prefix + ".model.q") {}


vector<lczero::Move> VQResnetPlayer::GetActions(const vector<lczero::PositionHistory>& history) {
    vector<lczero::Move> picked_moves(history.size());

    int batch_size = (int)history.size();

    torch::Tensor input_tensor = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});

    for (int hi = 0; hi < batch_size; hi++) {
        int transform_out;

        lczero::InputPlanes input_planes = lczero::EncodePositionForNN(
            lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
            history[hi],
            8,
            lczero::FillEmptyHistory::FEN_ONLY,
            &transform_out);

        assert(input_planes.size() == lczero::kInputPlanes);

        for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
            const auto& plane = input_planes[pi];
            for (auto bit : lczero::IterateBits(plane.mask)) {
                input_tensor[hi][pi][bit / 8][bit % 8] = plane.value;
            }
        }
    }

    cout << "Input tensor: " << DebugString(input_tensor);

    torch::Tensor q_values = q_model.forward(input_tensor);

    cout << "Q values tensor: " << DebugString(q_values);

    throw Exception("TODO");
}


}  // namespace probs
