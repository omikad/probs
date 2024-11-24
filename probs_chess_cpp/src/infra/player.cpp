#include <iostream>
#include <vector>
#include <ATen/Device.h>
#include <torch/torch.h>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/player.h"
#include "utils/exception.h"
#include "neural/encoder.h"
#include "utils/torch_utils.h"
#include "chess/policy_map.h"

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
        device(torch::kCPU),
        v_model(config_parser, config_key_prefix + ".model.v"),
        q_model(config_parser, config_key_prefix + ".model.q") {

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


vector<lczero::Move> VQResnetPlayer::GetActions(const vector<lczero::PositionHistory>& history) {
    int batch_size = (int)history.size();

    vector<lczero::Move> picked_moves(batch_size);
    vector<int> transforms(batch_size);

    torch::Tensor input_tensor = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});

    for (int hi = 0; hi < batch_size; hi++) {
        lczero::InputPlanes input_planes = lczero::EncodePositionForNN(
            lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
            history[hi],
            8,
            lczero::FillEmptyHistory::FEN_ONLY,
            &transforms[hi]);

        assert(input_planes.size() == lczero::kInputPlanes);

        for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
            const auto& plane = input_planes[pi];
            for (auto bit : lczero::IterateBits(plane.mask)) {
                input_tensor[hi][pi][bit / 8][bit % 8] = plane.value;
            }
        }
    }

    input_tensor = input_tensor.to(device);
    torch::Tensor q_values = q_model.forward(input_tensor);

    for (int hi = 0; hi < batch_size; hi++) {
        float best_score = -1000000;
        lczero::Move best_move;

        for (auto& move: history[hi].Last().GetBoard().GenerateLegalMoves()) {
            int move_idx = move.as_nn_index(transforms[hi]);
            int policy_idx = move_to_policy_idx_map[move_idx];
            int displacement = policy_idx / 64;
            int square = policy_idx % 64;
            int row = square / 8;
            int col = square % 8;
            float score = q_values[hi][displacement][row][col].item<float>();
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }

        picked_moves[hi] = best_move;        
    }

    return picked_moves;
}


}  // namespace probs
