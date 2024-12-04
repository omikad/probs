/*
  Adapted from https://github.com/LeelaChessZero/lc0
*/

#pragma once

#include <vector>

#include "chess/position.h"
#include "chess/game_tree.h"
#include "utils/exception.h"


namespace lczero {

constexpr int kNumOutputPolicyCnt = 1858;
constexpr int kNumOutputPolicyFilters = 73;
constexpr int kInputPlanes = 112;
constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;


// All input planes are 64 value vectors, every element of which is either
// 0 or some value, unique for the plane. Therefore, input is defined as
// a bitmask showing where to set the value, and the value itself.
struct InputPlane {
  InputPlane() = default;
  void SetAll() { mask = ~0ull; }
  void Fill(float val) {
    SetAll();
    value = val;
  }
  std::uint64_t mask = 0ull;
  float value = 1.0f;
};
using InputPlanes = std::vector<InputPlane>;


enum InputFormat {
    INPUT_CLASSICAL_112_PLANE,
    INPUT_112_WITH_CASTLING_PLANE,
    INPUT_112_WITH_CANONICALIZATION,
    INPUT_112_WITH_CANONICALIZATION_V2,
    INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON,
    INPUT_112_WITH_CANONICALIZATION_HECTOPLIES,
    INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
};

enum FillEmptyHistory {
    NO,
    FEN_ONLY,
    ALWAYS
};

// Returns the transform that would be used in EncodePositionForNN.
int TransformForPosition(InputFormat input_format, const PositionHistory& history);

// Encodes the last position in history for the neural network request.
InputPlanes EncodePositionForNN(
    InputFormat input_format,
    const probs::PositionHistoryTree& history_tree,
    const std::vector<int>& history_nodes,
    int history_planes,
    FillEmptyHistory fill_empty_history,
    int* transform_out);

bool IsCanonicalFormat(InputFormat input_format);
bool IsCanonicalArmageddonFormat(InputFormat input_format);
bool IsHectopliesFormat(InputFormat input_format);
bool Is960CastlingFormat(InputFormat input_format);

}  // namespace lczero