/*
    Adapted from Leela Chess Zero: https://github.com/LeelaChessZero/lc0/
*/

#pragma once

#include <string>
#include <optional>
#include <memory>

#include "chess/board.h"
#include "chess/position.h"
#include "utils/exception.h"


namespace lczero {
namespace python {

class GameState {
 public:
  GameState(const std::optional<std::string> startpos,
            const std::vector<std::string>& moves) {
    ChessBoard starting_board;
    int no_capture_ply;
    int full_moves;
    starting_board.SetFromFen(startpos.value_or(ChessBoard::kStartposFen),
                              &no_capture_ply, &full_moves);

    history_.Reset(starting_board, no_capture_ply,
                   full_moves * 2 - (starting_board.flipped() ? 1 : 2));

    for (const auto& m : moves) {
      Move move(m, history_.IsBlackToMove());
      move = history_.Last().GetBoard().GetModernMove(move);
      history_.Append(move);
    }
  }

  std::vector<std::string> moves() const {
    auto ms = history_.Last().GetBoard().GenerateLegalMoves();
    bool is_black = history_.IsBlackToMove();
    std::vector<std::string> result;
    for (auto m : ms) {
      if (is_black) m.Mirror();
      result.push_back(m.as_string());
    }
    return result;
  }

  std::vector<int> policy_indices() const {
    auto ms = history_.Last().GetBoard().GenerateLegalMoves();
    std::vector<int> result;
    for (auto m : ms) {
      result.push_back(m.as_nn_index(/* transform= */ 0));
    }
    return result;
  }

  std::string as_string() const {
    bool is_black = history_.IsBlackToMove();
    return (is_black ? history_.Last().GetThemBoard()
                     : history_.Last().GetBoard())
        .DebugString();
  }

 private:
  PositionHistory history_;
};

}  // namespace python
}  // namespace lczero