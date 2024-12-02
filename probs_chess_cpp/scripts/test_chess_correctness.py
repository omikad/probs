import traceback
import chess
import random
import os
import sys


def show_moves_difference(chess_legal_moves, probs_legal_moves):
    chess_str = ' '.join(chess_legal_moves)
    probs_str = ' '.join(probs_legal_moves)

    i = 0
    while i < len(chess_str) and i < len(probs_str) and chess_str[i] == probs_str[i]:
        i += 1
    print(f"Chess: {chess_str}")
    print(f"Probs: {probs_str}")
    print('=' * (7 + i) + '^')


if __name__ == "__main__":
    random.seed(111)
    GAMES = 100000
    STEPS = 500

    # 100K games by 500 steps: 25m35,742s

    lib_probs_chess_path = os.environ['LIB_PROBS_CHESS']
    assert lib_probs_chess_path is not None and len(os.environ['LIB_PROBS_CHESS']) > 1

    sys.path.append(lib_probs_chess_path)
    from libprobs_chess import ChessEnv

    for game_i in range(GAMES):

        try:
            chess_env = chess.Board()
            probs_env = ChessEnv(max_ply=STEPS)

            for step_i in range(STEPS + 1):
                if step_i == STEPS:
                    assert probs_env.game_state() == 'draw', f"Game {game_i}, step {step_i}, expected game finished by max ply constraint"
                    break
                
                probs_legal_moves = probs_env.legal_moves()
                probs_legal_moves.sort()

                chess_legal_moves = [str(m) for m in chess_env.legal_moves]
                chess_legal_moves.sort()

                ok = len(probs_legal_moves) == len(chess_legal_moves) and all(pm == cm for pm, cm in zip(probs_legal_moves, chess_legal_moves))
                if not ok:
                    print(f"Game {game_i}, step {step_i} legal moves are different:")

                    print(probs_env.as_string())
                    print(chess_env)

                    show_moves_difference(chess_legal_moves, probs_legal_moves)
                    assert False

                move = chess_legal_moves[random.randint(0, len(chess_legal_moves) - 1)]
                chess_env.push(chess.Move.from_uci(move))
                probs_env.move(move)

                # if game_i == 10 and step_i == 38:
                #     print(f"Previous move: {move}")
                #     print(f"Probs game_state: {probs_env.game_state()}")
                #     print()
                #     print("Probs board:")
                #     print(probs_env.as_string())
                #     print()
                #     print("Chess board:")
                #     print(chess_env)
                #     print()

                if probs_env.game_state() != 'undecided' or chess_env.is_game_over():
                    if chess_env.is_checkmate() and step_i % 2 == 0:
                        assert probs_env.game_state() == 'white_won', f"Game {game_i}, step {step_i}, expected game won by white, but has {probs_env.game_state()}"
                    elif chess_env.is_checkmate():
                        assert probs_env.game_state() == 'black_won', f"Game {game_i}, step {step_i}, expected game won by black, but has {probs_env.game_state()}"
                    else:
                        assert probs_env.game_state() == 'draw', f"Game {game_i}, step {step_i}, expected game ended with draw, but has {probs_env.game_state()}, chess outcome {chess_env.outcome()}"
                    break
        except Exception:
            print(f"Error at game_i={game_i}")
            print(traceback.format_exc())
            break

        if game_i % 1000 == 0:
            print(f"Tested {game_i+1}/{GAMES} games")
            
    print(f"OK! Checked {GAMES} games, each {STEPS} plys")