import time
import numpy as np
import subprocess
import chess


class UciChess:
    def __init__(self, verbose: bool):
        self.process = subprocess.Popen(["./ProbsChess"], stdin = subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf8')
        self.verbose = verbose

    def write(self, line):
        if self.verbose:
            print("UciChess>:", line)
        self.process.stdin.write(line)
        self.process.stdin.write("\n")
        self.process.stdin.flush()

    def read(self):
        response = self.process.stdout.readline()
        response = response.rstrip()
        if self.verbose:
            print("<ProbsChess:", response)
        return response

    def read_until(self, resp_wait_prefix, wait_cnt):
        for _ in range(wait_cnt):
            resp = self.read()
            if resp.startswith(resp_wait_prefix):
                return resp
        raise Exception(f"Can't find response starting with {resp_wait_prefix} out of {wait_cnt} responses")
    
    def close(self):
        # os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        self.process.kill()


def play_games(n_games, n_steps):
    outcomes = [[0] * 3 for _ in range(2)]
    # outcomes[0] is engine as white [wins, loses, draws]
    # outcomes[1] si engine as black [wins, loses, draws]

    for gi in range(n_games):
        chess_env = chess.Board()
        proc.write('ucinewgame')

        moves = []

        for si in range(n_steps):
            chess_legal_moves = [str(m) for m in chess_env.legal_moves]
            if (si + gi) % 2 == 1:
                move = chess_legal_moves[np.random.randint(len(chess_legal_moves))]

            else:
                if len(moves) == 0:
                    proc.write("position startpos")
                else:
                    proc.write("position startpos moves " + ' '.join(moves))
                proc.write("go wtime 100 btime 100 movestogo 10")
                move_str = proc.read_until("bestmove ", 1000)
                move = move_str.split()[1]
                if move not in chess_legal_moves:
                    print(f"Got move {move} not in legal moves: {chess_legal_moves}")
                    print(chess_env)
                assert(move in chess_legal_moves)

            moves.append(move)
            chess_env.push(chess.Move.from_uci(move))

            if chess_env.is_game_over():
                break

        if chess_env.outcome() is None or chess_env.outcome().winner is None:
            outcomes[gi % 2][2] += 1
        elif chess_env.outcome().winner == chess.WHITE:
            if gi % 2 == 0:
                outcomes[0][0] += 1
            else:
                outcomes[1][1] += 1
        else:   # black wins
            if gi % 2 == 0:
                outcomes[0][1] += 1
            else:
                outcomes[1][0] += 1

    print(f"Engine as white: wins,draws,loses={outcomes[0]}")
    print(f"Engine as black: wins,draws,loses={outcomes[1]}")


if __name__ == "__main__":
    np.random.seed(111)
    # GAMES, STEPS, VERBOSE = 1, 5, True
    GAMES, STEPS, VERBOSE = 100, 500, False

    proc = UciChess(verbose=VERBOSE)

    try:
        proc.write("uci")
        proc.read_until('uciok', 1000)

        proc.write('isready')
        proc.read_until('readyok', 1000)

        play_games(GAMES, STEPS)

        proc.write('quit')
        time.sleep(0.5)
    finally:
        proc.close()

    print('Done')