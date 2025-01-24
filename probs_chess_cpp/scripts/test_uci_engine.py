from collections import Counter
import pprint
import time
from typing import Optional
import numpy as np
import subprocess
import chess


class UciChess:
    def __init__(self, config: Optional[str], verbose: bool):
        cmd = ["./ProbsChess"]
        if config is not None:
            cmd.append('uci')
            cmd.append(config)
        self.process = subprocess.Popen(cmd, stdin = subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf8')
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


class Outcomes:
    def __init__(self):
        self.outcomes = [[0] * 3 for _ in range(2)]
        # outcomes[0] is engine as white [wins,draws,loses]
        # outcomes[1] si engine as black [wins,draws,loses]

    def process_result(self, chess_env, player_0_step_0):
        if chess_env.outcome() is None or chess_env.outcome().winner is None:
            self.outcomes[player_0_step_0][1] += 1
        elif chess_env.outcome().winner == chess.WHITE:
            if player_0_step_0 == 0:
                self.outcomes[0][0] += 1
            else:
                self.outcomes[1][2] += 1
        else:   # black wins
            if player_0_step_0 == 0:
                self.outcomes[0][2] += 1
            else:
                self.outcomes[1][0] += 1

    def print(self):
        print(f"First player as white: wins,draws,loses={self.outcomes[0]}")
        print(f"First player as black: wins,draws,loses={self.outcomes[1]}")
        wins = self.outcomes[0][0] + self.outcomes[1][0]
        draws = self.outcomes[0][1] + self.outcomes[1][1]
        total_games = sum(sum(row) for row in self.outcomes)
        score = (wins + draws / 2) / total_games
        print(f"First player score = {score}")


def play_vs_random():
    np.random.seed(111)
    # GAMES, STEPS, VERBOSE = 1, 10, True
    GAMES, STEPS, VERBOSE = 100, 500, False

    proc = UciChess(config=None, verbose=VERBOSE)

    try:
        proc.write("uci")
        proc.read_until('uciok', 1000)

        proc.write('isready')
        proc.read_until('readyok', 1000)

        outcomes = Outcomes()

        for gi in range(GAMES):
            chess_env = chess.Board()
            proc.write('ucinewgame')

            moves = []

            for si in range(STEPS):
                chess_legal_moves = [str(m) for m in chess_env.legal_moves]
                if (si + gi) % 2 == 1:
                    move = chess_legal_moves[np.random.randint(len(chess_legal_moves))]

                else:
                    if len(moves) == 0:
                        proc.write("position startpos")
                    else:
                        proc.write("position startpos moves " + ' '.join(moves))
                    proc.write("go wtime 1000 btime 1000 movestogo 10")
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

            outcomes.process_result(chess_env, gi % 2)

        outcomes.print()

        proc.write('quit')
        time.sleep(0.5)
    finally:
        proc.close()

    print('Done')


def play_model_vs_model():
    np.random.seed(111)
    GAMES, STEPS, VERBOSE = 100, 500, False
    RANDOMIZE_PLIES = 2
    SEARCH_STRINGS = [
        "go wtime 500 btime 500 movestogo 10",
        "go wtime 500 btime 500 movestogo 10",
    ]
    proc1 = UciChess(config=None, verbose=VERBOSE)
    # proc2 = UciChess(config=None, verbose=VERBOSE)
    proc2 = UciChess(config='../configs/uci_engine_copy.yaml', verbose=VERBOSE)

    try:
        processes = [proc1, proc2]

        for proc in processes:
            proc.write("uci")
            proc.read_until('uciok', 1000)
            proc.write('isready')
            proc.read_until('readyok', 1000)

        outcomes = Outcomes()

        for gi in range(GAMES):
            for proc in processes:
                proc.write('ucinewgame')

            chess_env = chess.Board()
            moves = []

            for si in range(STEPS):

                if si < RANDOMIZE_PLIES:
                    chess_legal_moves = list(chess_env.legal_moves)
                    move = str(np.random.choice(chess_legal_moves))

                else:
                    curr_proc_i = (gi + si) % 2

                    proc = processes[curr_proc_i]

                    if len(moves) == 0:
                        proc.write("position startpos")
                    else:
                        proc.write("position startpos moves " + ' '.join(moves))
                    proc.write(SEARCH_STRINGS[curr_proc_i])
                    move_str = proc.read_until("bestmove ", 1000)
                    move = move_str.split()[1]

                moves.append(move)
                chess_env.push(chess.Move.from_uci(move))
                if chess_env.is_game_over():
                    break

            outcomes.process_result(chess_env, gi % 2)

        outcomes.print()

        for proc in processes:
            proc.write('quit')
        time.sleep(0.5)
    finally:
        proc1.close()
        proc2.close()

    print('Done')


def measure_accuracy_vs_stockfish():
    np.random.seed(111)
    GAMES, STEPS, VERBOSE = 100, 500, False
    RANDOMIZE_PLIES = 2
    SEARCH_STRING = "go wtime 1000 btime 1000 movestogo 1"
    STOCKFISH_TIMES = (100, 100)
    proc = UciChess(config=None, verbose=VERBOSE)

    from stockfish import Stockfish
    stockfish = Stockfish(
        path="/home/excellent/Downloads/stockfish/stockfish/stockfish-ubuntu-x86-64-sse41-popcnt",
        parameters={
            'UCI_LimitStrength': True,  # Elo rating has been calibrated at a time control of 60s+0.6s and anchored to CCRL 40/4.
            'UCI_Elo': 1320,
            'Skill Level': 1,
        })

    try:
        proc.write("uci")
        proc.read_until('uciok', 1000)
        proc.write('isready')
        proc.read_until('readyok', 1000)

        outcomes = Outcomes()
        stats = Counter()

        for gi in range(GAMES):
            proc.write('ucinewgame')
            stockfish.set_position()
            chess_env = chess.Board()

            moves = []

            for si in range(STEPS):

                stockfish.set_position(moves)
                stockfish_move = stockfish.get_best_move(wtime=STOCKFISH_TIMES[0], btime=STOCKFISH_TIMES[1])

                if si < RANDOMIZE_PLIES:
                    chess_legal_moves = list(chess_env.legal_moves)
                    move = str(np.random.choice(chess_legal_moves))

                elif (gi + si) % 2 == 0:
                    if len(moves) == 0:
                        proc.write("position startpos")
                    else:
                        proc.write("position startpos moves " + ' '.join(moves))
                    proc.write(SEARCH_STRING)
                    move_str = proc.read_until("bestmove ", 1000)
                    move = move_str.split()[1]

                    stats["equal_cnt"] += move == stockfish_move
                    stats["total_cnt"] += 1

                else:
                    move = stockfish_move

                moves.append(move)
                chess_env.push(chess.Move.from_uci(move))
                if chess_env.is_game_over():
                    break

            outcomes.process_result(chess_env, gi % 2)

        outcomes.print()
        pprint.pprint(stockfish.get_parameters())
        eq = stats['equal_cnt']
        cnt = stats['total_cnt']
        print(f"Moves equal to stockfish = {eq} / {cnt}; accuracy = {eq / cnt * 100}%")

        proc.write('quit')
        time.sleep(0.5)
    finally:
        proc.close()

    print('Done')


if __name__ == "__main__":
    # play_vs_random()
    # play_model_vs_model()
    measure_accuracy_vs_stockfish()