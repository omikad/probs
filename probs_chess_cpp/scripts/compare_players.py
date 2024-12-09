import random
import os
import sys


if __name__ == "__main__":
    random.seed(111)

    lib_probs_chess_path = os.environ['LIB_PROBS_CHESS']
    assert lib_probs_chess_path is not None and len(os.environ['LIB_PROBS_CHESS']) > 1

    sys.path.append(lib_probs_chess_path)
    from libprobs_chess import ChessEnv