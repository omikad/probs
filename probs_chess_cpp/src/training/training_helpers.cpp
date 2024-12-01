#include "training/training_helpers.h"

using namespace std;


namespace probs {

lczero::Move GetMoveWithExploration(const vector<MoveEstimation>& moves_estimation, int env_ply, bool exploration_full_random, int exploration_num_first_moves) {
    assert(moves_estimation.size() > 0);

    if (exploration_full_random) {
        int pi = rand() % moves_estimation.size();
        return moves_estimation[pi].move;
    }

    if (exploration_num_first_moves > 0) {
        double max_score = 0;
        vector<double> scores;
        for (auto& move_and_score : moves_estimation) {
            double score = move_and_score.score;
            scores.push_back(score);
            max_score = max(max_score, score);
        }

        for (int i = 0; i < scores.size(); i++)
            scores[i] = exp(scores[i] - max_score);
        
        double summ = 0;
        for (int i = 0; i < scores.size(); i++)
            summ += scores[i];

        for (int i = 0; i < scores.size(); i++)
            scores[i] /= summ;

        double p = ((double)(rand() % 1000000) / 1000000);
        int idx = 0;
        for (; idx < scores.size(); idx++) {
            p -= scores[idx];
            if (p < 0)
                break;
        }

        return moves_estimation[idx].move;
    }

    // greedy
    int best_i = 0;
    for (int i = 1; i < moves_estimation.size(); i++)
        if (moves_estimation[i].score > moves_estimation[best_i].score)
            best_i = i;
    return moves_estimation[best_i].move;
}

}  // namespace probs
