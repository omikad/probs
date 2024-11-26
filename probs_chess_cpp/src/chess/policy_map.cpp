#include "chess/policy_map.h"

using namespace std;


namespace probs {

vector<int> ReversePolicyMap() {
    int maxval = 0;
    for (short val: kConvPolicyMap)
        maxval = max(maxval, (int)val);
    vector<int> result(maxval + 1, -1);

    for (const auto& move_idx : kConvPolicyMap)
        if (move_idx >= 0) {
            const auto policy_idx = &move_idx - kConvPolicyMap;
            result[move_idx] = policy_idx;
        }

    return result;
};
vector<int> move_to_policy_idx_map = ReversePolicyMap();


}  // namespace probs
