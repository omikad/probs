#pragma once
#include <vector>
#include <string>


namespace probs {

class UsageCounter {
    public:
        UsageCounter();
        void MarkCheckpoint(std::string name);
        void PrintStats() const;

    private:
        std::vector<std::string> checkpoint_names;
        std::vector<long long> checkpoint_times;
        std::vector<long long> checkpoint_physical_mems;
        std::vector<long long> checkpoint_virtual_mems;
};

}   // namespace probs