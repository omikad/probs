#include <chrono>
#include <iomanip>
#include <map>
#include <iostream>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "utils/usage_counter.h"


using namespace std;


namespace probs {

// Adapted from https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

pair<int, int> virtualAndPhysicalMemoryUsedByCurrentProcess() { //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int vm_mem = -1;
    int ph_mem = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            vm_mem = parseLine(line);
            if (ph_mem >= 0) break;
        }
        else if (strncmp(line, "VmRSS:", 6) == 0){
            ph_mem = parseLine(line);
            if (vm_mem >= 0) break;
        }
    }
    fclose(file);
    return pair<int, int>(vm_mem, ph_mem);
}


UsageCounter::UsageCounter() {
    MarkCheckpoint("");
}


void UsageCounter::MarkCheckpoint(string name) {
    checkpoint_names.push_back(name);

    // auto vm_ph_mem = virtualAndPhysicalMemoryUsedByCurrentProcess();
    // checkpoint_physical_mems.push_back(vm_ph_mem.first);
    // checkpoint_virtual_mems.push_back(vm_ph_mem.second);
    checkpoint_times.push_back(chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch()).count());
}


void UsageCounter::PrintStats() const {
    vector<string> names;

    map<string, vector<long long>> times;
    // map<string, vector<long long>> vm_mems;
    // map<string, vector<long long>> ph_mems;
    for (int i = 1; i < checkpoint_names.size(); i++) {
        string name = checkpoint_names[i];

        if (times.find(name) == times.end())
            names.push_back(name);

        long long time_delta = checkpoint_times[i] - checkpoint_times[i - 1];
        // long long vm_delta = checkpoint_virtual_mems[i] - checkpoint_virtual_mems[i - 1];
        // long long ph_delta = checkpoint_physical_mems[i] - checkpoint_physical_mems[i - 1];
        times[name].push_back(time_delta);
        // vm_mems[name].push_back(vm_delta);
        // ph_mems[name].push_back(ph_delta);
    }

    vector<vector<string>> print_tabs(4);

    for (const string& name : names) {
        stringstream tab_name; tab_name << "[USAGE] " << name << ":";

        long long total_time = 0; for (const long long time_delta : times[name]) total_time += time_delta;
        stringstream tab_time_total; tab_time_total << "time spent ";
        if (total_time < 1000)
            tab_time_total << total_time << " ns";
        else if (total_time < 1000000)
            tab_time_total << (double)total_time/1000 << " μs ";
        else if (total_time < 1000000000)
            tab_time_total << (double)(total_time/1000)/1000 << " ms";
        else
            tab_time_total << (double)(total_time/1000000)/1000 << " sec";
        if (times[name].size() > 1)
            tab_time_total << "; hits=" << times[name].size();

        // long long total_vm_mem = 0; for (const int vm_delta : vm_mems[name]) total_vm_mem += vm_delta;
        // stringstream tab_vm_total; tab_vm_total << "virt mem delta " << total_vm_mem << " KB";

        // long long total_ph_mem = 0; for (const int ph_delta : ph_mems[name]) total_ph_mem += ph_delta;
        // stringstream tab_ph_total; tab_ph_total << "phys mem delta " << total_ph_mem << " KB";

        print_tabs[0].push_back(tab_name.str());
        print_tabs[1].push_back(tab_time_total.str());
        // print_tabs[2].push_back(tab_vm_total.str());
        // print_tabs[3].push_back(tab_ph_total.str());
    }

    vector<int> widths(print_tabs.size(), 0);
    for (int ti = 0; ti < print_tabs.size(); ti++)
        for (auto& s : print_tabs[ti])
            widths[ti] = max(widths[ti], (int)s.size());

    int lines = 0;
    for (int ti = 0; ti < print_tabs.size(); ti++)
        lines = max(lines, (int)print_tabs[ti].size());

    for (int li = 0; li < lines; li++) {
        for (int ti = 0; ti < print_tabs.size(); ti++) {
            string s = li < print_tabs[ti].size() ? print_tabs[ti][li] : "";
            cout << s << string(widths[ti] - s.size() + 3, ' ');
        }
        cout << endl;
    }
}


}   // namespace probs