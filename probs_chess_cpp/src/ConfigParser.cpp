#include "ConfigParser.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

using namespace std;



ConfigParser::ConfigParser(const string& filePath) {
    ifstream file(filePath);
    if (!file.good()) {
        throw runtime_error("Error: Unable to open file at " + filePath);
    }
    config = YAML::LoadFile(filePath);
}

YAML::Node ConfigParser::get_node(const string& key) const {
    istringstream keyStream(key);
    string token;
    vector<YAML::Node> n{config};

    while (getline(keyStream, token, '.')) {
        if (!n.back()[token]) {
            throw runtime_error("Invalid configuration key: " + key);
        }
        n.push_back(n.back()[token]);
    }
    return n.back();
}

string ConfigParser::get_string(const string& key) const {
    auto node = get_node(key);
    return node.as<string>();
}

int ConfigParser::get_int(const string& key) const {
    auto node = get_node(key);
    return node.as<int>();
}

double ConfigParser::get_double(const string& key) const {
    auto node = get_node(key);
    return node.as<double>();
}