#include <fstream>
#include <stdexcept>
#include <iostream>

#include "infra/config_parser.h"

using namespace std;


ConfigParser::ConfigParser(const string& file_path) {
    ifstream file(file_path);
    if (!file.good()) {
        throw runtime_error("Error: Unable to open file at " + file_path);
    }
    config = YAML::LoadFile(file_path);
}

bool ConfigParser::KeyExist(const std::string& key) const {
    istringstream key_stream(key);
    string token;
    vector<YAML::Node> n{config};

    while (getline(key_stream, token, '.')) {
        if (!n.back()[token]) {
            return false;
        }
        n.push_back(n.back()[token]);
    }
    return true;
}

YAML::Node ConfigParser::GetNode(const string& key) const {
    istringstream key_stream(key);
    string token;
    vector<YAML::Node> n{config};

    while (getline(key_stream, token, '.')) {
        if (!n.back()[token]) {
            throw runtime_error("Invalid configuration key: " + key);
        }
        n.push_back(n.back()[token]);
    }
    return n.back();
}

string ConfigParser::GetString(const string& key) const {
    auto node = GetNode(key);
    return node.as<string>();
}

int ConfigParser::GetInt(const string& key) const {
    auto node = GetNode(key);
    return node.as<int>();
}

double ConfigParser::GetDouble(const string& key) const {
    auto node = GetNode(key);
    return node.as<double>();
}