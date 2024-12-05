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

YAML::Node ConfigParser::GetNode(const string& key, bool key_required, bool* key_found) const {
    istringstream key_stream(key);
    string token;
    vector<YAML::Node> n{config};
    *key_found = false;

    while (getline(key_stream, token, '.')) {
        if (!n.back()[token]) {
            if (key_required)
                throw runtime_error("Invalid configuration key: " + key);
            return n.back();
        }
        n.push_back(n.back()[token]);
    }
    *key_found = true;
    return n.back();
}

string ConfigParser::GetString(const string& key) const {
    bool key_found;
    auto node = GetNode(key, true, &key_found);
    return node.as<string>();
}

int ConfigParser::GetInt(const string& key, bool key_required, const int default_value) const {
    bool key_found;
    auto node = GetNode(key, key_required, &key_found);
    return key_found ? node.as<int>() : default_value;
}

double ConfigParser::GetDouble(const string& key, bool key_required, const double default_value) const {
    bool key_found;
    auto node = GetNode(key, key_required, &key_found);
    return key_found ? node.as<double>() : default_value;
}