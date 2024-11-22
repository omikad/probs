#pragma once

#include <yaml-cpp/yaml.h>
#include <string>

using namespace std;


class ConfigParser {
public:
    explicit ConfigParser(const string& file_path);

    string GetString(const string& key) const;
    int GetInt(const string& key) const;
    double GetDouble(const string& key) const;

private:
    YAML::Node GetNode(const string& key) const;
    YAML::Node config;
};
