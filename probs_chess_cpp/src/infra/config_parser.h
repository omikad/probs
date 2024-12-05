#pragma once

#include <yaml-cpp/yaml.h>
#include <string>


class ConfigParser {
public:
    explicit ConfigParser(const std::string& file_path);

    bool KeyExist(const std::string& key) const;
    std::string GetString(const std::string& key) const;
    int GetInt(const std::string& key, bool key_required, const int default_value) const;
    double GetDouble(const std::string& key, bool key_required, const double default_value) const;

private:
    YAML::Node GetNode(const std::string& key, bool key_required, bool* key_found) const;
    YAML::Node config;
};
