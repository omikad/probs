#pragma once

#include <yaml-cpp/yaml.h>
#include <string>


class ConfigParser {
public:
    explicit ConfigParser(const std::string& file_path);

    std::string GetString(const std::string& key) const;
    int GetInt(const std::string& key) const;
    double GetDouble(const std::string& key) const;

private:
    YAML::Node GetNode(const std::string& key) const;
    YAML::Node config;
};
