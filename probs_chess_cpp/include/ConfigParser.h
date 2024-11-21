#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <yaml-cpp/yaml.h>
#include <string>

using namespace std;


class ConfigParser {
public:
    explicit ConfigParser(const string& filePath);

    string get_string(const string& key) const;
    int get_int(const string& key) const;
    double get_double(const string& key) const;

private:
    YAML::Node config;
    YAML::Node get_node(const string& key) const;
};

#endif // CONFIGPARSER_H
