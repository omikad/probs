/*
    Adapted from Leela Chess Zero: https://github.com/LeelaChessZero/lc0/
*/

#pragma once

#include <stdexcept>
#include <iostream>

namespace lczero {

class Exception : public std::runtime_error {
 public:
  Exception(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace lczero

namespace probs {

class Exception : public std::runtime_error {
 public:
  Exception(const std::string& what) : std::runtime_error(what) {
    std::cerr << what << std::endl;
  }
};

}  // namespace probs