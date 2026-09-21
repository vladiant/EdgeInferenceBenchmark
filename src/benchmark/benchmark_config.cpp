#include "eib/benchmark_config.hpp"

#include <stdexcept>

namespace eib {

Method method_from_string(const std::string& name) {
    if (name == "classical") {
        return Method::Classical;
    }
    if (name == "cnn") {
        return Method::Cnn;
    }
    throw std::runtime_error("unknown method '" + name + "' (expected classical|cnn)");
}

std::string to_string(Method method) {
    switch (method) {
    case Method::Classical:
        return "classical";
    case Method::Cnn:
        return "cnn";
    }
    return "unknown";
}

} // namespace eib
