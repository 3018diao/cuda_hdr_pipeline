#ifndef INCLUDED_ARGPARSE
#define INCLUDED_ARGPARSE

#include <string_view>
#include <stdexcept>

namespace argparse {
    struct usage_error : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    bool parse_bool_flag(const char* const*& argv, std::string_view option);
    const char* parse_string_argument(const char* const*& argv, std::string_view option);
    bool parse_int_argument(int& value, const char* const*& argv, std::string_view option);
    bool parse_float_argument(float& value, const char* const*& argv, std::string_view option);
}

#endif // INCLUDED_ARGPARSE
