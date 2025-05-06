#include <cstring>

#include "argparse.h"

namespace {
    bool compare_option(const char* arg, std::string_view option) {
        return std::strncmp(arg, option.data(), option.length()) == 0;
    }
}

namespace argparse {
    bool parse_bool_flag(const char* const*& argv, std::string_view option) {
        if (!compare_option(*argv, option))
            return false;

        ++argv;
        return true;
    }

    const char* parse_string_argument(const char* const*& argv, std::string_view option) {
        if (!compare_option(*argv, option))
            return nullptr;

        const char* startptr = *argv + option.length();

        if (*startptr)
            return startptr;

        startptr = *++argv;

        if (!*startptr)
            throw usage_error("expected argument");

        return startptr;
    }

    bool parse_int_argument(int& value, const char* const*& argv, std::string_view option) {
        if (!compare_option(*argv, option))
            return false;

        const char* startptr = *argv + option.length();

        if (!*startptr) {
            startptr = *++argv;
            if (!*startptr)
                throw usage_error("expected integer argument");
        }

        char* endptr = nullptr;

        int v = std::strtol(startptr, &endptr, 10);

        if (*endptr)
            throw usage_error("argument is not an integer");

        value = v;
        return true;
    }

    bool parse_float_argument(float& value, const char* const*& argv, std::string_view option) {
        if (!compare_option(*argv, option))
            return false;

        const char* startptr = *argv + option.length();

        if (!*startptr) {
            startptr = *++argv;
            if (!*startptr)
                throw usage_error("expected float argument");
        }

        char* endptr = nullptr;

        float v = std::strtof(startptr, &endptr);

        if (*endptr)
            throw usage_error("argument is not a float");

        value = v;
        return true;
    }
}
