#ifndef INCLUDED_UTILS_OBJ_STREAM
#define INCLUDED_UTILS_OBJ_STREAM

#include <utility>
#include <cstdlib>
//#include <charconv>
#include <string_view>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "obj.h"

namespace OBJ {
    class Stream {
        const char* ptr;
        const char* end;
        float size;
        int line = 1;
        const char* name;

        void endLine() {
            if (line % 0x4000 == 0)
                std::cout << '\r' << std::fixed << std::setprecision(0) << 100.0f * (1.0f - (end - ptr) / size) << '%' << std::flush;
            ++line;
        }

        void endFile() {
            std::cout << "\r100%\n" << std::flush;
        }

    public:
        Stream(const char* begin, const char* end, const char* name)
            : ptr(begin), end(end), size(static_cast<float>(end - begin)), name(name) {
        }

        template <typename... Args>
        [[noreturn]]
        void throwError(Args&&... args) const {
            std::ostringstream err;
            err << name << '(' << line << "): error: ";
            (err << ... << args);
            throw OBJ::parse_error(err.str());
        }

        template <typename... Args>
        void warn(Args&&... args) const {
            ((std::cout << '\r' << name << '(' << line << "): warning: ") << ... << args) << std::endl;
        }

        bool skipLine() {
            while (ptr != end) {
                if (*ptr++ == '\n') {
                    endLine();
                    return true;
                }
            }
            return false;
        }

        template <char... C>
        bool consume() {
            static_assert(sizeof...(C) > 0);
            static_assert(((C != ' ') && ...) && ((C != '\t') && ...) && ((C != '\r') && ...) && ((C != '\n') && ...), "consume does not support whitespace characters");

            if (auto c = ptr; ptr + sizeof...(C) < end && ((*c++ == C) && ...)) {
                ptr = c;
                return true;
            }
            return false;
        }

        template <char C>
        bool expect() {
            if (!consume<C>())
                throwError("expected '", C, '\'');
        }

        bool consumeHorizontalWS() {
            if (ptr == end || (*ptr != ' ' && *ptr != '\t' && *ptr != '\r'))
                return false;

            do ++ptr; while (ptr != end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r'));

            return true;
        }

        void expectHorizontalWS() {
            if (!consumeHorizontalWS())
                throwError("expected horizontal white space");
        }

        bool finishLine() {
            consumeHorizontalWS();

            if (ptr == end)
                return true;

            if (*ptr == '\n') {
                ++ptr;
                endLine();
                return true;
            }

            return false;
        }

        void expectLineEnd() {
            if (!finishLine())
                throwError("expected newline");
        }

        std::string_view consumeNonWS() {
            auto begin = ptr;
            while (ptr != end && *ptr != ' ' && *ptr != '\t' && *ptr != '\r' && *ptr != '\n')
                ++ptr;
            return { begin, static_cast<std::size_t>(ptr - begin) };
        }

        std::string_view expectNonWS() {
            auto v = consumeNonWS();
            if (v.empty())
                throwError("expected string");
            return v;
        }

        bool consumeInteger(int& n) {
#ifdef _MSC_VER
            auto [token_end, err] = std::from_chars(ptr, end, n);

            if (static_cast<bool>(err)) {
                if (err == std::errc::result_out_of_range)
                    throwError("decimal number out of range");
                return false;
            }

            ptr = token_end;

            return true;
#else  // WORKAROUND for lack of std::from_chars in gcc and clang
            char* token_end;
            n = std::strtol(ptr, &token_end, 10);

            if (token_end == ptr)
                return false;

            ptr = token_end;

            return true;
#endif
        }

        int expectInteger() {
#ifdef _MSC_VER
            int n;
            auto [token_end, err] = std::from_chars(ptr, end, n);

            if (static_cast<bool>(err)) {
                if (err == std::errc::result_out_of_range)
                    throwError("decimal number out of range");
                throwError("expected decimal number");
            }

            ptr = token_end;

            return n;
#else  // WORKAROUND for lack of std::from_chars in gcc and clang
            char* token_end;
            int n = std::strtol(ptr, &token_end, 10);

            if (token_end == ptr)
                return false;

            ptr = token_end;

            return n;
#endif
        }

        bool consumeFloat(float& f) {
#ifdef _MSC_VER
            auto [token_end, err] = std::from_chars(ptr, end, f);

            if (static_cast<bool>(err)) {
                if (err == std::errc::result_out_of_range)
                    throwError("floating point number out of range");
                return false;
            }

            ptr = token_end;

            return true;
#else  // WORKAROUND for lack of std::from_chars in gcc and clang
            char* token_end;
            f = std::strtof(ptr, &token_end);

            if (token_end == ptr)
                return false;

            ptr = token_end;

            return true;
#endif
        }

        float expectFloat() {
#ifdef _MSC_VER
            float f;
            auto [token_end, err] = std::from_chars(ptr, end, f);

            if (static_cast<bool>(err)) {
                if (err == std::errc::result_out_of_range)
                    throwError("floating point number out of range");
                throwError("expected floating point number");
            }

            ptr = token_end;

            return f;
#else  // WORKAROUND for lack of std::from_chars in gcc and clang
            char* token_end;
            float f = std::strtof(ptr, &token_end);

            if (token_end == ptr)
                throwError("expected floating point number");

            ptr = token_end;

            return f;
#endif
        }

        template <typename Consumer>
        void consume(Consumer&& consumer) {
            while (ptr != end) {
                char c = *ptr++;

                switch (c) {
                    case '\n':
                        endLine();
                    case '\r':
                    case '\t':
                    case ' ':
                        break;
                    default:
                        if (!consumer.consume(*this, c))
                            return;
                        break;
                }
            }

            endFile();
        }
    };
}

#endif // INCLUDED_UTILS_OBJ_STREAM
