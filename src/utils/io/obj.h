#ifndef INCLUDED_UTILS_OBJ
#define INCLUDED_UTILS_OBJ

#include <cmath>
#include <stdexcept>
#include <filesystem>

namespace math {
    template <typename F, int D>
    struct vector;


    template <typename V>
    constexpr int dimensions = 0;

    template <typename F, int D>
    constexpr int dimensions<vector<F, D>> = D;


    template <typename F>
    struct vector<F, 2> {
        F x, y;

        vector() = default;

        constexpr vector(F x, F y)
            : x(x), y(y) {
        }

        constexpr explicit vector(F a)
            : vector { a, a } {
        }

        template <typename U>
        constexpr vector(const vector<U, 2>& v)
            : vector { v.x, v.y } {
        }

        template <typename U>
        constexpr vector(const vector<U, 1>& v, F y)
            : vector { v.x, y } {
        }

        template <typename U>
        constexpr vector& operator =(const vector<U, 2>& v) {
            return *this = vector { v.x, v.y };
        }
    };

    template <typename F>
    constexpr const vector<F, 2> operator -(const vector<F, 2>& v) {
        return { -v.x, -v.y };
    }

    template <typename F>
    constexpr const vector<F, 2> operator +(const vector<F, 2>& a, const vector<F, 2>& b) {
        return { a.x + b.x, a.y + b.y };
    }

    template <typename F>
    constexpr const vector<F, 2> operator -(const vector<F, 2>& a, const vector<F, 2>& b) {
        return { a.x - b.x, a.y - b.y };
    }

    template <typename F>
    constexpr const vector<F, 2> operator *(F s, const vector<F, 2>& v) {
        return { s * v.x, s * v.y };
    }

    template <typename F>
    constexpr bool operator ==(const vector<F, 2>& a, const vector<F, 2>& b) {
        return a.x == b.x && a.y == b.y;
    }

    template <typename F>
    constexpr auto dot(const vector<F, 2>& a, const vector<F, 2>& b) {
        return a.x * b.x + a.y * b.y;
    }


    template <typename F>
    struct vector<F, 3> {
        F x, y, z;

        vector() = default;

        constexpr vector(F x, F y, F z)
            : x(x), y(y), z(z) {
        }

        constexpr explicit vector(F a)
            : vector { a, a, a } {
        }

        template <typename U>
        constexpr vector(const vector<U, 3>& v)
            : vector { v.x, v.y, v.z } {
        }

        template <typename U>
        constexpr vector(const vector<U, 2>& v, F z)
            : vector { v.x, v.y, z } {
        }

        template <typename U>
        constexpr vector& operator =(const vector<U, 3>& v) {
            return *this = vector { v.x, v.y, v.z };
        }
    };

    template <typename F>
    constexpr const vector<F, 3> operator -(const vector<F, 3>& v) {
        return { -v.x, -v.y, -v.z };
    }

    template <typename F>
    constexpr const vector<F, 3> operator +(const vector<F, 3>& a, const vector<F, 3>& b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    template <typename F>
    constexpr const vector<F, 3> operator -(const vector<F, 3>& a, const vector<F, 3>& b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    template <typename F>
    constexpr const vector<F, 3> operator *(F s, const vector<F, 3>& v) {
        return { s * v.x, s * v.y, s * v.z };
    }

    template <typename F>
    constexpr bool operator ==(const vector<F, 3>& a, const vector<F, 3>& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    template <typename F>
    constexpr auto dot(const vector<F, 3>& a, const vector<F, 3>& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template <typename F>
    constexpr const vector<F, 3> cross(const vector<F, 3>& a, const vector<F, 3>& b) {
        return { a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x };
    }


    template <typename F>
    struct vector<F, 4> {
        F x, y, z, w;

        vector() = default;

        constexpr vector(F x, F y, F z, F w)
            : x(x), y(y), z(z), w(w) {
        }

        constexpr explicit vector(F a)
            : vector { a, a, a, a } {
        }

        template <typename U>
        constexpr vector(const vector<U, 4>& v)
            : vector { v.x, v.y, v.z, v.w } {
        }

        template <typename U>
        constexpr vector(const vector<U, 3>& v, F w)
            : vector { v.x, v.y, v.z, w } {
        }

        template <typename U>
        constexpr vector& operator =(const vector<U, 4>& v) {
            return *this = vector { v.x, v.y, v.z, v.w };
        }
    };

    template <typename F>
    constexpr const vector<F, 4> operator -(const vector<F, 4>& v) {
        return { -v.x, -v.y, -v.z, -v.w };
    }

    template <typename F>
    constexpr const vector<F, 4> operator +(const vector<F, 4>& a, const vector<F, 4>& b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    }

    template <typename F>
    constexpr const vector<F, 4> operator -(const vector<F, 4>& a, const vector<F, 4>& b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }

    template <typename F>
    constexpr const vector<F, 4> operator *(F s, const vector<F, 4>& v) {
        return { s * v.x, s * v.y, s * v.z, s * v.w };
    }

    template <typename F>
    constexpr bool operator ==(const vector<F, 4>& a, const vector<F, 4>& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }

    template <typename F>
    constexpr auto dot(const vector<F, 4>& a, const vector<F, 4>& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }



    template <typename F, int D>
    constexpr const vector<F, D> operator *(const vector<F, D>& v, F s) {
        return s * v;
    }

    template <typename F, int D>
    constexpr decltype(auto) operator +=(vector<F, D>& a, const vector<F, D>& b) {
        return a = a + b;
    }

    template <typename F, int D>
    constexpr decltype(auto) operator -=(vector<F, D>& a, const vector<F, D>& b) {
        return a = a - b;
    }

    template <typename F, int D>
    constexpr decltype(auto) operator *=(vector<F, D>& v, F s) {
        return v = v * s;
    }

    template <typename F, int D>
    constexpr bool operator !=(const vector<F, D>& a, const vector<F, D>& b) {
        return !(a == b);
    }



    template <typename V>
    constexpr auto length(const V& v) {
        return sqrt(dot(v, v));
    }

    template <typename V>
    constexpr auto normalize(const V& v) {
        return v * rcp(length(v));
    }


    using float2 = vector<float, 2U>;
    using float3 = vector<float, 3U>;
    using float4 = vector<float, 4U>;

    using double2 = vector<double, 2U>;
    using double3 = vector<double, 3U>;
    using double4 = vector<double, 4U>;

    using short2 = vector<short, 2U>;
    using short3 = vector<short, 3U>;
    using short4 = vector<short, 4U>;

    using ushort2 = vector<unsigned short, 2U>;
    using ushort3 = vector<unsigned short, 3U>;
    using ushort4 = vector<unsigned short, 4U>;

    using int2 = vector<int, 2U>;
    using int3 = vector<int, 3U>;
    using int4 = vector<int, 4U>;

    using uint2 = vector<unsigned int, 2U>;
    using uint3 = vector<unsigned int, 3U>;
    using uint4 = vector<unsigned int, 4U>;
}


namespace OBJ {
    class parse_error : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    struct MeshSink {
        virtual int add_vertex(const math::float3& position, const math::float3& normal) = 0;
        virtual void add_triangle(int v_1, int v_2, int v_3) = 0;

    protected:
        MeshSink() = default;
        MeshSink(const MeshSink&) = default;
        MeshSink(MeshSink&&) = default;
        MeshSink& operator =(const MeshSink&) = default;
        MeshSink& operator =(MeshSink&&) = default;
        ~MeshSink() = default;
    };

    MeshSink& readTriangles(MeshSink& sink, const char* begin, const char* end, const char* name);
    MeshSink& readTriangles(MeshSink& sink, const std::filesystem::path& path);
}

#endif // INCLUDED_UTILS_OBJ
