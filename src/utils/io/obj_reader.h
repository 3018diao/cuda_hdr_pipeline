#ifndef INCLUDED_UTILS_OBJ_READER
#define INCLUDED_UTILS_OBJ_READER

#include "obj_stream.h"

namespace OBJ {
    template <typename Consumer>
    class Reader {
        Consumer& consumer;

        void consumeVertex(OBJ::Stream& stream) {
            float x = stream.expectFloat();
            stream.expectHorizontalWS();
            float y = stream.expectFloat();
            stream.expectHorizontalWS();
            float z = stream.expectFloat();

            if (float w; stream.consumeHorizontalWS() && stream.consumeFloat(w)) {
                stream.expectLineEnd();
                consumer.consumeVertex(stream, x, y, z, w);
                return;
            }

            stream.expectLineEnd();
            consumer.consumeVertex(stream, x, y, z);
        }

        void consumeNormal(OBJ::Stream& stream) {
            float x = stream.expectFloat();
            stream.expectHorizontalWS();
            float y = stream.expectFloat();
            stream.expectHorizontalWS();
            float z = stream.expectFloat();
            stream.expectLineEnd();

            consumer.consumeNormal(stream, x, y, z);
        }

        void consumeTexcoord(OBJ::Stream& stream) {
            float u = stream.expectFloat();

            if (float v; stream.consumeHorizontalWS() && stream.consumeFloat(v)) {
                if (float w; stream.consumeHorizontalWS() && stream.consumeFloat(w)) {
                    stream.expectLineEnd();
                    consumer.consumeTexcoord(stream, u, v, w);
                    return;
                }

                stream.expectLineEnd();
                consumer.consumeTexcoord(stream, u, v);
                return;
            }

            stream.expectLineEnd();
            consumer.consumeTexcoord(stream, u);
        }

        void consumeFace(OBJ::Stream& stream) {
            do {
                int v = stream.expectInteger();
                int t = 0;
                int n = 0;

                if (stream.consume<'/'>()) {
                    stream.consumeInteger(t);

                    if (stream.consume<'/'>())
                        stream.consumeInteger(n);
                }

                consumer.consumeFaceVertex(stream, v, n, t);
            } while (!stream.finishLine());

            consumer.finishFace(stream);
        }

        void consumeObjectName(OBJ::Stream& stream) {
            auto name = stream.expectNonWS();

            while (!stream.finishLine()) {
                auto n = stream.consumeNonWS();
                name = { &name[0], static_cast<std::size_t>((&n[0] + size(n)) - &name[0]) };
            }

            consumer.consumeObjectName(stream, name);
        }

        void consumeGroupName(OBJ::Stream& stream) {
            do {
                auto name = stream.expectNonWS();
                consumer.consumeGroupName(stream, name);
            } while (!stream.finishLine());

            consumer.finishGroupAssignment(stream);
        }

        void consumeSmoothingGroup(OBJ::Stream& stream) {
            int n;

            if (!stream.consumeInteger(n)) {
                if (stream.consume<'o', 'f', 'f'>())
                    n = 0;
                else
                    stream.throwError("expected smoothing group index or 'off'");
            }

            stream.expectLineEnd();
            consumer.consumeSmoothingGroup(stream, n);
        }

        void consumeMtlLib(OBJ::Stream& stream) {
            auto name = stream.expectNonWS();
            stream.expectLineEnd();
            consumer.consumeMtlLib(stream, name);
        }

        void consumeUseMtl(OBJ::Stream& stream) {
            auto name = stream.expectNonWS();
            stream.expectLineEnd();
            consumer.consumeUseMtl(stream, name);
        }

    public:
        Reader(Consumer& consumer)
            : consumer(consumer) {
        }

        bool consume(OBJ::Stream& stream, char c) {
            switch (c) {
                case 'v':
                    if (stream.consumeHorizontalWS()) {
                        consumeVertex(stream);
                        break;
                    } else if (stream.consume<'n'>()) {
                        if (stream.consumeHorizontalWS()) {
                            consumeNormal(stream);
                            break;
                        }
                    } else if (stream.consume<'t'>()) {
                        if (stream.consumeHorizontalWS()) {
                            consumeTexcoord(stream);
                            break;
                        }
                    }
                    [[fallthrough]];
                case 'f':
                    if (stream.consumeHorizontalWS()) {
                        consumeFace(stream);
                        break;
                    }
                    [[fallthrough]];
                case 'o':
                    if (stream.consumeHorizontalWS()) {
                        consumeObjectName(stream);
                        break;
                    }
                    [[fallthrough]];
                case 'g':
                    if (stream.consumeHorizontalWS()) {
                        consumeGroupName(stream);
                        break;
                    }
                    [[fallthrough]];
                case 's':
                    if (stream.consumeHorizontalWS()) {
                        consumeSmoothingGroup(stream);
                        break;
                    }
                    [[fallthrough]];
                case 'm':
                    if (stream.consume<'t', 'l', 'l', 'i', 'b'>()) {
                        if (stream.consumeHorizontalWS())
                            consumeMtlLib(stream);
                        break;
                    }
                    [[fallthrough]];
                case 'u':
                    if (stream.consume<'s', 'e', 'm', 't', 'l'>()) {
                        if (stream.consumeHorizontalWS())
                            consumeUseMtl(stream);
                        break;
                    }
                    [[fallthrough]];
                default:
                    stream.throwError("unknown command");
                    break;
                case '#':
                    stream.skipLine();
                    break;
            }

            return true;
        }
    };
}

#endif // INCLUDED_UTILS_OBJ_READER
