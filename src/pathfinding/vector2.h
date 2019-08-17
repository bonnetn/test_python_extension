#ifndef PATHFINDER_VECTOR2_H
#define PATHFINDER_VECTOR2_H

#include <sstream>

template<typename T>
struct Vector2 {
    T x;
    T y;
};

template<typename T>
bool operator==(Vector2<T> const &a, Vector2<T> const &b) {
    return a.x == b.x && a.y == b.y;
}

template<typename T>
bool operator!=(Vector2<T> const &a, Vector2<T> const &b) {
    return !(a == b);
}

template<typename T>
std::ostream &operator<<(std::ostream &os, Vector2<T> const &v) {
    return os << "Vec2[" << v.x << ", " << v.y << "]";
}

template<typename T>
double norm2(Vector2<T> a) {
    return a.x * a.x + a.y * a.y;
}

template<typename T>
double distance(Vector2<T> a, Vector2<T> b) {
    Vector2<T> c{
            a.x - b.x,
            a.y - b.y,
    };

    return norm2(c);
}

#endif //PATHFINDER_VECTOR2_H
