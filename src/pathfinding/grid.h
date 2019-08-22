#ifndef PATHFINDER_GRID_H
#define PATHFINDER_GRID_H

#include <array>
#include "vector2.h"

template<typename Vec, typename Cell>
class MapGrid {
public:
    MapGrid(const Cell *grid, Vec strides, Vec dimensions) :
            _grid(grid), _strides(strides), _dimensions(dimensions) {};

    Vec dimensions() const { return _dimensions; }

    bool get(Vec const &v) const {
        auto x = v.x;
        auto y = v.y;

        if (x < 0 || x >= _dimensions.x) {
            throw std::runtime_error("out of bounds");
        }
        if (y < 0 || y >= _dimensions.y) {
            throw std::runtime_error("out of bounds");
        }
        return *(_grid + _strides.x * x + _strides.y * y);
    }

private:
    const Cell *_grid;
    const Vec _strides, _dimensions;
};

template<typename Vec, typename Cell>
bool inGrid(Vec const &p, MapGrid<Vec, Cell> const &grid) {
    auto dim = grid.dimensions();
    return p.x >= 0 && p.x < dim.x && p.y >= 0 && p.y < dim.y;
}

Vector2<long> neighbors[]{
        {.x=1, .y=0},
        {.x=0, .y=-1},
        {.x=-1, .y=0},
        {.x=0, .y=1},

        {.x=1, .y=-1},
        {.x=-1, .y=-1},
        {.x=-1, .y=1},
        {.x=1, .y=1},
};

template<typename Vec, typename Cell>
std::vector<Vec> get_neighbors(Vec const &p, MapGrid<Vec, Cell> const &grid) {
    std::vector<Vec> result;
    for (auto const &n : neighbors) {
        auto v = Vec{
                .x=p.x + n.x,
                .y=p.y + n.y,
        };
        if (inGrid(v, grid)) {
            result.emplace_back(v);
        }
    }

    return result;
}

#endif //PATHFINDER_GRID_H
