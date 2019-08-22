#ifndef PATHFINDER_GRID_H
#define PATHFINDER_GRID_H

#include "vector2.h"

template<typename Vec, typename Cell>
class MapGrid {
public:
    MapGrid(Cell *grid, Vec strides, Vec dimensions) :
            _grid(grid), _strides(strides), _dimensions(dimensions) {};

    const Vec dimensions() const { return _dimensions; }

    bool get(Vec v) const {
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
    Cell *_grid;
    Vec _strides, _dimensions;
};

template<typename Vec, typename Cell>
bool inGrid(Vec p, MapGrid<Vec, Cell> const &grid) {
    auto dim = grid.dimensions();
    return p.x >= 0 && p.x < dim.x && p.y >= 0 && p.y < dim.y;
}

template<typename Vec, typename Cell>
std::vector<Vec> get_neighbors(Vec p, MapGrid<Vec, Cell> const &grid) {
    std::vector<Vec> result;
    for (auto x = -1; x <= 1; x++) {
        for (auto y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) {
                continue;
            }
            auto v = Vec{
                    p.x + x,
                    p.y + y,
            };
            if (inGrid(v, grid)) {
                result.emplace_back(v);
            }
        }
    }

    return result;
}

#endif //PATHFINDER_GRID_H
