#ifndef PATHFINDER_GRID_H
#define PATHFINDER_GRID_H

#include "vector2.h"

template<typename Axis, typename Cell>
class MapGrid {
public:
    MapGrid(Cell *grid, Axis strideX, Axis strideY, Axis lenX, Axis lenY) :
            _grid(grid), _strideX(strideX), _strideY(strideY), _lenX(lenX), _lenY(lenY) {};

    long lengthX() const { return _lenX; }

    long lengthY() const { return _lenY; }

    bool get(Vector2<Axis> v) const {
        auto x = v.x;
        auto y = v.y;

        if (x < 0 || x >= _lenX) {
            throw std::runtime_error("out of bounds");
        }
        if (y < 0 || y >= _lenY) {
            throw std::runtime_error("out of bounds");
        }
        return *(_grid + _strideX * x + _strideY * y);
    }

private:
    Cell *_grid;
    Axis _strideX, _strideY;
    Axis _lenX, _lenY;
};

template<typename Axis, typename Cell>
bool inGrid(Vector2<Axis> p, MapGrid<Axis, Cell> const &grid) {
    return p.x >= 0 && p.x < grid.lengthX() && p.y >= 0 && p.y < grid.lengthY();
}

template<typename Axis, typename Cell>
std::vector<Vector2<Axis>> get_neighbors(Vector2<Axis> p, MapGrid<Axis, Cell> const &grid) {
    std::vector<Vector2<Axis>> result;
    for (auto x = -1; x <= 1; x++) {
        for (auto y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) {
                continue;
            }
            auto v = Vector2<Axis>{
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
