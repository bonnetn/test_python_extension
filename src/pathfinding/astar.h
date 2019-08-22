//
// Created by pulp on 22/08/2019.
//

#ifndef PATHFINDER_ASTAR_H
#define PATHFINDER_ASTAR_H

#include <vector>
#include <optional>

namespace astar {
    namespace {
        constexpr double INF = std::numeric_limits<double>::infinity();

        template<typename T>
        using Array2D = std::vector<std::vector<T>>;

        template<typename Vec2>
        double h(Vec2 current, Vec2 goal) {
            return distance(current, goal);
        }

        template<typename Vec2>
        std::pair<Array2D<Vec2>, Array2D<double>>
        astar(MapGrid<Vec2, bool> const &grid, Vec2 const &start, Vec2 const &end) {
            auto dim = grid.dimensions();

            auto openSet = PriorityQueue<Vec2>{};
            openSet.push({start, 0});
            auto closedSet = Array2D<bool>(dim.x, std::vector<bool>(dim.y, false));

            auto cameFrom = Array2D<Vec2>(dim.x, std::vector<Vec2>(dim.y));

            auto gScore = Array2D<double>(dim.x, std::vector<double>(dim.y, INF));
            gScore[start.x][start.y] = 0;

            auto fScore = Array2D<double>(dim.x, std::vector<double>(dim.y, INF));
            fScore[start.x][start.y] = h(start, end);

            while (!openSet.empty()) {
                auto current = openSet.top().v;
                if (current == end) {
                    return std::make_pair(cameFrom, gScore);
                }

                openSet.pop();
                if (closedSet[current.x][current.y]) {
                    continue;
                }
                closedSet[current.x][current.y] = true;

                for (auto neighbor : get_neighbors(current, grid)) {
                    if (closedSet[neighbor.x][neighbor.y] || grid.get(neighbor)) {
                        continue;
                    }
                    auto tentativeGScore = gScore[current.x][current.y] + 1;
                    if (tentativeGScore < gScore[neighbor.x][neighbor.y]) {
                        cameFrom[neighbor.x][neighbor.y] = current;
                        gScore[neighbor.x][neighbor.y] = tentativeGScore;
                        fScore[neighbor.x][neighbor.y] = tentativeGScore + h(neighbor, end);
                    }
                    openSet.push({neighbor, fScore[neighbor.x][neighbor.y]});
                }
            }

            return std::make_pair(cameFrom, gScore);
        }
    }

    template<typename Vec2>
    std::optional<std::vector<Vec2>> find_path(MapGrid<Vec2, bool> const &grid, Vec2 const &start, Vec2 const &end) {
        auto[prev, dist] = astar(grid, start, end);

        if (dist[end.x][end.y] == INF) {
            return {};
        }

        std::vector<Vec2> result;
        Vec2 cur = end;
        while (cur != start) {
            result.emplace_back(cur);
            cur = prev[cur.x][cur.y];
        }
        result.emplace_back(start);
        return {result};
    }
}
#endif //PATHFINDER_ASTAR_H
