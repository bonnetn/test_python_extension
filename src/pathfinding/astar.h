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

            auto openSet = PriorityQueue<Vec2>{};
            openSet.push({start, 0});
            auto closedSet = Array2D<bool>(grid.dimensions().x, std::vector<bool>(grid.dimensions().y, false));

            auto cameFrom = Array2D<Vec2>(grid.dimensions().x, std::vector<Vec2>(grid.dimensions().y));

            auto gScore = Array2D<double>(grid.dimensions().x, std::vector<double>(grid.dimensions().y, INF));
            gScore[start.x][start.y] = 0;

            auto fScore = Array2D<double>(grid.dimensions().x, std::vector<double>(grid.dimensions().y, INF));
            fScore[start.x][start.y] = h(start, end);

            while (!openSet.empty()) {
                auto current = openSet.top().v;
                if (current == end) {
                    return std::make_pair(cameFrom, gScore);
                }

                openSet.pop();
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
