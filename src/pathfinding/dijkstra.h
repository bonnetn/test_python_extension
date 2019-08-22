#ifndef PATHFINDER_DIJKSTRA_H
#define PATHFINDER_DIJKSTRA_H


#include <limits>
#include <queue>
#include <optional>
#include "priority_queue_element.h"

namespace dijkstra {

    template<typename T>
    using Array2D = std::vector<std::vector<T>>;

    template<typename T>
    using ObstacleMap = MapGrid<T, bool>;

    constexpr double INF = std::numeric_limits<double>::infinity();

    template<typename Vec2>
    std::pair<Array2D<Vec2>, Array2D<double>> dijkstra(ObstacleMap<Vec2> grid, Vec2 start, Vec2 end) {
        /*
         * From https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm :
    1  function Dijkstra(Graph, source):
    2      dist[source] ← 0                           // Initialization
    3
    4      create vertex priority queue Q
    5
    6      for each vertex v in Graph:
    7          if v ≠ source
    8              dist[v] ← INFINITY                 // Unknown distance from source to v
    9          prev[v] ← UNDEFINED                    // Predecessor of v
    10
    11         Q.add_with_priority(v, dist[v])
    12
    13
    14     while Q is not empty:                      // The main loop
    15         u ← Q.extract_min()                    // Remove and return best vertex
    16         for each neighbor v of u:              // only v that are still in Q
    17             alt ← dist[u] + length(u, v)
    18             if alt < dist[v]
    19                 dist[v] ← alt
    20                 prev[v] ← u
    21                 Q.decrease_priority(v, alt)
    22
    23     return dist, prev     */
        auto dist = Array2D<double>(grid.dimensions().x, std::vector<double>(grid.dimensions().y, INF));
        auto prev = Array2D<Vec2>(grid.dimensions().y, std::vector<Vec2>(grid.dimensions().y));


        PriorityQueue<Vec2> Q;
        Q.push({.v=start, .dist=0});

        dist[start.x][start.y] = 0;

        while (!Q.empty()) {
            auto element = Q.top();
            auto u = element.v;
            Q.pop();
            if (element.dist != dist[u.x][u.y])
                continue;

            for (auto v: get_neighbors(u, grid)) {
                if (grid.get(v))
                    continue;

                auto alt = dist[u.x][u.y] + distance(u, v);
                if (alt < dist[v.x][v.y]) {
                    dist[v.x][v.y] = alt;
                    prev[v.x][v.y] = u;
                    Q.push({.v=v, .dist=alt});

                    if (v == end) {
                        return std::make_pair(prev, dist);
                    }
                }
            }
        }
        return std::make_pair(prev, dist);
    }

    template<typename Vec2>
    std::optional<std::vector<Vec2>> find_path(ObstacleMap<Vec2> grid, Vec2 start, Vec2 end) {
        auto[prev, dist] = dijkstra(grid, start, end);

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

#endif //PATHFINDER_DIJKSTRA_H
