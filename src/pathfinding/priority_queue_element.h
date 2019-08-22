//
// Created by pulp on 22/08/2019.
//

#ifndef PATHFINDER_PRIORITY_QUEUE_ELEMENT_H
#define PATHFINDER_PRIORITY_QUEUE_ELEMENT_H

template<typename T>
struct PriorityQueueElement {
    PriorityQueueElement(T v, double dist): v{v},dist{dist} {};

    T getV() const { return v; };
    double getDist() const { return dist; };

private:
    T v;
    double dist;
};

template<typename T>
bool operator<(PriorityQueueElement<T> const &a, PriorityQueueElement<T> const &b) {
    return a.getDist() > b.getDist();
}

template<typename T>
using PriorityQueue = std::priority_queue<PriorityQueueElement<T>>;

#endif //PATHFINDER_PRIORITY_QUEUE_ELEMENT_H
