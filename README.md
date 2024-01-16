# RouteMaster
# Trip Planner Project - Implementation Analysis

## Introduction

This document provides a detailed analysis of the design and implementation choices made for the Trip Planner project in Comp Sci 214, focusing on the selection and justification of specific Abstract Data Types (ADTs) and algorithms.

## ADTs (Abstract Data Types)

### Priority Queue
- **ADT Name**: Priority Queue
- **Role**: Critical for implementing Dijkstra's algorithm, managing vertices during shortest path calculation.
- **Chosen Data Structure**: Binary Heap
- **Rationale**: Optimal performance for insertions and removals, crucial for handling large graphs in Dijkstra's algorithm.

### Graph
- **ADT Name**: Graph
- **Role**: Backbone of the Trip Planner, representing network of locations and connections.
- **Chosen Data Structure**: Weighted Undirected Graph with Adjacency Lists
- **Rationale**: Efficient for sparse road networks, quick traversal, and realistic route planning.

### Hash Table
- **ADT Name**: Hash Table
- **Role**: Storing and accessing POIs efficiently.
- **Chosen Data Structure**: Custom Hash Table with Separate Chaining
- **Rationale**: Effective collision management, consistent access times, suitable for diverse POI categories.

### Linked List
- **ADT Name**: Linked List
- **Role**: Dynamic data storage and manipulation, crucial for routes and POIs updates.
- **Chosen Data Structure**: Singly Linked List
- **Rationale**: Efficient insertion and deletion, flexible and simple for the project's needs.

## Algorithms

### Dijkstra's Algorithm
- **Role**: Finding shortest paths in the graph for route planning.
- **Rationale**: Efficient for weighted graphs, performance advantage over other algorithms like Bellman-Ford.

### Modified Dijkstra's Algorithm for Finding Nearby POIs
- **Role**: Finding nearest POIs within a certain category.
- **Rationale**: Optimized for finding nearby POIs, terminates early once the required number is found.

## Comparative Analysis of Data Structures and Algorithm Choices

This section includes a comparative analysis of chosen data structures and algorithms against their alternatives, discussing the trade-offs and justifications for each choice.

- **Graph (WUGraph) vs. Other Representations**
  - Justification: Balances space efficiency and fast traversal needs.

- **HashTable with Separate Chaining vs. Open Addressing**
  - Justification: Better performance in high collision scenarios, suitable for diverse POI categories.

- **Priority Queue (Binary Heap) vs. Other Heaps**
  - Justification: Good balance between efficiency and implementation complexity.

## Complexity Analysis

### Dijkstra's Algorithm with Binary Heap
- **Time Complexity**: O((V + E) log V)
- **Space Complexity**: O(V)

### Modified Dijkstra's Algorithm for Nearby POIs
- **Time Complexity**: Generally performs better than standard Dijkstra's
- **Space Complexity**: Similar to the standard Dijkstra's algorithm

## Conclusion

This analysis explored critical design decisions in the development of the Trip Planner project. Each data structure and algorithm was chosen
