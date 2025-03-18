import math


class Graph:
    def __init__(self, size, directed):
        # initialize the graph instance with the specified size and directed flag
        self.size = size
        self.edges = {}  # dictionary to store edges
        self.nodes = {}  # dictionary to store nodes
        self.start_city = None  # variable to store the start city for TSP
        self.directed = directed  # flag indicating if the graph is directed or not

    def add_edge(self, a, b, weight=1):
        # add an edge between nodes 'a' and 'b' with an optional weight
        self.edges.setdefault(a, []).append((b, weight))
        if not self.directed:
            # if the graph is undirected, add the reverse edge as well
            self.edges.setdefault(b, []).append((a, weight))

    def add_node(self, a, x, y):
        # add a node 'a' with coordinates (x, y) to the graph
        if a not in self.edges:
            self.nodes[a] = (x, y)  # store node coordinates in the nodes dictionary
            distances = {key: self.euclidean_distance(a, key) for key in self.nodes if key != a}
            for key, distance in distances.items():
                # add edges from the new node to existing nodes with distances based on Euclidean distance
                self.add_edge(a, key, distance)

    def euclidean_distance(self, a, b):
        # calculate Euclidean distance between nodes 'a' and 'b'
        x1, y1 = self.nodes[a]
        x2, y2 = self.nodes[b]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return round(distance, 2)

    def vertices(self):
        # return a list of vertices (nodes) in the graph
        return list(self.edges.keys())

    def getPathCost(self, path, incl_return_distance=False):
        # calculate the total cost of a given path in the graph
        pairs = zip(path, path[1:])
        cost = sum(self.euclidean_distance(city1, city2) for city1, city2 in pairs)

        if incl_return_distance:
            # if incl_return_distance is True, include the return distance to the starting city
            cost += self.euclidean_distance(path[0], path[-1])

        return cost
