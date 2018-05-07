# encoding: utf-8

'''
密度聚类算法： OPTICS
'''

from point import Point


class Cluster:

    def __init__(self, points):
        self.points = points

    def centroid(self):
        """calculate the centroid for the cluster
        """
        return Point(sum([p.latitude for p in self.points])/len(self.points),
            sum([p.longitude for p in self.points])/len(self.points))

    def region(self):
        """calculate the region (centroid, bounding radius) for the cluster
        """
        centroid = self.centroid()
        radius = reduce(lambda r, p: max(r, p.distance(centroid)), self.points)
        return centroid, radius


class Optics:

    def __init__(self, points, max_radius, min_cluster_size):
        self.points = points
        self.max_radius = max_radius                # 半径
        self.min_cluster_size = min_cluster_size    # 最小聚簇数目

    def setup(self):
        """get ready for a clustering run
        """
        for p in self.points:
            p.rd = None  # 可达距离
            p.processed = False
        self.unprocessed = [p for p in self.points]
        self.ordered = []

    def core_distance(self, point, neighbors):
        """distance from a point to its nth neighbor (n = min_cluser_size)
        """
        if point.cd is not None: return point.cd
        if len(neighbors) >= self.min_cluster_size - 1:
            sortedneighbors = sorted([n.distance(point) for n in neighbors])
            point.cd = sortedneighbors[self.min_cluster_size - 2]
            return point.cd

    def neighbors(self, point):
        """neighbors for a point within max_radius
        """
        return [p for p in self.points if p is not point and p.distance(point) <= self.max_radius]

    def processed(self, point):
        """mark a point as processed
        """
        point.processed = True
        self.unprocessed.remove(point)
        self.ordered.append(point)

    def update(self, neighbors, point, seeds):
        """update seeds if a smaller reachability distance is found
        """
        for n in [n for n in neighbors if not n.processed]:
            # find new reachability distance new_rd
            # if rd is null, keep new_rd and add n to the seed list
            # otherwise if new_rd < old rd, update rd
            new_rd = max(point.cd, point.distance(n))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

    def run(self):
        self.setup()
        # for each unprocessed point (p)...

        while self.unprocessed:
            point = self.unprocessed[0]
            # mark p as processed
            # find p's neighbors
            self.processed(point)
            pointneighbors = self.neighbors(point)
            # if p has a core_distance, i.e has min_cluster_size - 1 neighbors
            if self.core_distance(point, pointneighbors) is not None:
                # update reachability_distance for each unprocessed neighbor
                seeds = []
                self.update(pointneighbors, point, seeds)
                # as long as we have unprocessed neighbors...
                while(seeds):
                    # find the neighbor n with smallest reachability distance
                    seeds.sort(key=lambda n: n.rd)
                    n = seeds.pop(0)
                    # mark n as processed
                    # find n's neighbors
                    self.processed(n)
                    nneighbors = self.neighbors(n)
                    # if p has a core_distance...
                    if self.core_distance(n, nneighbors) is not None:
                        # update reachability_distance for each of n's neighbors
                        self.update(nneighbors, n, seeds)

        # when all points have been processed
        # return the ordered list
        return self.ordered

    def cluster(self, cluster_threshold):
        clusters = []
        separators = []

        for i in range(len(self.ordered)):
            this_i = i
            next_i = i + 1
            this_p = self.ordered[i]
            this_rd = this_p.rd if this_p.rd else float('infinity')
            # use an upper limit to separate the clusters
            if this_rd > cluster_threshold:
                separators.append(this_i)

        separators.append(len(self.ordered))

        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(Cluster(self.ordered[start:end]))

        return clusters


points = [
    Point(37.769006, -122.429299), # cluster #1
    Point(37.769044, -122.429130), # cluster #1
    Point(37.768775, -122.429092), # cluster #1
    Point(37.776299, -122.424249), # cluster #2
    Point(37.776265, -122.424657), # cluster #2
]

optics = Optics(points, 100, 2) # 100m radius for neighbor consideration, cluster size >= 2 points
optics.run()                    # run the algorithm
clusters = optics.cluster(50)   # 50m threshold for clustering

for cluster in clusters:
    print(cluster.points)
