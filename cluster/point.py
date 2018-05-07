# encoding: utf-8
import math


class Point:

    def __init__(self, latitude, longitude):

        self.latitude = latitude
        self.longitude = longitude
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False  # has this point been processed

    def distance(self, point):
        """calculate the distance between any two points on earth
        """

        p1_lat, p1_lon, p2_lat, p2_lon = [math.radians(c) for c in
                                          (self.latitude, self.longitude, point.latitude, point.longitude)]

        numerator = math.sqrt(
            math.pow(math.cos(p2_lat) * math.sin(p2_lon - p1_lon), 2) +
            math.pow(
                math.cos(p1_lat) * math.sin(p2_lat) -
                math.sin(p1_lat) * math.cos(p2_lat) *
                math.cos(p2_lon - p1_lon), 2))

        denominator = (
            math.sin(p1_lat) * math.sin(p2_lat) +
            math.cos(p1_lat) * math.cos(p2_lat) *
            math.cos(p2_lon - p1_lon))

        # convert distance from radians to meters
        # note: earth's radius ~ 6372800 meters
        return math.atan2(numerator, denominator) * 6372800
