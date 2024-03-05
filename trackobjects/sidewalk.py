"""
Copyright 2024, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of sidewalk-simulation.

sidewalk-simulation is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

sidewalk-simulation is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with sidewalk-simulation.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np

from trackobjects import Track


class SideWalk(Track):
    def __init__(self, simulation_constants):
        self._width = simulation_constants.sidewalk_width
        self._length = simulation_constants.sidewalk_length

        self._end_point = np.array([0.0, self._length])
        self._way_points = [np.array([0.0, 0.0]), self._end_point]
        self._start_points = self._way_points

    def is_beyond_track_bounds(self, position):
        beyond_side = abs(position[0]) > self._width/2.
        return beyond_side

    def is_beyond_finish(self, position):
        return position[1] > self._length or position[1] < 0.

    @staticmethod
    def get_heading(*args):
        return np.pi / 2

    @staticmethod
    def closest_point_on_route(position):
        closest_point_on_route = np.array([.0, position[1]])
        shortest_distance = abs(position[0])

        return closest_point_on_route, shortest_distance

    def get_track_bounding_rect(self) -> (float, float, float, float):
        x1 = - 2 * self._width
        x2 = 2 * self._width

        y1 = 0.
        y2 = self._length

        return x1, y1, x2, y2

    def get_way_points(self) -> list:
        return self._way_points

    def get_start_position(self, index) -> np.ndarray:
        return self._start_points[index]

    @property
    def total_distance(self) -> float:
        return self._length

    @property
    def track_width(self) -> float:
        return self._width
