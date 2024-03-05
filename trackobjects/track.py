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
import abc

import numpy as np


class Track(abc.ABC):
    @abc.abstractmethod
    def is_beyond_track_bounds(self, position: np.ndarray) -> bool:
        pass

    @abc.abstractmethod
    def is_beyond_finish(self, position: np.ndarray) -> bool:
        pass

    @abc.abstractmethod
    def get_heading(self, position: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def closest_point_on_route(self, position: np.ndarray) -> (np.ndarray, float):
        pass

    @abc.abstractmethod
    def get_track_bounding_rect(self) -> (float, float, float, float):
        pass

    @abc.abstractmethod
    def get_way_points(self) -> list:
        pass

    @abc.abstractmethod
    def get_start_position(self, index) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def total_distance(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def track_width(self) -> float:
        pass
