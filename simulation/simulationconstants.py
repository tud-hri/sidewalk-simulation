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


class SimulationConstants:
    """ object that stores all constants needed to recall a saved simulation. """

    def __init__(self, dt, sidewalk_width, sidewalk_length, collision_tolerance, max_time):
        self.dt = dt
        self.sidewalk_width = sidewalk_width
        self.sidewalk_length = sidewalk_length
        self.collision_tolerance = collision_tolerance
        self.max_time = max_time
