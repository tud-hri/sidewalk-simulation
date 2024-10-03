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

from .controlableobject import ControllableObject


class PedestrianObject(ControllableObject):
    """
    A pedestrian dynamics model based on Mombaur et al. 2008
    """

    def __init__(self, initial_position=np.array([0.0, 0.0]), initial_velocity=np.array([3.0, 0.0]), initial_angular_velocity=0.0, initial_heading=np.pi / 2.0,
                 max_angular_acceleration=np.pi / 2, max_long_acceleration=2., max_lat_acceleration=2.):
        # initial_values
        self.initial_position = initial_position.copy()
        self.initial_velocity = initial_velocity.copy()
        self.initial_heading = initial_heading

        # state variables
        self.position = initial_position.copy()  # [x, y] in world frame
        self.velocity = initial_velocity.copy()  # [longitudinal, lateral] in pedestrian frame
        self.angular_velocity = initial_angular_velocity
        self.heading = initial_heading

        self.max_angular_acceleration = max_angular_acceleration
        self.max_long_acceleration = max_long_acceleration
        self.max_lat_acceleration = max_lat_acceleration

        self.angular_acceleration = 0.0
        self.long_acceleration = 0.0
        self.lat_acceleration = 0.0

        self.use_discrete_inputs = False

    def update_model(self, dt):
        self.position, self.velocity, self.angular_velocity, self.heading = self.calculate_time_step_2d(dt,
                                                                                                        self.position,
                                                                                                        self.velocity,
                                                                                                        self.heading,
                                                                                                        self.angular_velocity,
                                                                                                        self.long_acceleration,
                                                                                                        self.lat_acceleration,
                                                                                                        self.angular_acceleration)

    @staticmethod
    def calculate_time_step_2d(dt, position, velocity, heading, angular_velocity, long_acceleration, lat_acceleration, angular_acceleration=0.):
        """
        A dynamic pedestrian model of the kinematics of the trunk of a pedestrian.
        Based on Mombaur et al. 2008
        """

        x_dot = velocity[0] * np.cos(heading) - velocity[1] * np.sin(heading)
        y_dot = velocity[0] * np.sin(heading) + velocity[1] * np.cos(heading)
        angular_v_dot = angular_acceleration

        x_dot_dot = long_acceleration * np.cos(heading) - lat_acceleration * np.sin(heading)
        y_dot_dot = long_acceleration * np.sin(heading) + lat_acceleration * np.cos(heading)

        new_position = position + np.array([x_dot, y_dot]) * dt + .5 * np.array([x_dot_dot, y_dot_dot]) * dt ** 2
        new_heading = heading + angular_velocity * dt + 0.5 * angular_v_dot * dt ** 2
        new_speed = velocity + np.array([long_acceleration, lat_acceleration]) * dt
        new_angular_velocity = angular_velocity + angular_acceleration * dt

        return new_position, new_speed, new_angular_velocity, new_heading

    @staticmethod
    def calculate_time_step_1d(dt, position, velocity, acceleration, resistance_coefficient, constant_resistance,
                               ignore_velocity_limits=False):
        raise NotImplementedError('the Pedestrian Model only works in 2 dimensions')

    def reset_to_initial_values(self):
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        self.heading = 0.0

        # inputs
        self.angular_acceleration = 0.0
        self.long_acceleration = 0.0
        self.lat_acceleration = 0.0

    def set_continuous_input(self, control_input):
        long_acceleration, lat_acceleration, omega_dot = control_input

        self.angular_acceleration = omega_dot * self.max_angular_acceleration
        self.long_acceleration = long_acceleration * self.max_long_acceleration
        self.lat_acceleration = lat_acceleration * self.max_lat_acceleration

    def set_discrete_input(self, control_input):
        pass

    @property
    def acceleration(self):
        return np.array([self.long_acceleration, self.lat_acceleration, self.angular_acceleration])
