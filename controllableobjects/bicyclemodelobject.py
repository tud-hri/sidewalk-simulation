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
from math import cos, sin, tan

import numpy as np

from .controlableobject import ControllableObject


class BicycleModelObject(ControllableObject):
    """
    A Bicycle model that uses the rear axle point to express its position and velocity
    """

    def __init__(self, wheelbase=2., initial_position=np.array([0.0, 0.0]), initial_velocity=3.0, initial_heading=np.pi / 2.0,
                 max_steering_angle=np.pi / 2, max_acceleration=2., resistance_coefficient=0.005, constant_resistance=0.5):
        self.wheelbase = wheelbase

        # initial_values
        self.initial_position = initial_position.copy()
        self.initial_velocity = initial_velocity
        self.initial_heading = initial_heading

        # state variables
        self.position = initial_position.copy()
        self.velocity = initial_velocity
        self.heading = initial_heading

        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration

        self.steering_angle = 0.0
        self.acceleration = 0.0

        self.resistance_coefficient = resistance_coefficient
        self.constant_resistance = constant_resistance

        self.use_discrete_inputs = False

    def update_model(self, dt):
        self.position, self.velocity, self.heading = self.calculate_time_step_2d(dt,
                                                                                 self.position,
                                                                                 self.velocity,
                                                                                 self.heading,
                                                                                 self.acceleration,
                                                                                 steering_angle=self.steering_angle,
                                                                                 wheelbase=self.wheelbase,
                                                                                 resistance_coefficient=self.resistance_coefficient,
                                                                                 constant_resistance=self.constant_resistance)

    @staticmethod
    def calculate_time_step_2d(dt, position, velocity, heading, acceleration,
                               steering_angle=0., wheelbase=None, resistance_coefficient=0., constant_resistance=0.):
        """
        A dynamic bicycle model of the kinematics of the center of mass of a bicycle.
        The center of mass is assumed to be at the center of the wheelbase.
        """
        nett_acceleration = acceleration - constant_resistance - resistance_coefficient * np.linalg.norm(velocity) ** 2

        beta = np.arctan(np.tan(steering_angle) / 2.)  # Slip angle
        x_dot = velocity * np.cos(heading + beta)
        y_dot = velocity * np.sin(heading + beta)
        a_x = nett_acceleration * np.cos(heading + beta)
        a_y = nett_acceleration * np.sin(heading + beta)
        turning_rate = (velocity / (.5 * wheelbase)) * np.sin(beta)

        new_position = position + np.array([x_dot, y_dot]) * dt + .5 * np.array([a_x, a_y]) * dt ** 2
        new_heading = heading + turning_rate * dt
        new_speed = velocity + nett_acceleration * dt
        new_speed = max(0., new_speed)
        return new_position, new_speed, new_heading

    @staticmethod
    def calculate_time_step_1d(dt, position, velocity, acceleration, resistance_coefficient, constant_resistance,
                               ignore_velocity_limits=False):
        raise NotImplementedError('the Bicycle Model only works in 2 dimensions')

    def reset_to_initial_values(self):
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        self.heading = 0.0

        # inputs
        self.steering_angle = 0.0
        self.acceleration = 0.0

    def set_continuous_input(self, control_input):
        acceleration, steering = control_input
        self.steering_angle = steering * self.max_steering_angle
        self.acceleration = acceleration * self.max_acceleration

    def set_discrete_input(self, control_input):
        acceleration, steering = control_input
        self.steering_angle = steering * self.max_steering_angle * 0.5
        self.acceleration = acceleration * self.max_acceleration * 0.5
