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
import sys

from PyQt5 import QtWidgets, QtCore
import numpy as np

from agents import PedestrianCEIAgentPedestrianDynamics
from controllableobjects import PedestrianObject
from gui import SimulationGui
from simulation.simmaster import SimMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import SideWalk

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    simulation_constants = SimulationConstants(dt=50,
                                               sidewalk_width=2.5,
                                               sidewalk_length=15.,
                                               collision_tolerance=0.25,
                                               max_time=30e3)

    time_horizon = 7.
    belief_frequency = 4
    preferred_velocity = 1.3

    track = SideWalk(simulation_constants)
    gui = SimulationGui(track, number_of_belief_points=int(time_horizon * belief_frequency))
    sim_master = SimMaster(gui, track, simulation_constants, file_name='online_simulation')

    cultural_biases = [1., 1.]
    risk_thresholds = [0.7, 0.7]
    initial_positions = [track.get_start_position(0),
                         track.get_start_position(1)]

    first_pedestrian_object = PedestrianObject(initial_position=initial_positions[0],
                                               initial_velocity=np.array([preferred_velocity, 0.]),
                                               initial_angular_velocity=0.0,
                                               initial_heading=np.pi / 2.0,
                                               max_lat_acceleration=1.,
                                               max_angular_acceleration=np.pi)

    second_pedestrian_object = PedestrianObject(initial_position=initial_positions[1],
                                                initial_velocity=np.array([preferred_velocity, 0.]),
                                                initial_angular_velocity=0.0,
                                                initial_heading=-np.pi / 2.0,
                                                max_lat_acceleration=1.,
                                                max_angular_acceleration=np.pi)

    cei_agent_one = PedestrianCEIAgentPedestrianDynamics(controllable_object=first_pedestrian_object,
                                                         dt=simulation_constants.dt,
                                                         sim_master=sim_master,
                                                         track=track,
                                                         risk_threshold=risk_thresholds[0],
                                                         time_horizon=time_horizon,
                                                         planning_frequency=belief_frequency,
                                                         preferred_velocity=preferred_velocity,
                                                         preferred_heading=np.pi / 2,
                                                         comfortable_range=.4,
                                                         opponent_id=1,
                                                         cultural_bias=cultural_biases[0])
    cei_agent_two = PedestrianCEIAgentPedestrianDynamics(controllable_object=second_pedestrian_object,
                                                         dt=simulation_constants.dt,
                                                         sim_master=sim_master,
                                                         track=track,
                                                         risk_threshold=risk_thresholds[1],
                                                         time_horizon=time_horizon,
                                                         planning_frequency=belief_frequency,
                                                         preferred_velocity=preferred_velocity,
                                                         preferred_heading=-np.pi / 2,
                                                         comfortable_range=.4,
                                                         opponent_id=0,
                                                         cultural_bias=cultural_biases[1]
                                                         )

    sim_master.add_agent(0, first_pedestrian_object, cei_agent_one)
    sim_master.add_agent(1, second_pedestrian_object, cei_agent_two)

    gui.add_controllable_dot(first_pedestrian_object, color=QtCore.Qt.red)
    gui.add_controllable_dot(second_pedestrian_object, color=QtCore.Qt.green)

    gui.register_sim_master(sim_master)
    sys.exit(app.exec_())
