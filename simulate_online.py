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

from agents import PedestrianCEIAgent
from controllableobjects import BicycleModelObject
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

    track = SideWalk(simulation_constants)
    gui = SimulationGui(track, number_of_belief_points=int(time_horizon * belief_frequency))
    sim_master = SimMaster(gui, track, simulation_constants, file_name='online_simulation')

    first_bicycle_object = BicycleModelObject(initial_position=track.get_start_position(0),
                                              initial_velocity=1.3,
                                              wheelbase=0.1)

    second_bicycle_object = BicycleModelObject(initial_position=track.get_start_position(1),
                                               initial_velocity=1.3,
                                               initial_heading=-np.pi / 2.0,
                                               wheelbase=0.1)

    # Comfortable range is +-0.7 m (Kim 2013, Gorrini 2014)
    cei_agent_one = PedestrianCEIAgent(controllable_object=first_bicycle_object,
                                       dt=simulation_constants.dt,
                                       sim_master=sim_master,
                                       track=track,
                                       risk_threshold=0.7,
                                       time_horizon=time_horizon,
                                       planning_frequency=belief_frequency,
                                       preferred_velocity=1.3,
                                       preferred_heading=np.pi/2,
                                       comfortable_range=.3,
                                       opponent_id=1,
                                       cultural_bias=1.)
    cei_agent_two = PedestrianCEIAgent(controllable_object=second_bicycle_object,
                                       dt=simulation_constants.dt,
                                       sim_master=sim_master,
                                       track=track,
                                       risk_threshold=0.5,
                                       time_horizon=time_horizon,
                                       planning_frequency=belief_frequency,
                                       preferred_velocity=1.3,
                                       preferred_heading=-np.pi / 2,
                                       comfortable_range=.3,
                                       opponent_id=0,
                                       cultural_bias=1.
                                       )

    gui.add_controllable_dot(first_bicycle_object, color=QtCore.Qt.red)
    sim_master.add_agent(0, first_bicycle_object, cei_agent_one)

    gui.add_controllable_dot(second_bicycle_object, color=QtCore.Qt.green)
    sim_master.add_agent(1, second_bicycle_object, cei_agent_two)

    gui.register_sim_master(sim_master)
    sys.exit(app.exec_())
