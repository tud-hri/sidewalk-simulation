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
import os
import pickle
import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from controllableobjects import PedestrianObject
from simulation.playback_master import PlaybackMaster
from simulation.simulationconstants import SimulationConstants
from gui import SimulationGui

simulation_constants: SimulationConstants


def playback(file_name):
    app = QtWidgets.QApplication(sys.argv)
    with open(os.path.join('data', file_name), 'rb') as f:
        playback_data = pickle.load(f)

    simulation_constants = playback_data['simulation_constants']

    dt = simulation_constants.dt  # ms
    track = playback_data['track']

    gui = SimulationGui(track, in_replay_mode=True, number_of_belief_points=len(playback_data['beliefs'][0][0]))
    sim_master = PlaybackMaster(gui, track, simulation_constants, playback_data)

    first_pedestrian_object =  PedestrianObject(initial_position=track.get_start_position(0), initial_heading=np.pi / 2.0)

    second_pedestrian_object =  PedestrianObject(initial_position=track.get_start_position(1), initial_heading=-np.pi / 2.0)

    gui.add_controllable_dot(first_pedestrian_object, color=QtGui.QColor(255, 127, 14))
    sim_master.add_agent(0, first_pedestrian_object)

    gui.add_controllable_dot(second_pedestrian_object, color=QtGui.QColor(31, 119, 180))
    sim_master.add_agent(1, second_pedestrian_object)

    gui.register_sim_master(sim_master)
    app.exec_()


if __name__ == '__main__':
    condition = 'symmetric'
    iteration = '0'

    file = os.path.join('simulations', condition, iteration + '.pkl')
    playback(file)
