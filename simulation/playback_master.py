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

from agents import PedestrianCEIAgent
from controllableobjects.controlableobject import ControllableObject
from simulation.simmaster import SimMaster


class PlaybackMaster(SimMaster):
    def __init__(self, gui, track, simulation_constants, playback_data):
        super().__init__(gui, track, simulation_constants, file_name=None)

        self.playback_data = playback_data
        self.maxtime_index = len([p for p in playback_data['positions'][0] if p is not None]) - 1

    def add_agent(self, key, controllable_object: ControllableObject, agent=None):
        super(PlaybackMaster, self).add_agent(key, controllable_object, agent=agent)

        self._controllable_objects[key].position = self.playback_data['positions'][key][0]
        self._controllable_objects[key].velocity = self.playback_data['velocities'][key][0]

        self.gui.update_all_graphics()

    def set_time(self, time_promille):
        new_index = int((time_promille / 1000.) * self.maxtime_index)
        self.time_index = new_index - 1
        self._t = self.time_index * self.dt
        self.do_time_step()

    def initialize_plots(self):
        time = [(self.playback_data['dt'] / 1000.) * index for index in range(len(self.playback_data['velocities'][0]))]
        self.gui.initialize_plots(self.playback_data, time, 'b', 'r')

    def do_time_step(self, reverse=False):
        if reverse and self.time_index > 0:
            self._t -= self.dt
            self.time_index -= 1
            self.gui.show_overlay()
        elif not reverse and self.time_index < self.maxtime_index:
            self._t += self.dt
            self.time_index += 1
            self.gui.show_overlay()
        elif not reverse:
            if self.main_timer.isActive():
                self.gui.toggle_play()
            self.gui.show_overlay(self.playback_data['end_state'])
            if self._is_recording:
                self.gui.record_frame()
                self.gui.stop_recording()
            return

        self.gui.update_time_label(self.t / 1000.0)

        for side in self._controllable_objects.keys():
            self._controllable_objects[side].position = self.playback_data['positions'][side][self.time_index]
            self._controllable_objects[side].velocity = self.playback_data['velocities'][side][self.time_index]
            self._controllable_objects[side].heading = self.playback_data['headings'][side][self.time_index]

        self.gui.update_all_graphics()
        for agent in self.playback_data['agent_types'].keys():
            if agent in [0, 1]:
                self.gui.update_plots(agent,
                                      self.playback_data['beliefs'][agent][self.time_index],
                                      self.playback_data['belief_time_stamps'][agent][self.time_index],
                                      self.playback_data['position_plans'][agent][self.time_index])

        if self._is_recording:
            self.gui.record_frame()
