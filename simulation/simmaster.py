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

from PyQt5 import QtCore

from agents import PedestrianCEIAgent
from agents.agent import Agent
from controllableobjects.controlableobject import ControllableObject
from simulation.abstractsimmaster import AbstractSimMaster
from gui import SimulationGui


class SimMaster(AbstractSimMaster):
    def __init__(self, gui, track, simulation_constants, *, file_name=None, sub_folder=None, save_to_mat_and_csv=True):
        super().__init__(track, simulation_constants, file_name, sub_folder=sub_folder, save_to_mat_and_csv=save_to_mat_and_csv)

        self.main_timer = QtCore.QTimer()
        self.main_timer.setInterval(self.dt)
        self.main_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.main_timer.setSingleShot(False)
        self.main_timer.timeout.connect(self.do_time_step)

        self.count_down_timer = QtCore.QTimer()
        self.count_down_timer.setInterval(1000)
        self.count_down_timer.timeout.connect(self.count_down)

        self.count_down_clock = 3  # counts down from 3
        self.history_length = 5

        self.gui = gui
        self.velocity_history = {}

    def start(self):
        self._store_current_status()
        self.count_down()
        self.count_down_timer.start()

    def pause(self):
        self.main_timer.stop()

    def count_down(self):
        if self.count_down_clock == 0:
            self.main_timer.start()
            self.count_down_timer.stop()
            self.gui.show_overlay()
        else:
            self.gui.show_overlay(str(self.count_down_clock))
            self.count_down_clock -= 1

    def reset(self):
        super().reset()

        self.count_down_clock = 3  # counts down from 3

        for key in self._agents.keys():
            self._controllable_objects[key].reset()
            self._agents[key].reset()
            self.velocity_history[key] = [self._controllable_objects[key].velocity] * self.history_length

        self.gui.update_all_graphics()
        self.gui.reset()

    def add_agent(self, key, controllable_object: ControllableObject, agent: Agent):
        self._controllable_objects[key] = controllable_object
        self._agents[key] = agent
        self.agent_types[key] = type(agent)
        if type(agent) is PedestrianCEIAgent:
            self.risk_bounds[key] = agent.risk_threshold

        self.velocity_history[key] = [controllable_object.velocity] * self.history_length
        super().add_agent(key, controllable_object, agent)

    def get_velocity_history(self, key):
        return self.velocity_history[key]

    def _update_history(self):
        for key in self._agents.keys():
            try:
                self.velocity_history[key] = [self._controllable_objects[key].velocity] + self.velocity_history[key][:-1]
            except KeyError:
                # no vehicle exists on that side
                pass

    def _end_simulation(self):
        self.gui.show_overlay(self.end_state)
        self._save_to_file()

        if self._is_recording:
            self.gui.record_frame()
            self.gui.stop_recording()

    def do_time_step(self, reverse=False):

        for controllable_object, tag in zip(self._controllable_objects.values(), self._agents.values()):
            if controllable_object.use_discrete_inputs:
                controllable_object.set_discrete_input(tag.compute_discrete_input(self.dt / 1000.0))
            else:
                controllable_object.set_continuous_input(tag.compute_continuous_input(self.dt / 1000.0))

        # This for loop over agents is done twice because the models that compute the new input need the current state of other vehicles.
        # So plan first for all vehicles before applying the accelerations and calculating the new state
        for controllable_object, tag in zip(self._controllable_objects.values(), self._agents.values()):
            controllable_object.update_model(self.dt / 1000.0)

            if self._track.is_beyond_track_bounds(controllable_object.position):
                self.main_timer.stop()
                self.end_state = "Beyond track bounds"
            elif self._track.is_beyond_finish(controllable_object.position):
                self.main_timer.stop()
                self.end_state = "Finished"

        for key, outer_object in self._controllable_objects.items():
            position = outer_object.position
            for inner_key, inner_object in self._controllable_objects.items():
                if key != inner_key:
                    inner_object_position = inner_object.position
                    if np.linalg.norm(np.array(position) - np.array(inner_object_position)) < self.simulation_constants.collision_tolerance:
                        self.main_timer.stop()
                        self.end_state = "Collided"
                        continue

        self._update_history()
        self.gui.update_all_graphics()

        for tag in [0, 1]:
            try:
                agent = self._agents[tag]
                self.gui.update_plots(tag, agent.belief, agent.belief_time_stamps, agent.position_plan)
            except KeyError:
                pass # the agent does not exist in this simulation

        self.gui.update_time_label(self.t / 1000.0)
        self._t += self.dt
        self.time_index += 1
        self._store_current_status()

        if self._t >= self.max_time:
            self.main_timer.stop()
            self.end_state = self.end_state = "Time ran out"

        if self.end_state != 'Not finished':
            self._end_simulation()

        if self._is_recording:
            self.gui.record_frame()
