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
import tqdm

from agents import PedestrianCEIAgentPedestrianDynamics
from simulation.abstractsimmaster import AbstractSimMaster
import numpy as np


class OfflineSimMaster(AbstractSimMaster):
    def __init__(self, track, simulation_constants, file_name, save_to_mat_and_csv=True, verbose=True, disable_collisions=False):
        super().__init__(track, simulation_constants, file_name, save_to_mat_and_csv=save_to_mat_and_csv)
        self.verbose = verbose
        self.disable_collisions = disable_collisions

        if verbose:
            self._progress_bar = tqdm.tqdm()
        else:
            self._progress_bar = None

        self._stop = False

    def add_agent(self, key, controllable_object, agent):
        self._controllable_objects[key] = controllable_object
        self._agents[key] = agent
        self.agent_types[key] = type(agent)

        if type(agent) is PedestrianCEIAgentPedestrianDynamics:
            self.risk_bounds[key] = agent.risk_threshold
        else:
            self.risk_bounds[key] = None

        super().add_agent(key, controllable_object, agent)

    def start(self):
        self._store_current_status()

        while self.t <= self.max_time and not self._stop:
            self.do_time_step()
            self._t += self.dt
            self.time_index += 1
            if self.verbose:
                self._progress_bar.update()

        if not self._stop:
            self.end_state = "Time ran out"

        data_dict = self._save_to_file()
        return data_dict

    def do_time_step(self, reverse=False):
        for controllable_object, agent in zip(self._controllable_objects.values(), self._agents.values()):
            if controllable_object.use_discrete_inputs:
                controllable_object.set_discrete_input(agent.compute_discrete_input(self.dt / 1000.0))
            else:
                controllable_object.set_continuous_input(agent.compute_continuous_input(self.dt / 1000.0))

        # This for loop over agents is done twice because the models that compute the new input need the current state of other agents.
        # So plan first for all agents before applying the accelerations and calculating the new state
        for controllable_object, agent in zip(self._controllable_objects.values(), self._agents.values()):
            controllable_object.update_model(self.dt / 1000.0)

            if self._track.is_beyond_track_bounds(controllable_object.position):
                self.end_state = "Beyond track bounds"
                self._stop = True
            elif self._track.is_beyond_finish(controllable_object.position):
                self.end_state = "Finished"
                self._stop = True

        for key, outer_object in self._controllable_objects.items():
            position = outer_object.position
            for inner_key, inner_object in self._controllable_objects.items():
                if key != inner_key:
                    inner_object_position = inner_object.position
                    if np.linalg.norm(np.array(position) - np.array(inner_object_position)) < self.simulation_constants.collision_tolerance:
                        self.end_state = "Collided"
                        self._stop = True
                        continue
        self._store_current_status()
