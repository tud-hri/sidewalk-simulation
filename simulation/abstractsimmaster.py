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
import copy
import csv
import os
import pickle

import numpy as np
import scipy.io

from agents import PedestrianCEIAgent
from controllableobjects import BicycleModelObject
from simulation.simulationconstants import SimulationConstants


class AbstractSimMaster(abc.ABC):
    def __init__(self, track, simulation_constants, file_name=None, sub_folder=None, save_to_mat_and_csv=True):
        self.simulation_constants = simulation_constants

        self._controllable_objects = {}
        self._agents = {}

        self._t = 0.  # [ms]
        self.time_index = 0
        self.dt = simulation_constants.dt
        self.max_time = simulation_constants.max_time

        self._track = track

        self._file_name = file_name
        self._save_to_mat_and_csv = save_to_mat_and_csv
        self._sub_folder = sub_folder
        self.end_state = 'Not finished'
        self.agent_types = {}

        self._is_recording = False

        # dicts for saving to file and a list that contains all attributes of the sim master object that will be saved
        self.beliefs = {}
        self.observed_velocities = {}
        self.perceived_risks = {}
        self.is_replanning = {}
        self.position_plans = {}
        self.action_plans = {}
        self.positions = {}
        self.raw_input = {}
        self.velocities = {}
        self.headings = {}
        self.steering_angles = {}
        self.accelerations = {}
        self.net_accelerations = {}
        self.belief_time_stamps = {}
        self.belief_point_contributing_to_risk = {}
        self.risk_bounds = {}

        self._attributes_to_save = ['dt', 'max_time', 'simulation_constants', 'agent_types', 'end_state',
                                    'beliefs', 'observed_velocities', 'perceived_risks', 'is_replanning', 'position_plans', 'action_plans', 'positions',
                                    'raw_input', 'velocities', 'headings', 'steering_angles', 'accelerations', 'net_accelerations', 'belief_time_stamps',
                                    'belief_point_contributing_to_risk', 'risk_bounds']

    def reset(self):
        self._t = 0.  # [ms]
        self.time_index = 0
        self.end_state = 'Not finished'

        self.beliefs = {}
        self.observed_velocities = {}
        self.perceived_risks = {}
        self.is_replanning = {}
        self.position_plans = {}
        self.action_plans = {}
        self.positions = {}
        self.raw_input = {}
        self.velocities = {}
        self.accelerations = {}
        self.headings = {}
        self.steering_angles = {}
        self.net_accelerations = {}
        self.belief_time_stamps = {}
        self.belief_point_contributing_to_risk = {}
        self.risk_bounds = {}

        self._attributes_to_save = ['dt', 'max_time', 'simulation_constants', 'vehicle_width', 'vehicle_length', 'agent_types', 'end_state',
                                    'beliefs', 'observed_velocities', 'perceived_risks', 'is_replanning', 'position_plans', 'action_plans', 'positions',
                                     'raw_input', 'velocities', 'headings', 'steering_angles', 'accelerations', 'net_accelerations', 'belief_time_stamps',
                                    'belief_point_contributing_to_risk', 'risk_bounds']

        number_of_time_steps = int(self.simulation_constants.max_time / self.simulation_constants.dt)
        for key in self._agents.keys():
            self.beliefs[key] = [None] * number_of_time_steps
            self.observed_velocities[key] = [None] * number_of_time_steps
            self.position_plans[key] = [None] * number_of_time_steps
            self.action_plans[key] = [None] * number_of_time_steps
            self.perceived_risks[key] = [None] * number_of_time_steps
            self.is_replanning[key] = [None] * number_of_time_steps
            self.positions[key] = [None] * number_of_time_steps
            self.raw_input[key] = [None] * number_of_time_steps
            self.velocities[key] = [None] * number_of_time_steps
            self.headings[key] = [None] * number_of_time_steps
            self.steering_angles[key] = [None] * number_of_time_steps
            self.accelerations[key] = [None] * number_of_time_steps
            self.net_accelerations[key] = [None] * number_of_time_steps
            self.belief_time_stamps[key] = [None] * number_of_time_steps
            self.belief_point_contributing_to_risk[key] = [None] * number_of_time_steps

    @abc.abstractmethod
    def do_time_step(self, reverse=False):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def add_agent(self, key, controllable_object, agent):
        number_of_time_steps = int(self.simulation_constants.max_time / self.simulation_constants.dt) + 1
        self.beliefs[key] = [None] * number_of_time_steps
        self.observed_velocities[key] = [None] * number_of_time_steps
        self.position_plans[key] = [None] * number_of_time_steps
        self.action_plans[key] = [None] * number_of_time_steps
        self.perceived_risks[key] = [None] * number_of_time_steps
        self.is_replanning[key] = [None] * number_of_time_steps
        self.positions[key] = [None] * number_of_time_steps
        self.raw_input[key] = [None] * number_of_time_steps
        self.velocities[key] = [None] * number_of_time_steps
        self.headings[key] = [None] * number_of_time_steps
        self.steering_angles[key] = [None] * number_of_time_steps
        self.accelerations[key] = [None] * number_of_time_steps
        self.net_accelerations[key] = [None] * number_of_time_steps
        self.belief_time_stamps[key] = [None] * number_of_time_steps
        self.belief_point_contributing_to_risk[key] = [None] * number_of_time_steps

    def get_current_state(self, key):
        try:
            return self._controllable_objects[key].position, self._controllable_objects[key].velocity, self._controllable_objects[key].heading
        except KeyError:
            # no vehicle exists on that key
            return None, None, None

    def enable_recording(self, boolean):
        self._is_recording = boolean

    @property
    def t(self):
        return self._t

    def _store_current_status(self):
        for key in self._agents.keys():
            if self.agent_types[key] is PedestrianCEIAgent:
                self.beliefs[key][self.time_index] = copy.deepcopy(self._agents[key].belief)
                self.observed_velocities[key][self.time_index] = self._agents[key].observed_velocity
                self.action_plans[key][self.time_index] = copy.deepcopy(self._agents[key].action_plan)
                self.position_plans[key][self.time_index] = copy.deepcopy(self._agents[key].position_plan)
                self.perceived_risks[key][self.time_index] = copy.deepcopy(self._agents[key].perceived_risk)
                self.is_replanning[key][self.time_index] = copy.deepcopy(self._agents[key].did_plan_update_on_last_tick)
                self.belief_time_stamps[key][self.time_index] = copy.deepcopy(self._agents[key].belief_time_stamps)
                try:
                    self.belief_point_contributing_to_risk[key][self.time_index] = copy.deepcopy(self._agents[key].belief_point_contributing_to_risk)
                except AttributeError:
                    self.belief_point_contributing_to_risk[key][self.time_index] = None

            self.positions[key][self.time_index] = self._controllable_objects[key].position
            self.velocities[key][self.time_index] = self._controllable_objects[key].velocity
            self.accelerations[key][self.time_index] = self._controllable_objects[key].acceleration
            self.net_accelerations[key][self.time_index] = self._controllable_objects[key].acceleration - self._controllable_objects[key].resistance_coefficient * \
                                                            self._controllable_objects[key].velocity ** 2 - self._controllable_objects[key].constant_resistance
            if isinstance(self._controllable_objects[key], BicycleModelObject):
                self.raw_input[key][self.time_index] = [self._controllable_objects[key].acceleration /
                                                        self._controllable_objects[key].max_acceleration,
                                                        self._controllable_objects[key].steering_angle /
                                                        self._controllable_objects[key].max_steering_angle]
                self.headings[key][self.time_index] = self._controllable_objects[key].heading
                self.steering_angles[key][self.time_index] = self._controllable_objects[key].steering_angle
            else:
                self.raw_input[key][self.time_index] = (self._controllable_objects[key].acceleration /
                                                        self._controllable_objects[key].max_acceleration)

    def _save_to_file(self, file_name_extension=''):
        save_dict = {}
        for variable_name in self._attributes_to_save:
            variable_to_save = self.__getattribute__(variable_name)
            if isinstance(variable_to_save, dict):
                for key in self._agents.keys():
                    try:
                        if isinstance(variable_to_save[key], list):
                            variable_to_save[key] = [value for value in variable_to_save[key] if value is not None]
                    except KeyError:
                        pass

            save_dict[variable_name] = variable_to_save

        if self._file_name is not None:
            if self._sub_folder:
                folder = os.path.join('data', self._sub_folder)
            else:
                folder = 'data'

            os.makedirs(folder, exist_ok=True)

            pkl_file_name = os.path.join(folder, self._file_name + file_name_extension + '.pkl')
            csv_file_name = os.path.join(folder, self._file_name + file_name_extension + '.csv')
            mat_file_name = os.path.join(folder, self._file_name + file_name_extension + '.mat')

            self._save_pkl(save_dict, pkl_file_name)
            if self._save_to_mat_and_csv:
                self._save_mat(save_dict, mat_file_name)
                self._save_csv(save_dict, csv_file_name)
        return save_dict

    def _save_pkl(self, save_dict, pkl_file_name):
        pkl_dict = copy.deepcopy(save_dict)
        pkl_dict['track'] = self._track

        with open(pkl_file_name, 'wb') as f:
            pickle.dump(pkl_dict, f)

    def _save_mat(self, save_dict, mat_file_name):
        mat_dict = self._convert_dict_to_mat_savable_dict(save_dict)
        scipy.io.savemat(mat_file_name, mat_dict, long_field_names=True)

    def _save_csv(self, save_dict, csv_file_name):
        csv_dict = self._convert_dict_to_csv_savable_dict(save_dict)

        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_dict.keys())

            length = max([len(v) for v in csv_dict.values() if isinstance(v, list)])
            for index in range(length):
                row = []
                for value in csv_dict.values():
                    if isinstance(value, list):
                        try:
                            row.append(value[index])
                        except IndexError:
                            row.append('')
                    elif index == 0:
                        row.append(value)
                    else:
                        row.append('')
                writer.writerow(row)

    def _convert_dict_to_mat_savable_dict(self, d):
        new_dict = {}
        for key, value in d.items():

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    new_keys, new_values = self._get_mat_dict_values(key + '_' + str(sub_key), sub_value)
                    for new_key, new_value in zip(new_keys, new_values):
                        new_dict[new_key] = new_value
            else:
                new_keys, new_values = self._get_mat_dict_values(key, value)
                for new_key, new_value in zip(new_keys, new_values):
                    new_dict[new_key] = new_value

        return new_dict

    @staticmethod
    def _get_mat_dict_values(old_key, old_value):
        keys = []
        values = []

        if isinstance(old_value, list):
            values += [np.array(old_value)]
            keys += [old_key]
        elif isinstance(old_value, SimulationConstants):
            for sim_constants_key, sim_constants_value in old_value.__dict__.items():
                keys += [old_key + '_' + sim_constants_key]
                values += [sim_constants_value]
        elif type(old_value) not in [int, float, str, bool]:
            values += [str(old_value)]
            keys += [old_key]
        else:
            values += [old_value]
            keys += [old_key]
        return keys, values

    def _convert_dict_to_csv_savable_dict(self, d):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, list):
                new_dict[key] = self._convert_list_to_csv_savable_list(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        new_dict[key + '.' + str(sub_key)] = self._convert_list_to_csv_savable_list(sub_value)
                    else:
                        new_dict[key + '.' + str(sub_key)] = sub_value
            elif isinstance(value, SimulationConstants):
                for sim_constants_key, sim_constants_value in value.__dict__.items():
                    new_dict[key + '.' + sim_constants_key] = sim_constants_value
            else:
                new_dict[key] = value

        return new_dict

    def _convert_list_to_csv_savable_list(self, l):
        for index in range(len(l)):
            item = l[index]
            if isinstance(item, list):
                l[index] = self._convert_list_to_csv_savable_list(item)
            elif isinstance(item, np.ndarray):
                l[index] = item.tolist()
        return l
