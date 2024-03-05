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
import os.path
import multiprocessing as mp

from agents import PedestrianCEIAgent
from controllableobjects import BicycleModelObject
from simulation.offlinesimmaster import OfflineSimMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import SideWalk

import numpy as np


def _simulate_trial(track, simulation_constants, iteration, cultural_biases, risk_thresholds, initial_positions,
                    subfolder=''):
    if subfolder:
        folder = os.path.join('simulations', subfolder)
    else:
        folder = 'simulations'
    file_name = os.path.join(folder, '%d' % iteration)

    sim_master = OfflineSimMaster(track, simulation_constants, file_name, save_to_mat_and_csv=True, verbose=False)

    time_horizon = 7.
    belief_frequency = 4
    preferred_velocity = 1.3

    first_bicycle_object = BicycleModelObject(initial_position=initial_positions[0],
                                              initial_velocity=preferred_velocity,
                                              initial_heading=np.pi / 2.0,
                                              wheelbase=0.1,
                                              resistance_coefficient=0.,
                                              constant_resistance=0.)

    second_bicycle_object = BicycleModelObject(initial_position=initial_positions[1],
                                               initial_velocity=preferred_velocity,
                                               initial_heading=-np.pi / 2.0,
                                               wheelbase=0.1,
                                               resistance_coefficient=0.,
                                               constant_resistance=0.)

    cei_agent_one = PedestrianCEIAgent(controllable_object=first_bicycle_object,
                                       dt=simulation_constants.dt,
                                       sim_master=sim_master,
                                       track=track,
                                       risk_threshold=risk_thresholds[0],
                                       time_horizon=time_horizon,
                                       planning_frequency=belief_frequency,
                                       preferred_velocity=preferred_velocity,
                                       preferred_heading=np.pi / 2,
                                       comfortable_range=.3,
                                       opponent_id=1,
                                       cultural_bias=cultural_biases[0])
    cei_agent_two = PedestrianCEIAgent(controllable_object=second_bicycle_object,
                                       dt=simulation_constants.dt,
                                       sim_master=sim_master,
                                       track=track,
                                       risk_threshold=risk_thresholds[1],
                                       time_horizon=time_horizon,
                                       planning_frequency=belief_frequency,
                                       preferred_velocity=preferred_velocity,
                                       preferred_heading=-np.pi / 2,
                                       comfortable_range=.3,
                                       opponent_id=0,
                                       cultural_bias=cultural_biases[1]
                                       )

    sim_master.add_agent(0, first_bicycle_object, cei_agent_one)
    sim_master.add_agent(1, second_bicycle_object, cei_agent_two)

    sim_master.start()

    print('')
    print('simulation ended with exit status: ' + sim_master.end_state)


def _simulate_scenario_with_multiprocessing(simulation_constants, track, iterations, cultural_biases, risk_thresholds,
                                            initial_positions, condition_name):
    os.makedirs(os.path.join('data', 'simulations', condition_name), exist_ok=True)

    n = range(iterations)
    args = zip([track] * iterations,
               [simulation_constants] * iterations,
               n,
               [cultural_biases] * iterations,
               [risk_thresholds] * iterations,
               [initial_positions] * iterations,
               [condition_name] * iterations
               )

    with mp.Pool(8) as p:
        p.starmap(_simulate_trial, args)


def simulate_symmetrical_scenario(simulation_constants, track, iterations=100):
    cultural_biases = [1., 1.]
    risk_thresholds = [0.6, 0.6]
    initial_positions = [track.get_start_position(0),
                         track.get_start_position(1)]
    condition_name = 'symmetric'
    _simulate_scenario_with_multiprocessing(simulation_constants, track, iterations, cultural_biases, risk_thresholds,
                                            initial_positions, condition_name)


def simulate_different_thresholds_scenario(simulation_constants, track, iterations=100):
    cultural_biases = [1., 1.]
    risk_thresholds = [0.5, 0.7]
    initial_positions = [track.get_start_position(0),
                         track.get_start_position(1)]
    condition_name = 'different_risk_thresholds'
    _simulate_scenario_with_multiprocessing(simulation_constants, track, iterations, cultural_biases, risk_thresholds,
                                            initial_positions, condition_name)


def simulate_different_sides_scenario(simulation_constants, track, iterations=100):
    cultural_biases = [1., 1.]
    risk_thresholds = [0.6, 0.6]
    initial_positions = [track.get_start_position(0) + np.array([0.2, 0.0]),
                         track.get_start_position(1) + np.array([-0.2, 0.0])]
    condition_name = 'different_sides'
    _simulate_scenario_with_multiprocessing(simulation_constants, track, iterations, cultural_biases, risk_thresholds,
                                            initial_positions, condition_name)


def simulate_same_belief_bias_scenario(simulation_constants, track, iterations=100):
    cultural_biases = [1.2, 1.2]
    risk_thresholds = [0.6, 0.6]
    initial_positions = [track.get_start_position(0),
                         track.get_start_position(1)]
    condition_name = 'same_belief_bias'
    _simulate_scenario_with_multiprocessing(simulation_constants, track, iterations, cultural_biases, risk_thresholds,
                                            initial_positions, condition_name)


def simulate_different_belief_bias_scenario(simulation_constants, track, iterations=100):
    cultural_biases = [0.8, 1.2]
    risk_thresholds = [0.6, 0.6]
    initial_positions = [track.get_start_position(0),
                         track.get_start_position(1)]
    condition_name = 'different_belief_bias'
    _simulate_scenario_with_multiprocessing(simulation_constants, track, iterations, cultural_biases, risk_thresholds,
                                            initial_positions, condition_name)


if __name__ == '__main__':
    constants = SimulationConstants(dt=50,  # [ms]
                                    sidewalk_width=2.5,  # [m]
                                    sidewalk_length=15.,  # [m]
                                    collision_tolerance=0.25,  # [m]
                                    max_time=30e3)  # [ms]

    sidewalk_track = SideWalk(constants)

    print("Simulating symmetrical scenario")
    simulate_symmetrical_scenario(constants, sidewalk_track)
    print("Symmetrical scenario done")

    print("Simulating different risk thresholds scenario")
    simulate_different_thresholds_scenario(constants, sidewalk_track)
    print("Different risk thresholds scenario done")

    print("Simulating different sides scenario")
    simulate_different_sides_scenario(constants, sidewalk_track)
    print("Different sides scenario done")

    print("Simulating same belief bias scenario")
    simulate_same_belief_bias_scenario(constants, sidewalk_track)
    print("Same belief bias scenario done")

    print("Simulating different belief bias scenario")
    simulate_different_belief_bias_scenario(constants, sidewalk_track)
    print("Different belief bias scenario done")
