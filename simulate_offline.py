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

from agents import PedestrianCEIAgentPedestrianDynamics
from controllableobjects import PedestrianObject
from simulation.offlinesimmaster import OfflineSimMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import SideWalk

import numpy as np


def simulate(track, simulation_constants, iteration, cultural_biases, risk_thresholds, initial_positions, folder):
    file_name = os.path.join(folder, '%d' % iteration)

    sim_master = OfflineSimMaster(track, simulation_constants, file_name, save_to_mat_and_csv=True, verbose=False)

    time_horizon = 7.
    belief_frequency = 4
    preferred_velocity = 1.3

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
                                                         comfortable_range=.3,
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
                                                         comfortable_range=.3,
                                                         opponent_id=0,
                                                         cultural_bias=cultural_biases[1]
                                                         )

    sim_master.add_agent(0, first_pedestrian_object, cei_agent_one)
    sim_master.add_agent(1, second_pedestrian_object, cei_agent_two)

    sim_master.start()

    print('')
    print('simulation ended with exit status: ' + sim_master.end_state)


if __name__ == '__main__':
    simulation_constants = SimulationConstants(dt=50,
                                               sidewalk_width=2.5,
                                               sidewalk_length=15.,
                                               collision_tolerance=0.25,
                                               max_time=30e3)

    track = SideWalk(simulation_constants)
    iterations = 100

    conditions = {
        'symmetric': {'cultural_biases': [1., 1.],
                      'risk_thresholds': [0.65, 0.65],
                      'initial_positions': [track.get_start_position(0), track.get_start_position(1)]},
        'different_belief_bias': {'cultural_biases': [0.7, 1.3],
                                  'risk_thresholds': [0.65, 0.65],
                                  'initial_positions': [track.get_start_position(0), track.get_start_position(1)]},
        'same_belief_bias': {'cultural_biases': [0.7, 0.7],
                             'risk_thresholds': [0.65, 0.65],
                             'initial_positions': [track.get_start_position(0), track.get_start_position(1)]},
        'different_sides': {'cultural_biases': [1., 1.],
                            'risk_thresholds': [0.65, 0.65],
                            'initial_positions': [track.get_start_position(0) + np.array([0.1, 0.0]),
                                                  track.get_start_position(1) + np.array([-0.1, 0.0])]},
        'different_risk_thresholds': {'cultural_biases': [1., 1.],
                                            'risk_thresholds': [0.6, 0.7],
                                            'initial_positions': [track.get_start_position(0), track.get_start_position(1)]}
    }

    for key, value_dict in conditions.items():
        cultural_biases = value_dict['cultural_biases']
        risk_thresholds = value_dict['risk_thresholds']
        initial_positions = value_dict['initial_positions']

        subfolder = key
        folder = os.path.join('simulations', subfolder)
        os.makedirs(os.path.join('data', folder), exist_ok=True)

        n = range(iterations)
        args = zip([track] * iterations,
                   [simulation_constants] * iterations,
                   n,
                   [cultural_biases] * iterations,
                   [risk_thresholds] * iterations,
                   [initial_positions] * iterations,
                   [folder] * iterations
                   )

        with mp.Pool(16) as p:
            p.starmap(simulate, args)

    # for i in range(3):
    #     winsound.Beep(frequency=2000, duration=1000)
    #     time.sleep(0.5)
