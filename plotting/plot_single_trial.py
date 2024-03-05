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

import matplotlib.pyplot as plt
import numpy as np


def plot_trial(file_name):
    with open(os.path.join('..', 'data', file_name), 'rb') as f:
        playback_data = pickle.load(f)

    simulation_constants = playback_data['simulation_constants']

    dt = simulation_constants.dt  # ms
    track = playback_data['track']

    fig = plt.figure(figsize=(10, 6), layout="tight")
    spec = fig.add_gridspec(5, 2, width_ratios=[1, 3])

    overview_ax = fig.add_subplot(spec[:, 0], aspect='equal')
    overview_ax.set_xlim((-simulation_constants.sidewalk_width / 2, simulation_constants.sidewalk_width / 2))
    overview_ax.set_ylim((0., simulation_constants.sidewalk_length))

    velocity_ax = fig.add_subplot(spec[0, 1])
    velocity_ax.set_ylabel('velocity [m/s]')

    heading_ax = fig.add_subplot(spec[1, 1], sharex=velocity_ax)
    heading_ax.set_ylabel('heading [radians/pi]')

    risk_ax = fig.add_subplot(spec[2, 1], sharex=velocity_ax)
    risk_ax.set_ylim((0., 1.))
    risk_ax.set_ylabel('risk')

    acceleration_ax = fig.add_subplot(spec[3, 1], sharex=velocity_ax)
    acceleration_ax.set_ylim((-1., 1.))
    acceleration_ax.set_ylabel('acceleration input')

    steering_ax = fig.add_subplot(spec[4, 1], sharex=velocity_ax)
    steering_ax.set_ylim((-1., 1.))
    steering_ax.set_ylabel('steer input')
    steering_ax.set_xlabel('time [s]')

    time = np.array([dt / 1000. * t for t in range(len(playback_data['velocities'][0]))])
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:brown', 'tab:grey']

    for c, key in enumerate(playback_data['positions'].keys()):
        positions = np.array(playback_data['positions'][key])
        overview_ax.plot(positions[:, 0], positions[:, 1], c=colors[c])

        velocity = np.array(playback_data['velocities'][key])

        velocity_ax.plot(time, velocity, c=colors[c])
        heading_ax.plot(time, np.array(playback_data['headings'][key]) / np.pi, c=colors[c])
        risk_ax.hlines(playback_data['risk_bounds'][key], time[0], time[-1], colors=colors[c], linestyles='dashed')
        risk_ax.plot(time, playback_data['perceived_risks'][key], c=colors[c])
        raw_inputs = np.array(playback_data['raw_input'][key])

        acceleration_ax.plot(time, raw_inputs[:, 0])
        steering_ax.plot(time, raw_inputs[:, 1], c=colors[c])

        upper_replans = np.array(playback_data['is_replanning'][key]) == 1
        lower_replans = np.array(playback_data['is_replanning'][key]) == -1
        failed_replans = np.array(playback_data['is_replanning'][key]) == 3

        velocity_ax.scatter(time[upper_replans], velocity[upper_replans], marker='^', c=colors[c])
        velocity_ax.scatter(time[lower_replans], velocity[lower_replans], marker='v', c=colors[c])
        velocity_ax.scatter(time[failed_replans], velocity[failed_replans], marker='o', c=colors[c])

    plt.show()


if __name__ == '__main__':
    condition = 'different_sides'
    iteration = '22'

    file = os.path.join('simulations', condition, iteration + '.pkl')
    plot_trial(file)
