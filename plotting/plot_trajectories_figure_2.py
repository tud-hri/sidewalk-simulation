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
import seaborn as sns

import numpy as np

if __name__ == '__main__':
    file = os.path.join('..', 'data', 'simulations', 'symmetric', '65.pkl')

    with open(file, 'rb') as f:
        loaded_data = pickle.load(f)

    simulation_indices = [0, 70, 96, 101, 103, 104, 108]
    times = [0.0, 3.5, 4.85, 5.05, 5.15, 5.20, 5.40]
    strategy_switches = [0, 1, 1, 2]

    plt.rcParams["font.family"] = "Century Gothic"

    for axes_index, simulation_index in enumerate(simulation_indices):

        position_trace_0 = np.array(loaded_data['positions'][0])[0:simulation_index + 1]
        position_trace_1 = np.array(loaded_data['positions'][1])[0:simulation_index + 1]

        fig, ax = plt.subplots(1, 1, figsize=(4, 8))
        ax.set_aspect('equal')
        # ax.spines[:].set_visible(False)

        ax.plot(position_trace_0[:,0], position_trace_0[:,1], color='#ff7f0e')
        ax.plot(position_trace_1[:,0], position_trace_1[:,1], color='#1f77b4')

        ax.set_ylabel('')
        ax.set_xlabel('Lateral position [m]')
        ax.set_yticks([], [])
        ax.set_xticks([], [])
        ax.set_xlim((-1.25, 1.25))
        # ax.set_ylim((0, 15))

        fig.tight_layout()

    plt.show()
