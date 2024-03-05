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

    simulation_indices = [0, 70, 96, 101]
    times = [0.0, 3.5, 4.85, 5.05]
    strategy_switches = [0, 1, 1, 2]

    plt.rcParams["font.family"] = "Century Gothic"

    for axes_index, simulation_index in enumerate(simulation_indices):
        plan = np.array(loaded_data['position_plans'][0][simulation_index])
        x_position = loaded_data['positions'][0][simulation_index][0]
        bounds = [x_position - 0.2, x_position + 0.2]
        mean = plan[:, 0].mean()

        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

        ax.spines[:].set_visible(False)

        sns.histplot(plan[:, 0], binwidth=0.1, binrange=(-1.25, 1.25), color='#ff7f0e', linewidth=0.5, ax=ax)
        ax.vlines(bounds, -25, 140, colors='k', linewidths=1., label='Dead band')
        ax.scatter(x_position, -14, marker='^', c='#ff7f0e', s=200, edgecolors='k', label='Lateral pedestrian position')
        ax.scatter(mean, -13, marker='o', c='#ff7f0e', s=80, edgecolors='k', label='Mean lateral plan position')

        ax.set_ylabel('')
        ax.set_xlabel('Lateral position [m]')
        ax.set_yticks([], [])
        ax.set_xticks([-1.25, 0.0, 1.25], [-1.25, 0.0, 1.25])
        ax.set_xlim((-1.25, 1.25))
        ax.set_ylim((-25, 140))

        ax.set_title(r'$\bf{%d}$' % (axes_index + 1) + '\nt = %.2f\nstrategy switches = %d' % (times[axes_index], strategy_switches[axes_index]))

        fig.tight_layout()

    plt.legend(loc='upper left')
    plt.show()
