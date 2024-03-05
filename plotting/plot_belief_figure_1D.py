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
import scipy.stats as stats

import numpy as np

if __name__ == '__main__':
    condition = 'symmetric'
    iteration = '0'

    file = os.path.join('simulations', condition, iteration + '.pkl')

    with open(os.path.join('..', 'data', file), 'rb') as f:
        playback_data = pickle.load(f)

    x = np.linspace(-1.25, 1.25, 10000)

    belief_point = playback_data['beliefs'][0][0][15]

    current_heading = stats.norm.pdf(x, 0., belief_point.current_heading_sigma) * belief_point.current_heading_weight
    pass_left = stats.norm.pdf(x, belief_point.passing_left_mu, belief_point.passing_left_sigma) * belief_point.passing_left_weight
    pass_right = stats.norm.pdf(x, belief_point.passing_right_mu, belief_point.passing_right_sigma) * belief_point.passing_right_weight

    plt.rcParams["font.family"] = "Century Gothic"
    plt.rcParams.update({
        "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

    fig, ax = plt.subplots(1, 1, figsize=(13, 3))
    plt.plot(x, current_heading, label='Continuing current heading', c='tab:purple', linewidth=2.)
    plt.plot(x, pass_left, label='Passing on my right', c='tab:red', linewidth=2.)
    plt.plot(x, pass_right, label='Passing on my left', c='tab:green', linewidth=2.)
    plt.plot(x, pass_right + current_heading + pass_left, label='Total belief', c='k', linewidth=4.)

    plt.vlines([-1.25, 1.25], -0.5, 1.5, colors='grey', linewidth=1.)

    plt.ylim((-0.25, 1.25))
    plt.ylabel('Believed probability density')
    plt.xlabel('x position [m]')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend()

    plt.tight_layout()
    plt.show()
