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

from plotting.load_data import load_traces_data, load_metrics_data, load_data_with_multi_processing
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    condition = 'symmetric'
    traces, metrics = load_data_with_multi_processing(os.path.join('..', 'data', 'simulations', condition), workers=20)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    sns.lineplot(traces, x='y', y='x', hue='iteration', units='id', estimator=None, ax=ax, sort=False)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    sns.lineplot(traces, x='y', y='x', c='lightgrey', units='id', estimator=None, ax=ax, sort=False, linewidth=0.3)
    sns.lineplot(traces.loc[~traces['passed_each_other']], x='y', y='x', hue='iteration', units='id', estimator=None, ax=ax, sort=False, linewidth=1.)
    plt.xlim((4, 11))

    fig, ax = plt.subplots(1, 1)
    sns.histplot(metrics, x='strategy_switches_str', ax=ax)
    plt.xlim((0, 25))

    fig, ax = plt.subplots(1, 1)
    sns.histplot(metrics.loc[metrics['pedestrian_tag'] == 0], x='end_state', ax=ax, binwidth=1)

    plt.show()
