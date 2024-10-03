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

from plotting.load_data import load_traces_data, load_metrics_data, load_all_data, load_data_with_multi_processing
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    simulations_folder = os.path.join('..', 'data', 'simulations')
    traces, metrics = load_data_with_multi_processing(simulations_folder, workers=20)
    plt.rcParams["font.family"] = "Century Gothic"
    end_states = metrics.loc[:, ['end_state', 'formatted_condition']].groupby(['formatted_condition', 'end_state'])
    end_states = end_states.size().unstack(fill_value=0)
    end_states = (end_states / 2).astype(int)
    print()
    print(end_states)
    print()

    conflicts_per_condition = {}
    conflict_ids = {}
    end_states_in_non_conflict = []

    for condition in metrics['condition'].unique():
        number_of_conflicts = 0
        conflict_iterations = []
        for iteration in range(100):
            sim_metrics = metrics.loc[(metrics['condition'] == condition) & (metrics['iteration'] == iteration)]
            if (sim_metrics['strategy_switches'].to_numpy() >= 2).all():
                number_of_conflicts += 1
                conflict_iterations.append(iteration)
            else:
                end_states_in_non_conflict.append(sim_metrics['end_state'].iat[0])

        conflicts_per_condition[condition] = number_of_conflicts
        conflict_ids[condition] = conflict_iterations

    print(conflicts_per_condition)
    print()
    print(conflict_ids)
    print()

    traces['-x'] = -traces['x']

    fig, ax = plt.subplots(1, 1)
    sns.histplot(metrics, x='strategy_switches', ax=ax, binwidth=1, hue='condition', kde=True, multiple="dodge")
    plt.xlim((0, 10))
    plt.xticks([i + 0.5 for i in range(11)], [str(i) for i in range(11)])
    plt.vlines([i for i in range(11)], 0, 200, colors='lightgray')

    number_of_conditions = len(metrics['condition'].unique())
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    fig = plt.figure(layout="constrained", figsize=(12, 5))

    gridspec = plt.GridSpec(number_of_conditions, 2, figure=fig)
    ax = fig.add_subplot(gridspec[0, 0])

    conditions_order = ['symmetric', 'different_sides', 'different_risk_thresholds', 'same_belief_bias', 'different_belief_bias']

    for condition_index, condition in enumerate(conditions_order):
        if condition_index:
            ax = fig.add_subplot(gridspec[condition_index, 0], sharex=ax, sharey=ax)
        sns.histplot(metrics.loc[metrics['condition'] == condition], x='strategy_switches_str', ax=ax,
                     color=colors[condition_index])
        condition_name = metrics.loc[metrics['condition'] == condition]['formatted_condition'].iat[0]
        ax.set_title(condition_name)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if condition_index == 2:
            ax.set_ylabel('Count (n=200)')
        if condition_index < 4:
            ax.tick_params(labelbottom=False,
                           bottom=False)
    plt.xlim((0, 10))
    ax.set_xlabel('Number of strategy switches')

    ax = fig.add_subplot(gridspec[:, 1])
    sns.kdeplot(metrics, x='strategy_switches', ax=ax, hue='condition', hue_order=conditions_order, bw_method=0.5)
    ax.set_xlabel('Number of strategy switches')
    ax.set_ylabel('Estimated Density')
    plt.xlim((0, 10))
    plt.tight_layout()

    fig = plt.figure(figsize=(12,8))
    gridspec = plt.GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gridspec[0, :])]
    axes[0].set_xlim((0, 15))
    for i in range(2):
        for j in range(2):
            if i + j == 0:
                axes.append(fig.add_subplot(gridspec[i + 1, j]))
            else:
                axes.append(fig.add_subplot(gridspec[i + 1, j], sharex=axes[1], sharey=axes[1]))

    traces['-x'] = -traces['x']
    for condition_index, condition in enumerate(conditions_order):
        ax = axes[condition_index]
        ax.set_aspect('equal')
        condition_data = traces.loc[(traces['condition'] == condition) & (traces['end_state'] == 'Finished')]
        sns.lineplot(condition_data, x='y', y='-x', c='lightgrey', units='id', estimator=None, ax=ax, sort=False,
                     linewidth=0.3)
        sns.lineplot(condition_data.loc[~condition_data['passed_each_other']], x='y', y='-x', hue='iteration',
                     units='id', estimator=None, ax=ax, sort=False, linewidth=1., legend=False, palette="crest_r")
        condition_name = metrics.loc[metrics['condition'] == condition, 'formatted_condition'].iat[0]
        ax.set_title(condition_name)
        ax.set_xlabel('y [m]')
        ax.set_ylabel('x [m]')

        ax.set_yticks([-1, -0.5, 0., 0.5, 1.0], ['1.0', '0.5', '0.0', '-0.5', '-1.0'])
        ax.set_xlabel('y [m]')
        ax.set_ylabel('x [m]')

    plt.xlim((4, 11))
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    count_data = metrics.loc[:, ['strategy_switches', 'formatted_condition']].groupby(['strategy_switches', 'formatted_condition']).size()
    count_data = count_data.unstack(fill_value=0)
    count_data = count_data.melt(value_name='count', ignore_index=False).reset_index()
    sns.scatterplot(count_data, x='strategy_switches', y='count', ax=ax, hue='formatted_condition')
    sns.lineplot(count_data, x='strategy_switches', y='count', ax=ax, hue='formatted_condition', legend=False)
    ax.set_xlabel('Number of strategy switches')
    ax.set_ylabel('Count (n=200)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend().set_title('Condition')
    plt.xlim((0, 10))
    plt.ylim((0, 200))
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1)
    sns.histplot(metrics, x='steered_back', ax=ax, binwidth=1, hue='condition', multiple="dodge")

    fig, ax = plt.subplots(1, 1)
    sns.histplot(metrics, x='end_state', ax=ax, binwidth=1, hue='condition', multiple="dodge")

    fig, ax = plt.subplots(1, 1)
    sns.boxplot(metrics.loc[:, ['x_distance_at_pass', 'y_distance_at_first_replan']])

    plt.show()
