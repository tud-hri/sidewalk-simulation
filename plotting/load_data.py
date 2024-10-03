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
import glob
import pickle
import tqdm

import pandas as pd
import numpy as np
import multiprocessing as mp

from tools import ProgressProcess

def load_all_data(simulation_data_folder):
    try:
        with open(os.path.join(simulation_data_folder, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
        with open(os.path.join(simulation_data_folder, 'traces.pkl'), 'rb') as f:
            traces = pickle.load(f)
    except FileNotFoundError:
        all_files = glob.glob(os.path.join(simulation_data_folder, '**', '*.pkl'), recursive=True)
        all_traces_dfs = []
        all_metrics_dfs = []

        for file in tqdm.tqdm(all_files):
            with open(file, 'rb') as f:
                traces_df, metrics_df = _load_traces_and_metrics_from_file(file)

            all_traces_dfs.append(traces_df)
            all_metrics_dfs.append(metrics_df)

        traces = pd.concat(all_traces_dfs, ignore_index=True)
        metrics = pd.concat(all_metrics_dfs, ignore_index=True)

        with open(os.path.join(simulation_data_folder, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        with open(os.path.join(simulation_data_folder, 'traces.pkl'), 'wb') as f:
            pickle.dump(traces, f)

    return traces, metrics


def _load_traces_and_metrics_from_file(file_name, progress_queue=None):
    with open(file_name, 'rb') as f:
        loaded_data = pickle.load(f)
    traces = _load_traces_data(loaded_data, file_name)
    metrics = _load_metrics_data(loaded_data, file_name)

    if progress_queue is not None:
        progress_queue.put(1)

    return traces, metrics


def load_traces_data(folder):
    all_files = glob.glob(os.path.join(folder, '*.pkl'))
    all_dfs = []

    for file in tqdm.tqdm(all_files):
        with open(file, 'rb') as f:
            loaded_data = pickle.load(f)
        df = _load_traces_data(loaded_data, file)
        all_dfs.append(df)
    traces = pd.concat(all_dfs, ignore_index=True)
    return traces


def _load_traces_data(loaded_data, file_name):
    data = {'iteration': [],
            'condition': [],
            'pedestrian_tag': [],
            'end_state': [],
            'id': [],
            'x': [],
            'y': [],
            'v': [],
            'heading': [],
            'a': [],
            'steering': [],
            'risk': [],
            'time': [],
            'passed_each_other': [],
            }

    condition = file_name.split(os.sep)[-2]

    iteration = int(os.path.split(file_name)[-1].replace('.pkl', ''))
    data_length = len(loaded_data['velocities'][0])
    dt = loaded_data['simulation_constants'].dt / 1000
    time = [t * dt for t in range(data_length)]

    passed_trace = (np.array(loaded_data['positions'][0])[:, 1] - np.array(loaded_data['positions'][1])[:, 1]) > 0

    for pedestrian_tag in loaded_data['velocities'].keys():
        data['iteration'] += [iteration] * data_length
        data['condition'] += [condition] * data_length
        data['pedestrian_tag'] += [pedestrian_tag] * data_length
        data['end_state'] += [loaded_data['end_state']] * data_length
        data['id'] += [str(iteration) + '-' + str(pedestrian_tag) + '-' + condition] * data_length
        positions = np.array(loaded_data['positions'][pedestrian_tag])
        data['x'] += positions[:, 0].tolist()
        data['y'] += positions[:, 1].tolist()
        data['v'] += loaded_data['velocities'][pedestrian_tag]
        data['heading'] += loaded_data['headings'][pedestrian_tag]
        data['a'] += loaded_data['accelerations'][pedestrian_tag]
        raw_inputs = np.array(loaded_data['raw_input'][pedestrian_tag])
        if loaded_data['steering_angles'][pedestrian_tag]:
            data['steering'] += np.array(loaded_data['steering_angles'][pedestrian_tag])
        else:
            data['steering'] += [0.] * len(loaded_data['velocities'][pedestrian_tag])
        data['risk'] += loaded_data['perceived_risks'][pedestrian_tag]
        data['time'] += time
        data['passed_each_other'] += passed_trace.tolist()

    df = pd.DataFrame(data)
    return df


def load_metrics_data(folder):
    all_files = glob.glob(os.path.join(folder, '*.pkl'))
    all_dfs = []

    for file in tqdm.tqdm(all_files):
        with open(file, 'rb') as f:
            loaded_data = pickle.load(f)
        df = _load_metrics_data(loaded_data, file)
        all_dfs.append(df)

    metrics = pd.concat(all_dfs, ignore_index=True)
    return metrics


def _load_metrics_data(loaded_data, file_name):
    data = {'iteration': [],
            'condition': [],
            'pedestrian_tag': [],
            'id': [],
            'strategy_switches': [],
            'x_variability': [],
            'x_distance_at_pass': [],
            'x_difference_at_pass': [],
            'y_distance_at_first_replan': [],
            'steered_back': [],
            'end_state': [],
            'number_of_solver_fails': []
            }

    condition = file_name.split(os.sep)[-2]
    iteration = int(os.path.split(file_name)[-1].replace('.pkl', ''))
    try:
        final_index = np.where(np.array(loaded_data['positions'][0])[:, 1] > np.array(loaded_data['positions'][1])[:, 1])[0][0]
    except IndexError:
        final_index = len(loaded_data['positions'][0]) - 1

    for pedestrian_tag in loaded_data['velocities'].keys():
        previous_high_level_plan = 'straight'
        number_of_switches = 0
        for index in range(final_index):
            plan = np.array(loaded_data['position_plans'][pedestrian_tag][index])

            mean_plan_x = plan[:, 0].mean()
            current_x = loaded_data['positions'][pedestrian_tag][index][0]

            if abs(mean_plan_x - current_x) <= 0.2:
                high_level_plan = previous_high_level_plan
            elif mean_plan_x - current_x < -0.2:
                high_level_plan = 'left'
            elif mean_plan_x - current_x > 0.2:
                high_level_plan = 'right'

            if previous_high_level_plan != high_level_plan:
                number_of_switches += 1

            previous_high_level_plan = high_level_plan

        positions = np.array(loaded_data['positions'][pedestrian_tag])
        x_variability = np.var(positions[:, 0])

        headings = np.array(loaded_data['headings'][pedestrian_tag])
        initial_heading = headings[0]
        steered_back = ((initial_heading - np.pi / 2 > headings) | (headings > initial_heading + np.pi / 2)).any()

        solver_fails = sum(np.array(loaded_data['is_replanning'][pedestrian_tag]) == 3)

        x_distance_at_pass = None
        x_difference_at_pass = None
        if loaded_data['end_state'] == 'Finished':
            x_distance_at_pass = abs(loaded_data['positions'][0][final_index][0] - loaded_data['positions'][1][final_index][0])
            x_difference_at_pass = loaded_data['positions'][0][final_index][0] - loaded_data['positions'][1][final_index][0]

        y_distance_at_first_replan = None
        if loaded_data['end_state'] == 'Finished':
            other_id = abs(pedestrian_tag - 1)
            try:
                first_replan_index = np.where(np.array(loaded_data['is_replanning'][pedestrian_tag]))[0][0]
                y_distance_at_first_replan = abs(loaded_data['positions'][pedestrian_tag][first_replan_index][1] -
                                             loaded_data['positions'][other_id][first_replan_index][1])
            except IndexError:
                pass

        data['iteration'] += [iteration]
        data['condition'] += [condition]
        data['pedestrian_tag'] += [pedestrian_tag]
        data['id'] += [str(iteration) + '-' + str(pedestrian_tag) + '-' + condition]
        data['strategy_switches'] += [number_of_switches]
        data['x_distance_at_pass'] += [x_distance_at_pass]
        data['x_difference_at_pass'] += [x_difference_at_pass]
        data['y_distance_at_first_replan'] += [y_distance_at_first_replan]
        data['x_variability'] += [x_variability]
        data['end_state'] += [loaded_data['end_state']]
        data['steered_back'] += [str(steered_back)]
        data['number_of_solver_fails'] += [solver_fails]

    df = pd.DataFrame(data)
    df['strategy_switches_str'] = pd.Categorical(df['strategy_switches'].astype(str), [str(i) for i in range(100)])

    def format_condition_name(name):
        formatted_name = name.replace('_', ' ').capitalize()
        return formatted_name

    df['formatted_condition'] = df['condition'].apply(format_condition_name)

    return df


def load_data_with_multi_processing(simulation_data_folder, workers=8):
    all_files = glob.glob(os.path.join(simulation_data_folder, '**', '*.pkl'), recursive=True)

    manager = mp.Manager()
    progress_process = ProgressProcess(len(all_files), manager)
    progress_process.start()

    arguments = zip(all_files, [progress_process.queue] * len(all_files))

    with mp.Pool(workers) as p:
        results = p.starmap(_load_traces_and_metrics_from_file, arguments)
    traces_list, metrics_list = list(zip(*results))

    traces = pd.concat(traces_list, ignore_index=True)
    metrics = pd.concat(metrics_list, ignore_index=True)

    return traces, metrics
