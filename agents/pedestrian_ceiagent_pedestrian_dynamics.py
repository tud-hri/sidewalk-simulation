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
import copy

import numpy as np
import casadi

from controllableobjects import PedestrianObject
from .agent import Agent
from colorama import Fore, Style


class BeliefPoint:
    def __init__(self):
        self.passing_left_mu = 0.
        self.passing_left_sigma = 0.
        self.passing_left_weight = 0.

        self.passing_right_mu = 0.
        self.passing_right_sigma = 0.
        self.passing_right_weight = 0.

        self.current_heading_mu = 0.
        self.current_heading_sigma = 0.
        self.current_heading_weight = 0.

        self.y = 0.

    def as_list(self):
        return [self.passing_left_mu, self.passing_left_sigma, self.passing_left_weight,
                self.passing_right_mu, self.passing_right_sigma, self.passing_right_weight,
                self.current_heading_mu, self.current_heading_sigma, self.current_heading_weight,
                self.y]

    def get_normalized_values(self, slices, track_width):
        """
        Returns an array of n 'slices' that represent the combined normalized probability density function of the
        belief points. Together with their normalized positions on the track width
        Always contains the start and end and mu values/position.
        slices should be minimally 5.
        This is used for plotting heatmaps.
        """
        if slices < 5:
            raise ValueError('slices must be at least 5')
        x = np.linspace(-track_width / 2, track_width / 2, slices - 3)[1:-1]
        x = np.append(x, [-track_width / 2,
                          self.passing_left_mu,
                          self.passing_right_mu,
                          self.current_heading_mu,
                          track_width / 2])
        x.sort()
        p = 0.

        for parameters in [(self.passing_left_mu, self.passing_left_sigma, self.passing_left_weight),
                           (self.passing_right_mu, self.passing_right_sigma, self.passing_right_weight),
                           (self.current_heading_mu, self.current_heading_sigma, self.current_heading_weight)]:
            mu, sigma, weight = parameters
            p += weight * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-.5 * ((x - mu) ** 2) / (sigma ** 2))
        p /= np.amax(p)
        x = (x + (track_width / 2)) / track_width
        return x, p


class PedestrianCEIAgentPedestrianDynamics(Agent):
    """
    An agent used in a Communication-Enabled Interaction pedestrian model.
    """

    def __init__(self, controllable_object: PedestrianObject, dt, sim_master, track, risk_threshold, time_horizon,
                 planning_frequency, preferred_velocity, preferred_heading, comfortable_range, opponent_id,
                 expected_lateral_acceleration=0.2, cultural_bias=1.):
        self.controllable_object = controllable_object
        self.dt = dt
        simulation_frequency = int(1000 / dt)
        self.sim_master = sim_master
        self.track = track
        self.risk_threshold = risk_threshold
        self.time_horizon = time_horizon
        self.planning_frequency = planning_frequency
        self.planning_simulation_ratio = int(simulation_frequency / planning_frequency)
        self.optimization_failed = False
        self.preferred_velocity = preferred_velocity
        self.preferred_heading = preferred_heading
        self.comfortable_range = comfortable_range
        self.expected_lateral_acceleration = expected_lateral_acceleration
        self.opponent_id = opponent_id
        self.perception_update_rate = 1 / (simulation_frequency * 0.5)
        self.cultural_bias = cultural_bias  # 1.0 means no bias; >1 is percentual bias to the right; <1 to the left

        # the action plan consists of the action (constant acceleration) to take at the coming time steps. The position plan is the set of positions along the
        # track where the ego vehicle will end up when taking these actions.
        self.plan_length = int((1000 / dt) * time_horizon)
        self.action_plan = np.array([[0.0, 0., 0.]] * self.plan_length)
        self.heading_plan = np.array([0.0] * self.plan_length)
        self.velocity_plan = np.array([[0.0, 0., 0.]] * self.plan_length)
        self.position_plan = np.array([[0.0, 0.]] * self.plan_length)
        self.action_bounds = [-1, 1]

        # the belief consists of sets of a mean and standard deviation for a distribution over positions at every time step.
        self.belief = []
        self.belief_time_stamps = []
        for belief_index in range(int(planning_frequency * time_horizon)):
            self.belief.append(BeliefPoint())
            self.belief_time_stamps.append((1 / self.planning_frequency) * (belief_index + 1))

        self.casadi_belief = None
        self.did_plan_update_on_last_tick = 0
        self.perceived_risk = 0.
        self.collision_risk = 0.
        self.side_risk = 0.

        # The observed communication is the current position, velocity, and acceleration of the other pedestrian
        self.observed_position = [0.0, 0.0]
        self.observed_long_velocity = self.controllable_object.initial_velocity[0]
        self.observed_lat_velocity = self.controllable_object.initial_velocity[1]
        self.observed_heading = -self.controllable_object.heading

        self._is_initialized = False
        self._initialize_optimization()

    def reset(self):
        self.action_plan = np.array([[0.0, 0., 0.]] * self.plan_length)
        self.heading_plan = np.array([0.0] * self.plan_length)
        self.velocity_plan = np.array([0.0, 0.] * self.plan_length)
        self.position_plan = np.array([[0.0, 0.]] * self.plan_length)

        self.belief = []
        self.belief_time_stamps = []
        for belief_index in range(int(self.planning_frequency * self.time_horizon)):
            self.belief.append(BeliefPoint())
            self.belief_time_stamps.append((1 / self.planning_frequency) * (belief_index + 1))

        self.casadi_belief = None
        self.did_plan_update_on_last_tick = 0
        self.perceived_risk = 0.

        self.observed_position = [0.0, 0.0]
        self.observed_long_velocity = self.controllable_object.initial_velocity[0]
        self.observed_lat_velocity = self.controllable_object.initial_velocity[1]
        self.observed_heading = -self.controllable_object.heading

        self._is_initialized = False
        self._initialize_optimization()

    def _observe_communication(self):
        other_position, other_velocity, other_heading = self.sim_master.get_current_state(self.opponent_id)
        other_long_velocity = other_velocity[0]
        other_lat_velocity = other_velocity[1]

        self.observed_position = other_position

        noise = np.random.normal(scale=np.sqrt(self.dt / 1000.)) * 0.03
        long_velocity_update = self.perception_update_rate * (other_long_velocity - self.observed_long_velocity) + noise
        self.observed_long_velocity = max(self.observed_long_velocity + long_velocity_update, 0.0)

        noise = np.random.normal(scale=np.sqrt(self.dt / 1000.)) * 0.03
        lat_velocity_update = self.perception_update_rate * (other_lat_velocity - self.observed_lat_velocity) + noise
        self.observed_lat_velocity = self.observed_lat_velocity + lat_velocity_update

        noise = np.random.normal(scale=np.sqrt(self.dt / 1000.)) * 0.03
        heading_update = self.perception_update_rate * (other_heading - self.observed_heading) + noise
        self.observed_heading = self.observed_heading + heading_update

    def _initialize_optimization(self):
        # Optimization based on Casadi (Andersson2018), documentation: https://web.casadi.org/docs/, used the opti() stack
        # Optimization steps based on time horizon of belief points
        N = int(self.time_horizon * self.planning_frequency)

        # Initialize decision variables and parameters
        self.optimizer = casadi.Opti()
        self.x = self.optimizer.variable(6, N + 1)  # The state [x, y, v_long, v_lat, v_rotation, heading] x N
        self.u = self.optimizer.variable(3, N)  # The control input [a_long, a_lat, omega_dot] (acceleration) x N
        total_cost = 0
        self.risk_bound_agent = self.optimizer.parameter()
        self.casadi_belief = []

        for i in range(N):
            # Dynamics, use as equality constraint
            x_now = self.x[:, i]
            u_now = self.u[:, i]

            x_new = self._dynamics_casadi(x_now, u_now)
            self.optimizer.subject_to(self.x[:, i + 1] == x_new)

            # Cost evaluation
            total_cost += self._cost_function_casadi(x_new, u_now, self.preferred_velocity, self.preferred_heading)

            # Inequality constraints
            belief_point = [self.optimizer.parameter(), self.optimizer.parameter(), self.optimizer.parameter(),
                            self.optimizer.parameter(), self.optimizer.parameter(), self.optimizer.parameter(),
                            self.optimizer.parameter(), self.optimizer.parameter(), self.optimizer.parameter(),
                            self.optimizer.parameter()]

            self.casadi_belief.append(belief_point)

            collision_risk = self._get_belief_probability(belief_point, x_new[0:2], self.comfortable_range)
            side_risk = self._get_side_risk_for_individual_point(x_new[0:2], self.track.track_width, (1 / self.planning_frequency) * (i + 1),
                                                                 self.time_horizon)
            total_risk = collision_risk + side_risk

            self.optimizer.subject_to(total_risk <= self.risk_bound_agent)

        # Set bounds
        self.optimizer.subject_to(self.u[:] >= self.action_bounds[0])
        self.optimizer.subject_to(self.u[:] <= self.action_bounds[1])

        # Set initial x
        self.x_initial = casadi.vertcat(self.optimizer.parameter(), self.optimizer.parameter(),
                                        self.optimizer.parameter(), self.optimizer.parameter(),
                                        self.optimizer.parameter(), self.optimizer.parameter())
        self.optimizer.subject_to(self.x[:, 0] == self.x_initial)

        # Set cost function
        self.optimizer.minimize(total_cost)

    @staticmethod
    def _get_belief_probability(belief_point, plan_point, comfortable_range):

        dy = (belief_point[9] - plan_point[1])
        dy_weight = casadi.exp(-dy ** 2 / (2 * comfortable_range) ** 2)
        bound = comfortable_range

        ub = plan_point[0] + bound
        lb = plan_point[0] - bound

        p = 0.

        for index in range(3):
            mu = belief_point[index * 3]
            sigma = belief_point[index * 3 + 1]
            weight = belief_point[index * 3 + 2]

            p += dy_weight * weight * (1 / 2) * (casadi.erf((ub - mu) / (sigma * casadi.sqrt(2))) -
                                                 casadi.erf((lb - mu) / (sigma * casadi.sqrt(2))))

        return p

    @staticmethod
    def _get_side_risk_for_individual_point(plan_point, sidewalk_width, d_t, planning_horizon):
        x = plan_point[0]

        edge = sidewalk_width / 2.
        r = 1 - casadi.tanh(10 * (x + edge - 0.15)) / 2 + casadi.tanh(10 * (x - edge + 0.15)) / 2
        r *= casadi.exp(-d_t / planning_horizon)
        return r

    def _dynamics_casadi(self, state, u, dt=None):
        if not dt:
            dt = 1 / self.planning_frequency
        # Equality constraints for the optimization
        a_long = u[0]
        a_lat = u[1]
        omega_dot = u[2]

        x = state[0]
        y = state[1]
        v_long = state[2]
        v_lat = state[3]
        v_angular = state[4]
        heading = state[5]

        a_long *= self.controllable_object.max_long_acceleration
        a_lat *= self.controllable_object.max_lat_acceleration
        omega_dot *= self.controllable_object.max_angular_acceleration

        x_dot = v_long * np.cos(heading) - v_lat * np.sin(heading)
        y_dot = v_long * np.sin(heading) + v_lat * np.cos(heading)

        x_dot_dot = a_long * np.cos(heading) - a_lat * np.sin(heading)
        y_dot_dot = a_long * np.sin(heading) + a_lat * np.cos(heading)

        new_x = x + x_dot * dt + .5 * x_dot_dot * dt ** 2
        new_y = y + y_dot * dt + .5 * y_dot_dot * dt ** 2
        new_heading = heading + v_angular * dt + 0.5 * omega_dot * dt ** 2
        new_v_long = v_long + a_long * dt
        new_v_lat = v_lat + a_lat * dt
        new_v_angular = v_angular + omega_dot * dt

        return casadi.vertcat(new_x, new_y, new_v_long, new_v_lat, new_v_angular, new_heading)

    def _cost_function_casadi(self, x, u, desired_velocity, desired_heading):
        # Cost function for the optimization
        a_long = u[0]
        a_lat = u[1]
        omega_dot = u[2]

        v_long = x[2]
        v_lat = x[3]
        heading = x[5]

        v_diff = (v_long - desired_velocity)
        sigmoid = .5 + casadi.tanh(100 * v_long) / 2.
        # v_diff_positive = sigmoid * v_diff
        v_negative = (1 - sigmoid) * v_long

        return (v_diff ** 2 +
                2. * v_lat ** 2 +
                100. * v_negative ** 2 +
                5. * (heading - desired_heading) ** 2 +
                1. * omega_dot ** 2 +
                1. * a_long ** 2 +
                1. * a_lat ** 2)

    def _update_belief(self):
        direction = np.sign(self.preferred_heading)
        track_left_x = -1 * direction * self.track.track_width / 2.

        ego_effect_weight = 0.5

        other_x = self.observed_position[0]
        other_y = self.observed_position[1]

        left_p = ((np.atan2((other_y - self.controllable_object.position[1]) / 4,
                            (other_x - self.controllable_object.position[0])) - self.preferred_heading) / np.pi) + 0.5
        left_p -= min(1.0, max(-1.0, self.observed_lat_velocity))

        left_p = min(1.0, max(0.0, left_p))
        right_p = 1. - left_p

        for belief_index, t in enumerate(self.belief_time_stamps):
            belief_point = self.belief[belief_index]
            belief_point.y = self.observed_position[1] + np.sin(self.observed_heading) * self.observed_long_velocity * t

            other_extrapolated_x = self.observed_position[0] + (
                    np.cos(self.observed_heading) * self.observed_long_velocity - np.sin(self.observed_heading) * self.observed_lat_velocity) * t

            if direction * (other_extrapolated_x - self.controllable_object.position[0]) < - self.comfortable_range:
                belief_point.passing_left_mu = other_extrapolated_x
            else:
                belief_point.passing_left_mu = self.controllable_object.position[0] - direction * self.comfortable_range

            space_on_my_left = abs(track_left_x - self.controllable_object.position[0])
            belief_point.passing_left_sigma = (space_on_my_left - self.comfortable_range) / 6.
            belief_point.passing_left_weight = (ego_effect_weight *
                                                (2. - self.cultural_bias) *
                                                left_p)

            if direction * (other_extrapolated_x - self.controllable_object.position[0]) > self.comfortable_range:
                belief_point.passing_right_mu = other_extrapolated_x
            else:
                belief_point.passing_right_mu = self.controllable_object.position[0] + direction * self.comfortable_range

            space_on_my_right = self.track.track_width - space_on_my_left
            belief_point.passing_right_sigma = (space_on_my_right - self.comfortable_range) / 6.
            belief_point.passing_right_weight = (ego_effect_weight *
                                                 self.cultural_bias *
                                                 right_p)

            belief_point.current_heading_mu = other_extrapolated_x
            belief_point.current_heading_sigma = .5 * t ** 2 * (self.expected_lateral_acceleration / 3)
            belief_point.current_heading_weight = 1 - ego_effect_weight

    def _evaluate_risk(self):
        collision_risks = self._get_collision_probabilities(self.belief, self.position_plan)
        side_risks = self._get_side_risk(self.position_plan)

        risks = np.array(collision_risks) + np.array(side_risks)[self.planning_simulation_ratio - 1::self.planning_simulation_ratio]
        max_risk = np.amax(risks)
        max_risk_index = np.argmax(risks)

        self.collision_risk = collision_risks[max_risk_index]
        self.side_risk = side_risks[self.planning_simulation_ratio - 1::self.planning_simulation_ratio][max_risk_index]

        return max_risk

    def _get_collision_probabilities(self, belief, position_plan):
        probabilities_over_plan = []

        for belief_index, belief_point in enumerate(belief):
            plan_point = position_plan[self.planning_simulation_ratio - 1::self.planning_simulation_ratio][belief_index]

            collision_probability = self._get_belief_probability(belief_point.as_list(), plan_point, self.comfortable_range)
            probabilities_over_plan += [collision_probability]

        return probabilities_over_plan

    def _get_side_risk(self, position_plan):
        side_risk = []

        for plan_index, plan_point in enumerate(position_plan):
            side_risk.append(self._get_side_risk_for_individual_point(plan_point, self.track.track_width, (self.dt / 1000) * (plan_index + 1),
                                                                      self.time_horizon))

        return side_risk

    def _update_plan(self, constraint_risk_bound):
        # Set initial values
        # The state [x, y, v_long, v_lat, v_rotation, heading] x N
        initial_state = casadi.vertcat(self.controllable_object.position[0],
                                       self.controllable_object.position[1],
                                       self.controllable_object.velocity[0],
                                       self.controllable_object.velocity[1],
                                       self.controllable_object.angular_velocity,
                                       self.controllable_object.heading)
        self.optimizer.set_value(self.x_initial, initial_state)

        # Set initial guess
        self.optimizer.set_initial(self.x[:, 0], initial_state)
        plan_indices_for_belief = (np.array(self.belief_time_stamps) / (self.dt / 1000)).astype(int) - 1
        self.optimizer.set_initial(self.x[0:2, 1:], self.position_plan[plan_indices_for_belief].T)
        self.optimizer.set_initial(self.x[2:5, 1:], self.velocity_plan[plan_indices_for_belief].T)
        self.optimizer.set_initial(self.x[5, 1:], self.heading_plan[plan_indices_for_belief].T)

        self.optimizer.set_initial(self.u, self.action_plan[::self.planning_simulation_ratio].T)

        # Give the belief parameters to the solver as parameters
        for belief_index, belief_point in enumerate(self.belief):
            casadi_belief_point = self.casadi_belief[belief_index]
            belief_list = belief_point.as_list()
            for i in range(10):
                self.optimizer.set_value(casadi_belief_point[i], belief_list[i])

        # Set the risk bounds of the agent as a parameter
        self.optimizer.set_value(self.risk_bound_agent, constraint_risk_bound)

        # Solver options
        p_opts = {"expand": True, 'ipopt.print_level': 0, 'print_time': 0,
                  "ipopt.max_iter": 3000,
                  "ipopt.nlp_scaling_method": "none",
                  "ipopt.dual_inf_tol": 100,
                  "ipopt.bound_relax_factor": 0.01,
                  "ipopt.compl_inf_tol": 0.1,
                  "ipopt.constr_viol_tol": 0.01,
                  "ipopt.acceptable_constr_viol_tol": 1.,
                  }

        self.optimizer.solver("ipopt", p_opts)

        try:
            solution = self.optimizer.solve()
            self.action_plan = np.repeat(solution.value(self.u).T, self.planning_simulation_ratio, axis=0)
            self._update_position_plan()

        except RuntimeError as e:
            print(str(e))
            print(Fore.GREEN + " Optimization failed produced RuntimeError" + Style.RESET_ALL)
            print(self.optimizer.stats()['return_status'])
            self.optimization_failed = True

            self.action_plan[:, :] = 0
            planning_frequency = int((1000 / self.dt) / 2)
            self.action_plan[0:planning_frequency][:, 2] = -self.controllable_object.angular_velocity * 2
            self.action_plan[0:planning_frequency][:, 0:2] = -self.controllable_object.velocity * 2
            self.action_plan[0:planning_frequency] /= np.array(
                [self.controllable_object.max_long_acceleration, self.controllable_object.max_lat_acceleration,
                 self.controllable_object.max_angular_acceleration])
            self._update_position_plan()
            print(Fore.MAGENTA + '--- reset done ---' + Style.RESET_ALL)

    def _update_position_plan(self):
        """
        This function updates the current positions and velocity plans. The plan is made in terms of acceleration, but also stored in future positions and
        velocities for convenience. This function uses the current position and velocity and acceleration plan to update these predictions.
        :return:
        """
        previous_position = copy.copy(self.controllable_object.position)
        previous_velocity = copy.copy(self.controllable_object.velocity)
        previous_angular_velocity = copy.copy(self.controllable_object.angular_velocity)
        previous_heading = copy.copy(self.controllable_object.heading)

        for index in range(self.plan_length):
            a_long, a_lat, omega_dot = self.action_plan[index, :]

            a_long *= self.controllable_object.max_long_acceleration
            a_lat *= self.controllable_object.max_lat_acceleration
            omega_dot *= self.controllable_object.max_angular_acceleration

            previous_position, previous_velocity, previous_angular_velocity, previous_heading = self.controllable_object.calculate_time_step_2d(
                self.dt / 1000.,
                previous_position,
                previous_velocity,
                previous_heading,
                previous_angular_velocity,
                a_long,
                a_lat,
                omega_dot)

            self.heading_plan[index] = previous_heading
            self.velocity_plan[index, :] = [*previous_velocity, previous_angular_velocity]
            self.position_plan[index, :] = previous_position

    def _continue_current_plan(self):
        self.action_plan = np.roll(self.action_plan, -1, axis=0)
        self.action_plan[-1, :] = self.action_plan[-2, :]

    def compute_discrete_input(self, dt):
        pass

    def compute_continuous_input(self, dt):
        if not self._is_initialized:
            self._observe_communication()
            self._update_belief()
            self._update_position_plan()
            self._update_plan(constraint_risk_bound=1.)
            self.perceived_risk = self._evaluate_risk()
            self._is_initialized = True
        else:
            self._observe_communication()
            self._update_belief()
            self._continue_current_plan()
            self._update_position_plan()
            self.perceived_risk = self._evaluate_risk()

            if self.optimization_failed:
                self.optimization_failed = False
                self.did_plan_update_on_last_tick = 3
                self._update_plan(constraint_risk_bound=0.9 * self.risk_threshold)
                self.perceived_risk = self._evaluate_risk()
            elif self.perceived_risk > self.risk_threshold:
                self.did_plan_update_on_last_tick = 1
                self._update_plan(constraint_risk_bound=0.75 * self.risk_threshold)
                self.perceived_risk = self._evaluate_risk()
            else:
                self.did_plan_update_on_last_tick = 0

        return self.action_plan[0, :]

    @property
    def name(self):
        pass
