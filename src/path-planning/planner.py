#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Path Planner
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2019-11-26
# ---------------------------------------------------------------------------

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# The num of MPC actions, here include vx and vy
NUM_OF_ACTS = 2

# The num of MPC states, here include px and py
NUM_OF_STATES = 2

NUM_OF_G_STATES = 1

# MPC parameters
lookahead_step_num = 60
lookahead_step_timeinterval = 0.1

# start point and end point
start_point = [0.0, 0.0]
end_point = [1, 1]

# obstacle coordinates
obstacles = [[0.3, 0.3], [0.8, 0.4], [0.5, 0.8], [0.7, 0.7]]
print(len(obstacles))
# threshold of safety
safety_r = 0.15

class FirstStateIndex:
	def __init__(self, n):
		self.px = 0
		self.py = self.px + n
		self.vx = self.py + n
		self.vy = self.vx + n - 1

class MPC:
	def __init__(self):
		self.first_state_index_ = FirstStateIndex(lookahead_step_num)
		self.num_of_x_ = NUM_OF_STATES * lookahead_step_num + NUM_OF_ACTS * (lookahead_step_num - 1)
		# self.num_of_g_ = NUM_OF_STATES * lookahead_step_num + NUM_OF_G_STATES * lookahead_step_num
		self.num_of_g_ = NUM_OF_STATES * lookahead_step_num + (NUM_OF_G_STATES + len(obstacles)) * lookahead_step_num

	def Solve(self, state):

		# define optimization variables
		x = SX.sym('x', self.num_of_x_)

		# define cost functions
		w_cte = 10.0
		w_dv = 1.0
		cost = 0.0

		# initial variables
		x_ = [0] * self.num_of_x_
		x_[self.first_state_index_.px] = state[0]
		x_[self.first_state_index_.py] = state[1]

		# penalty on states
		for i in range(lookahead_step_num - 1, lookahead_step_num):
			cte = (x[self.first_state_index_.px + i] - end_point[0])**2 + (x[self.first_state_index_.py + i] - end_point[1])**2
			cost += w_cte * cte
		# penalty on inputs
		for i in range(lookahead_step_num - 2):
			dvx = x[self.first_state_index_.vx + i + 1] - x[self.first_state_index_.vx + i]
			dvy = x[self.first_state_index_.vy + i + 1] - x[self.first_state_index_.vy + i]
			cost += w_dv*(dvx**2) + w_dv*(dvy**2)

		# define lowerbound and upperbound of x
		x_lowerbound_ = [-exp(10)] * self.num_of_x_
		x_upperbound_ = [exp(10)] * self.num_of_x_                                                                                  
		for i in range(self.first_state_index_.vx, self.num_of_x_):
			x_lowerbound_[i] = -0.3
			x_upperbound_[i] = 0.3

		# define lowerbound and upperbound of g constraints
		g_lowerbound_ = [0] * self.num_of_g_
		g_upperbound_ = [0] * self.num_of_g_

		g_lowerbound_[self.first_state_index_.px] = state[0]
		g_lowerbound_[self.first_state_index_.py] = state[1]

		g_upperbound_[self.first_state_index_.px] = state[0]
		g_upperbound_[self.first_state_index_.py] = state[1]

		for idx, obstacle in enumerate(obstacles):
					for i in range(lookahead_step_num):
						g_idx = self.first_state_index_.py + i + (1 + idx) * lookahead_step_num
						g_lowerbound_[g_idx] = safety_r**2
						g_upperbound_[g_idx] = exp(10)

		# define g constraints
		g = [SX(0)] * self.num_of_g_

		g[self.first_state_index_.px] = x[self.first_state_index_.px]
		g[self.first_state_index_.py] = x[self.first_state_index_.py]

		for i in range(lookahead_step_num - 1):
			curr_px_index = i + self.first_state_index_.px
			curr_py_index = i + self.first_state_index_.py
			curr_vx_index = i + self.first_state_index_.vx
			curr_vy_index = i + self.first_state_index_.vy

			curr_px = x[curr_px_index]
			curr_py = x[curr_py_index]
			curr_vx = x[curr_vx_index]
			curr_vy = x[curr_vy_index]

			next_px = x[1 + curr_px_index]
			next_py = x[1 + curr_py_index]

			next_m_px = curr_px + curr_vx * lookahead_step_timeinterval
			next_m_py = curr_py + curr_vy * lookahead_step_timeinterval

			# equality constraints
			g[1 + curr_px_index] = next_px - next_m_px
			g[1 + curr_py_index] = next_py - next_m_py

			# inequality constraints
			for idx, obstacle in enumerate(obstacles):
				g_idx = 1 + curr_py_index + (1 + idx) * lookahead_step_num
				g[g_idx] = (next_px - obstacle[0])**2 + (next_py - obstacle[1])**2

		# create the NLP
		nlp = {'x':x, 'f':cost, 'g':vertcat(*g)}

		# solver options
		opts = {}
		opts["ipopt.print_level"] = 0
		opts["print_time"] = 0

		solver = nlpsol('solver', 'ipopt', nlp, opts)

		# solve the NLP
		res = solver(x0=x_, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
		return res

mpc_ = MPC()
sol = mpc_.Solve(start_point)
# plot results
# plot results
fig = plt.figure(figsize=(7, 7))
planned_px = sol['x'][0:1 * lookahead_step_num]
planned_py = sol['x'][1 * lookahead_step_num:2 * lookahead_step_num]
plt.plot(planned_px, planned_py, 'o-', label='planned trajectory')

theta = np.arange(0, 2*np.pi, 0.01)

# Plotting the obstacles
for obstacle in obstacles:
    danger_x = obstacle[0] + (safety_r - 0.005) * np.cos(theta)
    danger_y = obstacle[1] + (safety_r - 0.005) * np.sin(theta)
    plt.plot(danger_x, danger_y, 'r--', label='danger area around obstacle {}'.format(obstacles.index(obstacle)))
    plt.plot(obstacle[0], obstacle[1], 'o', label='obstacle {}'.format(obstacles.index(obstacle)))

plt.plot(start_point[0], start_point[1], 'o', label='start point')
plt.plot(end_point[0], end_point[1], 'o', label='target point')

plt.legend(loc='upper left')
plt.axis('equal')
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.grid()
plt.show()