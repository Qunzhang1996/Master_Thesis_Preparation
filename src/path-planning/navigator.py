#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Path Planner
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2019-11-26
# Modify Date: 2020-06-07
# ---------------------------------------------------------------------------

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from math import sqrt
from matplotlib import animation
import copy
trajectory=[]
def onClick(event):
    global pause
    pause ^= True

class FirstStateIndex:
    def __init__(self, n):
        self.px = 0
        self.py = self.px + n
        self.vx = self.py + n
        self.vy = self.vx + n - 1

class MPC:
    def __init__(self, end_point, num_of_agent, safety_r, max_v=0.3, lookahead_step_num=5, lookahead_step_timeinterval=0.1):

        # The num of MPC actions, here include vx and vy
        NUM_OF_ACTS = 2

        # The num of MPC states, here include px and py
        NUM_OF_STATES = 2

        NUM_OF_G_STATES = num_of_agent

        self.end_point = end_point
        self.num_of_agent = num_of_agent
        self.safety_r = safety_r
        self.max_v = max_v
        self.lookahead_step_num = lookahead_step_num
        self.lookahead_step_timeinterval = lookahead_step_timeinterval
        self.first_state_index_ = FirstStateIndex(self.lookahead_step_num)
        self.num_of_x_ = NUM_OF_STATES * self.lookahead_step_num + NUM_OF_ACTS * (self.lookahead_step_num - 1)
        self.num_of_g_ = NUM_OF_STATES * self.lookahead_step_num + NUM_OF_G_STATES * self.lookahead_step_num

    def Solve(self, state, agent_state_pred):

        # define optimization variables
        x = SX.sym('x', self.num_of_x_)

        # define cost functions
        w_cte = 10.0
        w_dv = 1.0
        cost = 0.0

        # initial variables
        x_ = [0] * self.num_of_x_
        x_[self.first_state_index_.px:self.first_state_index_.py] = [state[0]] * self.lookahead_step_num
        x_[self.first_state_index_.py:self.first_state_index_.vx] = [state[1]] * self.lookahead_step_num
        x_[self.first_state_index_.vx:self.first_state_index_.vy] = [self.max_v] * (self.lookahead_step_num - 1)
        x_[self.first_state_index_.vy:self.num_of_x_]             = [self.max_v] * (self.lookahead_step_num - 1)

        # penalty on states
        for i in range(self.lookahead_step_num):
            cte = (x[self.first_state_index_.px + i] - self.end_point[0])**2 + (x[self.first_state_index_.py + i] - self.end_point[1])**2
            cost += w_cte * cte
        # penalty on inputs
        for i in range(self.lookahead_step_num - 2):
            dvx = x[self.first_state_index_.vx + i + 1] - x[self.first_state_index_.vx + i]
            dvy = x[self.first_state_index_.vy + i + 1] - x[self.first_state_index_.vy + i]
            cost += w_dv*(dvx**2) + w_dv*(dvy**2)

        # define lowerbound and upperbound of x
        x_lowerbound_ = [-exp(10)] * self.num_of_x_
        x_upperbound_ = [exp(10)] * self.num_of_x_
        for i in range(self.first_state_index_.vx, self.num_of_x_):
            x_lowerbound_[i] = -self.max_v
            x_upperbound_[i] = self.max_v

        # define lowerbound and upperbound of g constraints
        g_lowerbound_ = [0] * self.num_of_g_
        g_upperbound_ = [0] * self.num_of_g_

        g_lowerbound_[self.first_state_index_.px] = state[0]
        g_lowerbound_[self.first_state_index_.py] = state[1]

        g_upperbound_[self.first_state_index_.px] = state[0]
        g_upperbound_[self.first_state_index_.py] = state[1]

        for i in range(1 + self.first_state_index_.py + 1 * self.lookahead_step_num, self.num_of_g_):
            g_lowerbound_[i] = self.safety_r**2
            g_upperbound_[i] = exp(10)

        # define g constraints
        g = [None] * self.num_of_g_
        g[self.first_state_index_.px] = x[self.first_state_index_.px]
        g[self.first_state_index_.py] = x[self.first_state_index_.py]
        for i in range(self.num_of_agent):
            g[self.first_state_index_.py + (i + 1) * self.lookahead_step_num] = 0

        for i in range(self.lookahead_step_num - 1):
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

            next_m_px = curr_px + curr_vx * self.lookahead_step_timeinterval
            next_m_py = curr_py + curr_vy * self.lookahead_step_timeinterval

            # equality constraints
            g[1 + curr_px_index] = next_px - next_m_px
            g[1 + curr_py_index] = next_py - next_m_py

            # inequality constraints
            for j in range(self.num_of_agent):
                g[1 + curr_py_index + (j + 1) * self.lookahead_step_num] = (
                          next_px - agent_state_pred[j][1 + i][0])**2 + (next_py - agent_state_pred[j][1 + i][1])**2

        # create the NLP
        nlp = {'x':x, 'f':cost, 'g':vertcat(*g)}

        # solver options
        opts = {}
        opts["ipopt.print_level"] = 0
        opts["print_time"] = 0
        opts["ipopt.tol"] = 0.01
        opts["ipopt.compl_inf_tol"] = 0.001
        opts["ipopt.constr_viol_tol"] = 0.01

        solver = nlpsol('solver', 'ipopt', nlp, opts)

        # solve the NLP
        res = solver(x0=x_, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
        return res

def animate(i):

    global cur_pos, dist_to_goal, time_stamp, trajectory
    
    if not pause:
        # get predicted future states of the agents
        agent_pos_pred = []
        for j in range(num_of_agent):
            agent_state = []
            for k in range(lookahead_step_num):
                if time_stamp + k < len(agent_pt_list[j]):
                    agent_state.append(agent_pt_list[j][time_stamp + k])
                else:
                    agent_state.append(agent_pt_list[j][len(agent_pt_list[j]) - 1])
            agent_pos_pred.append(agent_state)

        if dist_to_goal > thres:

            # convert from DM to float
            cur_pos = list(map(float, cur_pos))

            # plot robot position
            current_pos.set_data(cur_pos[0], cur_pos[1])

            # plot agent positions
            agent_pos_1.set_data(agent_pos_pred[0][0][0], agent_pos_pred[0][0][1])
            danger_x = agent_pos_pred[0][0][0] + safety_r * np.cos(theta)
            danger_y = agent_pos_pred[0][0][1] + safety_r * np.sin(theta)
            agent_danger_zone_1.set_data(danger_x, danger_y)

            agent_pos_2.set_data(agent_pos_pred[1][0][0], agent_pos_pred[1][0][1])
            danger_x = agent_pos_pred[1][0][0] + safety_r * np.cos(theta)
            danger_y = agent_pos_pred[1][0][1] + safety_r * np.sin(theta)
            agent_danger_zone_2.set_data(danger_x, danger_y)

            agent_pos_3.set_data(agent_pos_pred[2][0][0], agent_pos_pred[2][0][1])
            danger_x = agent_pos_pred[2][0][0] + safety_r * np.cos(theta)
            danger_y = agent_pos_pred[2][0][1] + safety_r * np.sin(theta)
            agent_danger_zone_3.set_data(danger_x, danger_y)
            
            agent_pos_4.set_data(agent_pos_pred[3][0][0], agent_pos_pred[3][0][1])
            danger_x = agent_pos_pred[3][0][0] + safety_r * np.cos(theta)
            danger_y = agent_pos_pred[3][0][1] + safety_r * np.sin(theta)
            agent_danger_zone_4.set_data(danger_x, danger_y)

            # solve for optimal control actions
            sol = mpc_.Solve(cur_pos, agent_pos_pred)
            vx_opt = sol['x'][2 * lookahead_step_num]
            vy_opt = sol['x'][3 * lookahead_step_num - 1]

            # simulate forward
            cur_pos[0] = cur_pos[0] + vx_opt * lookahead_step_timeinterval
            cur_pos[1] = cur_pos[1] + vy_opt * lookahead_step_timeinterval

            dist_to_goal = sqrt((cur_pos[0] - end_point[0])**2 + (cur_pos[1] - end_point[1])**2)

            time_stamp += 1
            trajectory.append(copy.deepcopy(cur_pos))

            return current_pos, agent_pos_1, agent_danger_zone_1, agent_pos_2, agent_danger_zone_2, agent_pos_3, agent_danger_zone_3,agent_pos_4, agent_danger_zone_4
        
def plot_trajectory():
    x_vals, y_vals = zip(*trajectory)

    x_vals = np.array(x_vals).flatten()
    y_vals = np.array(y_vals).flatten()

    plt.figure(figsize=(7, 7))
    plt.plot(x_vals, y_vals, label='Trajectory')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    

    # MPC parameters
    lookahead_step_num = 10
    lookahead_step_timeinterval = 0.1

    # start point and end point of ego robot
    start_point = [0.0, 0.0]
    end_point = [1, 1]

    # agent velocity
    agent_vel = 0.3

    # start point and end point of agents
    agent_start = [[0.5, 0], [1, 0.5], [0, 1],[0,0.5]]
    agent_end   = [[0.5, 1], [0, 0.5], [1, 0],[1,0]]

    num_of_agent = len(agent_start)

    # threshold of safety
    safety_r = 0.3

    # max vx, vy
    max_v = 0.6

    agent_pt_list = []

    # calculate the coordinates of agents
    for i in range(num_of_agent):
        start_pos = agent_start[i]
        end_pos   = agent_end[i]

        if i == 0:
            agent_vel = 0.3
        elif i == 1:
            agent_vel = 0.5
        else:
            agent_vel = 0.2

        dist = sqrt((start_pos[0] - end_pos[0])**2 + (start_pos[1] - end_pos[1])**2)
        num_of_points = int(dist / (agent_vel * lookahead_step_timeinterval) + 1)
        xs = np.linspace(start_pos[0], end_pos[0], num_of_points)
        ys = np.linspace(start_pos[1], end_pos[1], num_of_points)
        point_list = []
        for agent_x, agent_y in zip(xs, ys):
            point_list.append([agent_x, agent_y])
        agent_pt_list.append(point_list)

    mpc_ = MPC(end_point=end_point,
               num_of_agent=num_of_agent,
               safety_r=safety_r,
               max_v=max_v,
               lookahead_step_num=lookahead_step_num,
               lookahead_step_timeinterval=lookahead_step_timeinterval)

    thres = 1e-2
    pause = False

    cur_pos = copy.deepcopy(start_point)
    dist_to_goal = sqrt((cur_pos[0] - end_point[0])**2 + (cur_pos[1] - end_point[1])**2)
    time_stamp = 0

    # create animation
    fig = plt.figure(figsize=(7, 7))
    fig.canvas.mpl_connect('button_press_event', onClick)

    theta = np.arange(0, 2*np.pi, 0.01)

    plt.plot(start_point[0], start_point[1], 'o', label='start point')
    plt.plot(end_point[0], end_point[1], 'o', label='target point')

    current_pos, = plt.plot([],[], ls='None', color='k', marker='o', label='current position')
    agent_pos_1, = plt.plot([],[], ls='None', color='r', marker='o', label='agent')
    agent_danger_zone_1, = plt.plot([],[], 'r--', label='danger zone')
    agent_pos_2, = plt.plot([],[], ls='None', color='r', marker='o')
    agent_danger_zone_2, = plt.plot([],[], 'r--')
    agent_pos_3, = plt.plot([],[], ls='None', color='r', marker='o')
    agent_danger_zone_3, = plt.plot([],[], 'r--')
    agent_pos_4, = plt.plot([],[], ls='None', color='r', marker='o')
    agent_danger_zone_4, = plt.plot([],[], 'r--')

    plt.legend(loc='upper left')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.axis('equal')
    plt.grid()

    ani = animation.FuncAnimation(fig, animate, frames=100, interval=200)

    plt.show()
    plot_trajectory()