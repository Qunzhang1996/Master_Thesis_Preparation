# path-planning
Collision avoidance path planning project for robots.
## Prerequisite
1. [CasADi](https://web.casadi.org/)
## Quick Start
1. `from navigator import MPC`
2. Inputs of `MPC class`:
    * current state of ego robot
    * current state & predicted future states of surrounding agents
3. Output of `MPC class`:
    * current control action of ego robot
## Useful Notes
1. Robot state vector contains x, y. Action vector contains vx, vy.

2. Properties of `MPC class`:
    * `lookahead_step_num`: The prediction horizon for MPC. Default is 5.
    * `lookahead_step_timeinterval`: The time interval between two consecutive states planned by MPC. Default is 0.1s. Make sure this equals to the time interval in your simulation.
    * `end_point`: The goal state of ego robot.
    * `num_of_agent`: The number of surrounding agents.
    * `safety_r`: The minimum distance allowed between ego robot and surrounding agents.
    * `max_v`: The maximum velocity allowed for ego robot. We assume max_vx=max_vy=max_v. Default is 0.3m/s.
    
3. Call function `Solve(state, agent_state_pred)` repeatedly in your main loop. Arguments:
    * `state`: The current state of ego robot. This should be a 1-D list `[ego_x0, ego_y0]`.
    * `agent_state_pred`: The current state and predicted future states of surrounding agents. This should be a 3-D list `[[[agent_1_x0, agent_1_y0], [agent_1_x1, agent_1_y1], ...], [[agent_2_x0, agent_2_y0], [agent_2_x1, agent_2_y1], ...], ...]`. The shape of this list is determined by `num_of_agent` and `lookahead_step_num`.
    
4. Adjustable parameters in `MPC class`:
    * `w_cte`: The larger, the faster ego robot moves towards the goal state.
    * `w_dv`: The larger, the smoother ego robot trajectory is.
