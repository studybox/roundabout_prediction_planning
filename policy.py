import numpy as np
#from commonroad.geometry.shape import Rectangle
#from commonroad.planning.planning_problem import PlanningProblem, GoalRegion
#from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle ,ObstacleType
#from commonroad.scenario.trajectory import State, Trajectory
#from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet
from commonroad.scenario.laneletcomplement import CurvePt, CurveIndex, LaneLetCurve, VecSE2, lerp_curve, get_curve_index, lerp_curve_with_ind
#from route import RoutePlanner
from frenet_planning_wrapper import opt, run_dist
#from commonroad.scenario.lanelet import LaneletNetwork
import logging
from ego_vehicle import project_on_path, project_on_lanelet, move_along_path

class RulebasedPolicy(object):
    def __init__(self, config, veh, env, single_reference=False):
        self.config = config
        self.veh = veh
        self.env = env
        self.logger = logging.getLogger("ego_{}".format(veh.id))
        self.logger.setLevel(logging.ERROR)
        file_handler = logging.FileHandler('policy_{}.log'.format(veh.id))
        formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.single_reference = single_reference

    def check_obstacle_on_lanelet(self, obstacle, lanelets):
        check_obs = 0#False
        if len(lanelets.intersection(obstacle.initial_shape_lanelet_ids)) > 0:
            # 1. if current obstacle on the road (lanelets)
            check_obs = 1#True

        else:
            # 2. if future obstacle state on the road (lanelets)
            for k in range(self.config["delta_step"], self.config["prediction_steps"]+self.config["delta_step"], self.config["delta_step"]):
                future_state = obstacle.state_at_time(self.veh.current_time_step+k)
                if future_state is None:
                    break
                if future_state.posF.ind[1] in lanelets:
                    check_obs = 2#True
                    break
            # 3. if future obstacle occupancy on the road (lanelets)
            if check_obs == 0:
                for k in range(self.config["delta_step"], self.config["prediction_steps"]+self.config["delta_step"], self.config["delta_step"]):
                    future_occupancy = obstacle.occupancy_at_time(self.veh.current_time_step+k)
                    if future_occupancy is None:
                        break
                    shape_lanelet_ids = set(self.env.lanelet_network.find_lanelet_by_shape(future_occupancy.shape))
                    if len(lanelets.intersection(shape_lanelet_ids)):
                        check_obs = 3#True
                        break
        return check_obs

    def check_obstacle_exit(self, obstacle):
        exit_lanelet_id = None
        if len(obstacle.initial_shape_lanelet_ids) ==1 and list(obstacle.initial_shape_lanelet_ids)[0] in self.env.goal_lanelet_ids:
            exit_lanelet_id = obstacle.initial_state.posF.ind[1]
        else:
            for k in range(self.config["delta_step"], self.config["prediction_steps"]+self.config["delta_step"], self.config["delta_step"]):
                future_state = obstacle.state_at_time(self.veh.current_time_step+k)
                if future_state is None:
                    break
                if future_state.posF.ind[1] in self.env.goal_lanelet_ids:
                    exit_lanelet_id = future_state.posF.ind[1]
                    break
        return exit_lanelet_id

    def _get_obs_tail_pos(self, obs, state):
        obs_tail_pos = -obs.obstacle_shape.length*0.5*np.array([np.cos(state.orientation),
                                                            np.sin(state.orientation)]) + state.position
        obs_tail_pos = VecSE2(obs_tail_pos[0], obs_tail_pos[1], state.orientation)
        return obs_tail_pos

    def _get_obs_head_pos(self, obs, state):
        obs_head_pos = obs.obstacle_shape.length*0.5*np.array([np.cos(state.orientation),
                                                            np.sin(state.orientation)]) + state.position
        obs_head_pos = VecSE2(obs_head_pos[0], obs_head_pos[1], state.orientation)
        return obs_head_pos

    def _check_future_front_obs(self, front_veh, future_front_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                     future_front_s, future_front_v, lt, t, merge_end):
        conflict_lanelet = self.env.lanelet_network.find_lanelet_by_id(conflict_lanelet_ids[0])
        check_obs = True
        #future_front_s = future_front_v = future_front_a = float('inf')
        future_front_a = 0.0
        if future_front_state is not None:
            # determine the front obs is on path or, on merging path
            front_obs_lanelet_id = future_front_state.posF.ind[1]
            if front_obs_lanelet_id in lanelet_occupancies or front_obs_lanelet_id in after_merge_lanelet_ids:
                # on same lanelet
                self.logger.debug("==MERGE_CK_FRONT: veh {} on same lane {}".format(front_veh.obstacle_id, front_obs_lanelet_id))
                obs_tail_pos = self._get_obs_tail_pos(front_veh, future_front_state)
                obs_tail_proj, future_front_s = project_on_path(obs_tail_pos, self.veh.path, self.veh.ind.i, self.veh.ind.i+50)
                future_front_v = np.cos(obs_tail_proj.phi) * future_front_state.velocity
                future_front_a = future_front_state.acceleration
            elif front_obs_lanelet_id in conflict_lanelet_ids:
                # on conflict lanelet
                self.logger.debug("==MERGE_CK_FRONT: veh {} on conflict lane {}".format(front_veh.obstacle_id, front_obs_lanelet_id))
                obs_tail_pos = self._get_obs_tail_pos(front_veh, future_front_state)
                obs_tail_proj, obs_tail_dist = project_on_lanelet(obs_tail_pos, conflict_lanelet)
                future_front_s = merge_end - (conflict_lanelet.center_curve[-1].s - obs_tail_dist)
                future_front_v = np.cos(obs_tail_proj.phi) * future_front_state.velocity
                self.logger.debug("==MERGE_CK_FRONT: pos: {} c: {}, v:{}".format(obs_tail_pos, np.cos(obs_tail_proj.phi), future_front_state.velocity))
                future_front_a = future_front_state.acceleration
            else:
                self.logger.warning("==MERGE_CK_FRONT: veh {} on wrong lane {}".format(front_veh.obstacle_id, front_obs_lanelet_id))
                check_obs = False
        else:
            self.logger.warning("==MERGE_CK_FRONT: veh {} has no pred at {}".format(front_veh.obstacle_id, t))
            future_front_s = future_front_v*(t-lt) + future_front_s
            future_front_a = 0.0

        if future_front_v <0 or future_front_v >20:
            self.logger.warning("==MERGE_CK_FRONT: veh {} speed {}".format(front_veh.obstacle_id, future_front_v))
        if future_front_a <-8.0 or future_front_a >4.0:
            self.logger.warning("==MERGE_CK_FRONT: veh {} acc {}".format(front_veh.obstacle_id, future_front_a))
        return check_obs, future_front_s, future_front_v, future_front_a

    def _check_future_back_obs(self, back_veh, future_back_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                     future_back_s, future_back_v, lt, t, merge_end):
        conflict_lanelet = self.env.lanelet_network.find_lanelet_by_id(conflict_lanelet_ids[0])
        check_obs = True
        #future_back_s = future_back_v = future_back_a = float('inf')
        future_back_a = 0.0
        if future_back_state is not None:
            # determine the front obs is on path or, on merging path
            back_obs_lanelet_id = future_back_state.posF.ind[1]
            if back_obs_lanelet_id in lanelet_occupancies or back_obs_lanelet_id in after_merge_lanelet_ids:
                self.logger.warning("==MERGE_CK_BACK: future back veh {} is on lane {}".format(back_veh.obstacle_id, back_obs_lanelet_id))
                # on same lanelet
                obs_head_pos = self._get_obs_head_pos(back_veh, future_back_state)
                obs_head_proj, future_back_s = project_on_path(obs_head_pos, self.veh.path, 0, self.veh.ind.i)
                future_back_v = np.cos(obs_head_proj.phi) * future_back_state.velocity
                future_back_a = future_back_state.acceleration
            elif back_obs_lanelet_id in conflict_lanelet_ids:
                # on conflict lanelet
                self.logger.debug("==MERGE_CK_BACK: veh {} on conflict lane {}".format(back_veh.obstacle_id, back_obs_lanelet_id))
                obs_head_pos = self._get_obs_head_pos(back_veh, future_back_state)
                obs_head_proj, obs_head_dist = project_on_lanelet(obs_head_pos, conflict_lanelet)
                future_back_s = merge_end - (conflict_lanelet.center_curve[-1].s - obs_head_dist)
                future_back_v = np.cos(obs_head_proj.phi) * future_back_state.velocity
                future_back_a = future_back_state.acceleration
            else:
                self.logger.warning("==MERGE_CK_BACK: veh {} on wrong lane {}".format(back_veh.obstacle_id, back_obs_lanelet_id))
                check_obs = False
        else:
            self.logger.warning("==MERGE_CK_BACK: veh {} has no pred at {}".format(back_veh.obstacle_id, t))
            future_back_s = future_back_v*(t-lt) + future_back_s
            future_back_a = 0.0

        if future_back_v <0 or future_back_v >20:
            self.logger.warning("==MERGE_CK_BACK: veh {} speed {}".format(back_veh.obstacle_id, future_back_v))
        if future_back_a <-8.0 or future_back_a >4.0:
            self.logger.warning("==MERGE_CK_BACK: veh {} acc {}".format(back_veh.obstacle_id, future_back_a))

        return check_obs, future_back_s, future_back_v, future_back_a

    def find_front_veh(self, s0, obstacles, obstacles_preds):
        # check if has front vehicle
        front_s = front_veh_head_s = front_veh_tail_s = float('inf')
        front_veh_v = float('inf')
        front_veh = front_veh_pred = None
        include_left = include_right = False
        check_obs = False
        if self.veh.current_d+0.5*self.veh._width >= 2.0:
            include_left = True
        elif self.veh.current_d-0.5*self.veh._width <= -2.0:
            include_right = True
        for i in range(self.veh.ind.i, len(self.veh.path)):
            s = self.veh.path[i].s
            if s - s0 > self.config["front_s_max"]:
                break
            future_path_lanelets = [self.veh.center_lanelet_occupancies[i]]
            path_lanelet = self.env.lanelet_network.find_lanelet_by_id(future_path_lanelets[0])
            if path_lanelet.adj_left is not None and include_left:
                future_path_lanelets.append(path_lanelet.adj_left)
            if path_lanelet.adj_right is not None and include_right:
                future_path_lanelets.append(path_lanelet.adj_right)
            future_path_lanelets = set(future_path_lanelets)
            self.logger.debug("FIND_FRONT: future_ego at: {}".format(future_path_lanelets))
            for obs_id, obstacle in obstacles.items():
                if isinstance(obstacle, StaticObstacle):
                    continue
                if obstacles_preds is not None and obs_id in obstacles_preds:
                    for k, pred in enumerate(obstacles_preds[obs_id]):
                        if self.single_reference and pred[1] < 0.3:
                            continue
                        obstacle._prediction = pred[0]
                        check_obs = self.check_obstacle_on_lanelet(obstacle, future_path_lanelets)
                        self.logger.debug(">>FIND_FRONT: check_obs: {} for veh: {} on {}".format(check_obs, obstacle.obstacle_id, obstacle.initial_shape_lanelet_ids))
                        if check_obs:
                            obs_head_pos = self._get_obs_head_pos(obstacle, obstacle.initial_state)
                            obs_head_proj, obs_head_s = project_on_path(obs_head_pos, self.veh.path, self.veh.ind.i, i)
                            # head-to-head distance to determine front yes/no
                            if obs_head_s < s0 + 0.5*self.veh._length:
                                self.logger.debug(">>FIND_FRONT: veh: {} not in front".format(obstacle.obstacle_id))
                                continue #no
                            if self.veh.current_state.posF.ind[1] in self.env.source_lanelet_ids and len(future_path_lanelets.intersection(obstacle.initial_shape_lanelet_ids)) == 0:
                                # obstacle on main road and ego on entrance road
                                self.logger.debug(">>FIND_FRONT: veh: {} on conflict road : {}".format(obstacle.obstacle_id, obstacle.initial_shape_lanelet_ids))
                                continue
                            obs_tail_pos = self._get_obs_tail_pos(obstacle, obstacle.initial_state)
                            obs_tail_proj, obs_tail_s = project_on_path(obs_tail_pos, self.veh.path, self.veh.ind.i, i)
                            # bumper-to-bumper distance to determine closeness
                            delta_s = max(0.0, obs_tail_s - (s0+0.5*self.veh._length))
                            self.logger.debug(">>FIND_FRONT: veh: {} delta_s: {}".format(obstacle.obstacle_id, delta_s))

                            if delta_s < front_s:
                                front_s = delta_s
                                front_veh_head_s = obs_head_s
                                front_veh_tail_s = obs_tail_s
                                front_veh_v = np.cos(obs_tail_proj.phi) * obstacle.initial_state.velocity
                                front_veh = obstacle.obstacle_id
                                front_veh_pred = k

                else:
                    check_obs = self.check_obstacle_on_lanelet(obstacle, future_path_lanelets)
                    self.logger.debug(">>FIND_FRONT: check_obs: {} for veh: {} on {}".format(check_obs, obstacle.obstacle_id, obstacle.initial_shape_lanelet_ids))
                    if check_obs:
                        obs_head_pos = self._get_obs_head_pos(obstacle, obstacle.initial_state)
                        obs_head_proj, obs_head_s = project_on_path(obs_head_pos, self.veh.path, self.veh.ind.i, i)
                        # head-to-head distance to determine front yes/no
                        if obs_head_s < s0 + 0.5*self.veh._length:
                            self.logger.debug(">>FIND_FRONT: veh: {} not in front".format(obstacle.obstacle_id))
                            continue #no
                        if self.veh.current_state.posF.ind[1] in self.env.source_lanelet_ids and len(future_path_lanelets.intersection(obstacle.initial_shape_lanelet_ids)) == 0:
                            # obstacle on main road and ego on entrance road
                            self.logger.debug(">>FIND_FRONT: veh: {} on conflict road : {}".format(obstacle.obstacle_id, obstacle.initial_shape_lanelet_ids))
                            continue
                        obs_tail_pos = self._get_obs_tail_pos(obstacle, obstacle.initial_state)
                        obs_tail_proj, obs_tail_s = project_on_path(obs_tail_pos, self.veh.path, self.veh.ind.i, i)
                        # bumper-to-bumper distance to determine closeness
                        delta_s = max(0.0, obs_tail_s - (s0+0.5*self.veh._length))
                        self.logger.debug(">>FIND_FRONT: veh: {} delta_s: {}".format(obstacle.obstacle_id, delta_s))

                        if delta_s < front_s:
                            front_s = delta_s
                            front_veh_head_s = obs_head_s
                            front_veh_tail_s = obs_tail_s
                            front_veh_v = np.cos(obs_tail_proj.phi) * obstacle.initial_state.velocity
                            front_veh = obstacle.obstacle_id
            if front_veh is not None:
                break

        if front_veh_pred is not None:
            obstacles[front_veh]._prediction = obstacles_preds[front_veh][front_veh_pred][0]
            self.logger.debug(">>FIND_FRONT: veh: {} pred_k : {}".format(front_veh, front_veh_pred))

            #print(front_veh_pred)
        if front_veh is not None:
            delta_v = self.veh.current_state.velocity - front_veh_v
            if delta_v <= 0:
                self.ttc = float('inf')
            else:
                self.ttc = front_s/delta_v
        else:
            self.ttc = float('inf')
        self.logger.warning("FIND TTC: front veh {} check_obs {} ttc {}".format(front_veh, check_obs, self.ttc))
        #print(front_veh, self.ttc, front_s)
        return front_veh_head_s, front_veh_tail_s, front_veh_v, front_veh

    def find_merge_vehs(self, s0, sE, obstacles, obstacles_preds, front_vehid=None, front_veh_head_s=None):
        # check merge vehicles from conflict lanelet
        current_lanelet_id = self.veh.center_lanelet_occupancies[self.veh.ind.i]
        current_lanelet = self.env.lanelet_network.find_lanelet_by_id(current_lanelet_id)
        successor = self.env.lanelet_network.find_lanelet_by_id(current_lanelet.successor[0])
        assert current_lanelet.adj_right is None
        if self.veh.current_d+0.5*self.veh._width >= 2.0:
            # consider both conflict lanelets
            check_both = True
            lanelet_occupancies = [current_lanelet_id, current_lanelet.adj_left]
        else:
            # consider right
            check_both = False
            lanelet_occupancies = [current_lanelet_id]
        back_lanelet_id = successor.predecessor[0]
        if back_lanelet_id == current_lanelet_id:
            back_lanelet_id = successor.predecessor[1]
        back_lanelet = self.env.lanelet_network.find_lanelet_by_id(back_lanelet_id)
        if back_lanelet.adj_right is not None:
            self.logger.warning("FIND MERGE: merge lane {} is not rightmost".format(back_lanelet_id))
        dist0 = back_lanelet.center_curve[-1].s
        front_s = back_s = float('inf')
        front_veh = back_veh = None

        ret_lanelet_occupancies = [current_lanelet_id, current_lanelet.adj_left]
        ret_merge_lanelets = [back_lanelet_id, back_lanelet.adj_left]
        ret_after_merge_lanelet_ids = [suc for suc in back_lanelet.successor if suc not in self.env.goal_lanelet_ids]
        left_lanelet = self.env.lanelet_network.find_lanelet_by_id(back_lanelet.adj_left)
        ret_after_merge_lanelet_ids.append(left_lanelet.successor[0])

        if check_both:
            merge_lanelets = ret_merge_lanelets
            after_merge_lanelet_ids = ret_after_merge_lanelet_ids
        else:
            merge_lanelets = [back_lanelet_id]
            after_merge_lanelet_ids = [suc for suc in back_lanelet.successor if suc not in self.env.goal_lanelet_ids]
        # sort the merging vehicle order
        merge_head_s, merge_tail_s, merge_v, merge_vehids, merge_veh_preds = [], [], [], [], []
        self.logger.debug("FIND_MERGE: merge onto lanes: {}".format(merge_lanelets))
        self.logger.debug("FIND_MERGE: after lanes: {}".format(after_merge_lanelet_ids))

        for obs_id, obstacle in obstacles.items():
            if isinstance(obstacle, StaticObstacle) or obstacle.obstacle_id == front_vehid:
                continue
            if obstacles_preds is not None and obs_id in obstacles_preds:
                for k, pred in enumerate(obstacles_preds[obs_id]):
                    if self.single_reference and pred[1] < 0.3:
                        continue
                    obstacle._prediction = pred[0]
                    check_obs = self.check_obstacle_on_lanelet(obstacle, set(merge_lanelets+after_merge_lanelet_ids))
                    self.logger.debug(">>FIND_MERGE: check_obs: {} for veh: {} on {}".format(check_obs, obstacle.obstacle_id, obstacle.initial_shape_lanelet_ids))
                    if check_obs:
                        # according to the prediction remove if the veh exit
                        exit_lanelet_id = self.check_obstacle_exit(obstacle)
                        if exit_lanelet_id is not None:# and self.env.exit_order(current_lanelet_id, exit_lane_id) < 0:
                            exit_lanelet = self.env.lanelet_network.find_lanelet_by_id(exit_lanelet_id)
                            exit_predecessor = self.env.lanelet_network.find_lanelet_by_id(exit_lanelet.predecessor[0])
                            if back_lanelet_id in exit_predecessor.successor:
                                self.logger.debug(">>FIND_MERGE: veh: {} exit on lane {}".format(obstacle.obstacle_id, exit_lanelet_id))
                                continue
                        # project back on the right side
                        obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(obstacle.initial_state.posF.ind[1])
                        self.logger.debug(">>FIND_MERGE: veh: {} back {} cur {}".format(obstacle.obstacle_id, back_lanelet_id, obs_current_lanelet.lanelet_id))

                        if len(obs_current_lanelet.successor) == 0:
                            self.logger.debug(">>FIND_MERGE: veh: {} on exit {}".format(obstacle.obstacle_id, obs_current_lanelet.lanelet_id))
                            continue

                        if back_lanelet_id in obs_current_lanelet.predecessor or obs_current_lanelet.lanelet_id in ret_lanelet_occupancies:
                            self.logger.debug(">>FIND_MERGE: veh: {} on intersection {}-{}".format(obstacle.obstacle_id, back_lanelet_id, obs_current_lanelet.lanelet_id))
                            continue
                        if obs_current_lanelet.adj_right:
                            obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(obs_current_lanelet.adj_right)
                        if back_lanelet_id in obs_current_lanelet.predecessor or obs_current_lanelet.lanelet_id in ret_lanelet_occupancies:
                            self.logger.debug(">>FIND_MERGE: veh: {} on intersection {}-{}".format(obstacle.obstacle_id, back_lanelet_id, obs_current_lanelet.lanelet_id))
                            continue
                        if obs_current_lanelet.adj_right is not None:
                            self.logger.warning(">>FIND MERGE: obs lane {} is not rightmost".format(obs_current_lanelet.lanelet_id))
                        # obstacle tail
                        obs_tail_pos = self._get_obs_tail_pos(obstacle, obstacle.initial_state)
                        obs_tail_proj, obs_tail_dist = project_on_lanelet(obs_tail_pos, obs_current_lanelet)
                        obs_tail_dist = obs_current_lanelet.center_curve[-1].s - obs_tail_dist
                        # obstacle head
                        obs_head_pos = self._get_obs_head_pos(obstacle, obstacle.initial_state)
                        obs_head_proj, obs_head_dist = project_on_lanelet(obs_head_pos, obs_current_lanelet)
                        obs_head_dist = obs_current_lanelet.center_curve[-1].s - obs_head_dist
                        while obs_current_lanelet.lanelet_id != back_lanelet_id:
                            successors = obs_current_lanelet.successor
                            if len(successors) > 1:
                                if successors[0] in self.env.goal_lanelet_ids:
                                    obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(successors[1])
                                else:
                                    obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(successors[0])
                            else:
                                obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(successors[0])
                            obs_tail_dist += obs_current_lanelet.center_curve[-1].s
                            obs_head_dist += obs_current_lanelet.center_curve[-1].s
                        obs_tail_s = sE - obs_tail_dist
                        obs_head_s = sE - obs_head_dist
                        if obs_head_s > front_veh_head_s:
                            self.logger.debug(">>FIND_MERGE: veh: {} in front veh's front".format(obstacle.obstacle_id))
                            continue
                        if np.abs(obs_head_s) > 50:
                             self.logger.debug(">>FIND_MERGE: veh: {} outside range".format(obstacle.obstacle_id))
                             continue
                        merge_head_s.append(obs_head_s)
                        merge_tail_s.append(obs_tail_s)
                        merge_v.append(np.cos(obs_tail_proj.phi) * obstacle.initial_state.velocity)
                        merge_vehids.append(obstacle.obstacle_id)
                        merge_veh_preds.append(k)
                        break
            else:
                check_obs = self.check_obstacle_on_lanelet(obstacle, set(merge_lanelets))
                self.logger.debug(">>FIND_MERGE: check_obs: {} for veh: {} on {}".format(check_obs, obstacle.obstacle_id, obstacle.initial_shape_lanelet_ids))
                if check_obs:
                    # according to the prediction remove if the veh exit
                    exit_lanelet_id = self.check_obstacle_exit(obstacle)
                    if exit_lanelet_id is not None:# and self.env.exit_order(current_lanelet_id, exit_lane_id) < 0:
                        exit_lanelet = self.env.lanelet_network.find_lanelet_by_id(exit_lanelet_id)
                        exit_predecessor = self.env.lanelet_network.find_lanelet_by_id(exit_lanelet.predecessor[0])
                        if back_lanelet_id in exit_predecessor.successor:
                            self.logger.debug(">>FIND_MERGE: veh: {} exit on lane {}".format(obstacle.obstacle_id, exit_lanelet_id))
                            continue
                    # project back on the right side
                    obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(obstacle.initial_state.posF.ind[1])
                    if len(obs_current_lanelet.successor) == 0:
                        self.logger.debug(">>FIND_MERGE: veh: {} on exit {}".format(obstacle.obstacle_id, obs_current_lanelet.lanelet_id))
                        continue

                    if back_lanelet_id in obs_current_lanelet.predecessor or obs_current_lanelet.lanelet_id in ret_lanelet_occupancies:
                        self.logger.debug(">>FIND_MERGE: veh: {} on intersection {}-{}".format(obstacle.obstacle_id, back_lanelet_id, obs_current_lanelet.lanelet_id))
                        continue

                    if obs_current_lanelet.adj_right:
                        obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(obs_current_lanelet.adj_right)
                    if obs_current_lanelet.adj_right is not None:
                        self.logger.warning(">>FIND MERGE: obs lane {} is not rightmost".format(obs_current_lanelet.lanelet_id))
                    # obstacle tail
                    obs_tail_pos = self._get_obs_tail_pos(obstacle, obstacle.initial_state)
                    obs_tail_proj, obs_tail_dist = project_on_lanelet(obs_tail_pos, obs_current_lanelet)
                    obs_tail_dist = obs_current_lanelet.center_curve[-1].s - obs_tail_dist
                    # obstacle head
                    obs_head_pos = self._get_obs_head_pos(obstacle, obstacle.initial_state)
                    obs_head_proj, obs_head_dist = project_on_lanelet(obs_head_pos, obs_current_lanelet)
                    obs_head_dist = obs_current_lanelet.center_curve[-1].s - obs_head_dist
                    while obs_current_lanelet.lanelet_id != back_lanelet_id:
                        successors = obs_current_lanelet.successor
                        if len(successors) > 1:
                            if successors[0] in self.env.goal_lanelet_ids:
                                obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(successors[1])
                            else:
                                obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(successors[0])
                        else:
                            obs_current_lanelet = self.env.lanelet_network.find_lanelet_by_id(successors[0])
                        obs_tail_dist += obs_current_lanelet.center_curve[-1].s
                        obs_head_dist += obs_current_lanelet.center_curve[-1].s
                    obs_tail_s = sE - obs_tail_dist
                    obs_head_s = sE - obs_head_dist
                    if obs_head_s > front_veh_head_s:
                        self.logger.debug(">>FIND_MERGE: veh: {} in front veh's front".format(obstacle.obstacle_id))
                        continue
                    if np.abs(obs_head_s) > 50:
                         self.logger.debug(">>FIND_MERGE: veh: {} outside range".format(obstacle.obstacle_id))
                         continue
                    merge_head_s.append(obs_head_s)
                    merge_tail_s.append(obs_tail_s)
                    merge_v.append(np.cos(obs_tail_proj.phi) * obstacle.initial_state.velocity)
                    merge_vehids.append(obstacle.obstacle_id)
                    merge_veh_preds.append(None)
        self.logger.debug("FIND_MERGE: vehs: {} Ss: {}".format(merge_vehids, merge_head_s))
        if len(merge_vehids) == 0:
            return None, None, None, None, None, None, None, None,\
                   ret_lanelet_occupancies, ret_after_merge_lanelet_ids, ret_merge_lanelets
        elif len(merge_vehids) == 1:
            if merge_veh_preds[0] is not None:
                self.logger.debug(">>FIND_MERGE: veh: {} pred_k : {}".format(merge_vehids[0], merge_veh_preds[0]))
                obstacles[merge_vehids[0]]._prediction = obstacles_preds[merge_vehids[0]][merge_veh_preds[0]][0]
                #print("m", merge_veh_preds[0])

            if merge_head_s[0] > s0 + 0.5*self.veh._length:
                return merge_head_s[0], merge_tail_s[0], merge_v[0], merge_vehids[0], None, None, None, None,\
                       ret_lanelet_occupancies, ret_after_merge_lanelet_ids, ret_merge_lanelets
            else:
                return None, None, None, None, merge_head_s[0], merge_tail_s[0], merge_v[0], merge_vehids[0],\
                       ret_lanelet_occupancies, ret_after_merge_lanelet_ids, ret_merge_lanelets
        else:
            order_ind = np.argsort(merge_head_s)
            for i in range(len(order_ind)):
                if merge_head_s[order_ind[i]] > s0 + 0.5*self.veh._length:
                    if merge_veh_preds[i] is not None:
                        self.logger.debug(">>FIND_MERGE: veh: {} pred_k : {}".format(merge_vehids[i], merge_veh_preds[i]))
                        obstacles[merge_vehids[i]]._prediction = obstacles_preds[merge_vehids[i]][merge_veh_preds[i]][0]

                        #print("mi", merge_veh_preds[i])
                    if i == 0:
                        return merge_head_s[order_ind[i]], merge_tail_s[order_ind[i]], merge_v[order_ind[i]], merge_vehids[order_ind[i]], None, None, None, None, \
                               ret_lanelet_occupancies, ret_after_merge_lanelet_ids, ret_merge_lanelets
                    else:
                        if merge_veh_preds[i-1] is not None:
                            self.logger.debug(">>FIND_MERGE: veh: {} pred_k : {}".format(merge_vehids[i-1], merge_veh_preds[i-1]))
                            obstacles[merge_vehids[i-1]]._prediction = obstacles_preds[merge_vehids[i-1]][merge_veh_preds[i-1]][0]
                            #print("mi-1", merge_veh_preds[i-1])

                        return merge_head_s[order_ind[i]], merge_tail_s[order_ind[i]], merge_v[order_ind[i]], merge_vehids[order_ind[i]], \
                               merge_head_s[order_ind[i-1]], merge_tail_s[order_ind[i-1]], merge_v[order_ind[i-1]], merge_vehids[order_ind[i-1]], \
                               ret_lanelet_occupancies, ret_after_merge_lanelet_ids, ret_merge_lanelets
            if merge_veh_preds[-1] is not None:
                self.logger.debug(">>FIND_MERGE: veh: {} pred_k : {}".format(merge_vehids[-1], merge_veh_preds[-1]))
                obstacles[merge_vehids[-1]]._prediction = obstacles_preds[merge_vehids[-1]][merge_veh_preds[-1]][0]
                #print("me", merge_veh_preds[-1])
            return None, None, None, None, \
                   merge_head_s[order_ind[-1]], merge_tail_s[order_ind[-1]], merge_v[order_ind[-1]], merge_vehids[order_ind[-1]],\
                   ret_lanelet_occupancies, ret_after_merge_lanelet_ids, ret_merge_lanelets


    def action(self, obstacles, obstacles_preds = None):
        # check whether on entrance road
        lanelet_id = self.veh.current_state.posF.ind[1]
        if lanelet_id in self.env.source_lanelet_ids:
            self.logger.debug("ENTRANCE TASK on lane {}".format(lanelet_id))
            return self.entrance_task(obstacles, obstacles_preds)
        else:
            self.logger.debug("MAIN TASK on lane {}".format(lanelet_id))
            return self.mainroad_task(obstacles, obstacles_preds)

    def entrance_task(self,  obstacles, obstacles_preds):
        # check if can merge
        # front veh and back veh
        s0 = self.veh.path[self.veh.ind.i].s + (self.veh.path[self.veh.ind.i+1].s - self.veh.path[self.veh.ind.i].s)*self.veh.ind.t
        self.logger.debug("Ego veh: {} s: {}".format(self.veh.id, s0))
        front_head_s, front_tail_s, front_v, front_vehid = self.find_front_veh(s0, obstacles, obstacles_preds)
        self.logger.debug("Front veh: {}, Head S: {} Tail S:{} V:{}".format(front_vehid, front_head_s, front_tail_s, front_v))
        for i in range(len(self.veh.path)):
            if self.veh.stopline_occupancies[i] == 1.0:
                sS = self.veh.path[i].s
            if self.veh.center_lanelet_occupancies[i] != self.veh.center_lanelet_occupancies[0]:
                sE = self.veh.path[i].s
                break
        else:
            self.logger.warning("Merge end not get")
        merge_front_head_s, merge_front_tail_s, merge_front_v, merge_front_vehid, \
         back_head_s, back_tail_s, back_v, back_vehid, \
         lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids = self.find_merge_vehs(s0, sE, obstacles, obstacles_preds, front_vehid, front_head_s)

        self.logger.debug("Merge veh: {}, Head S: {} Tail S:{} V:{}".format(merge_front_vehid, merge_front_head_s, merge_front_tail_s, merge_front_v))
        self.logger.debug("Back veh: {}, Head S: {} Tail S:{} V:{}".format(back_vehid, back_head_s, back_tail_s, back_v))

        self.veh.front_vehid = front_vehid
        self.veh.merge_front_vehid = merge_front_vehid
        self.veh.back_vehid = back_vehid
        front_on_same_lane = True
        if merge_front_vehid is not None and merge_front_tail_s<front_tail_s:
            self.logger.info("merge veh:{} over front veh: {} stop: {} merge_end: {}".format(merge_front_vehid, front_vehid, sS, sE))
            closest_front_vehid = merge_front_vehid
            closest_front_tail_s = merge_front_tail_s
            closest_front_v = merge_front_v
            front_on_same_lane = False
        else:
            closest_front_vehid = front_vehid
            closest_front_tail_s = front_tail_s
            closest_front_v = front_v

        if closest_front_vehid or back_vehid:
            self.enough_gap = False
            merge_option = self.merge_maneuver(closest_front_vehid, closest_front_tail_s, closest_front_v, front_on_same_lane,
                                               back_vehid, back_head_s, back_v, s0, sS, sE, obstacles, obstacles_preds,
                                               lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids)
            self.logger.debug("merge_option {}".format(merge_option))

            if merge_option:
                return merge_option
            else:
                if self.veh.action is not None and self.veh.action[0] == 2 and self.veh.current_sv <= 1e-2:
                    if front_vehid is not None:
                        follow_option = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v, obstacles, obstacles_preds)
                        self.logger.debug("follow_option {}".format(follow_option))
                        if follow_option:
                            return follow_option
                        else:
                            speed_option = self.speed_maneuver(obstacles, obstacles_preds)
                            if speed_option is not None:
                                self.logger.debug("speed_option {}".format(speed_option))
                                return speed_option
                            else:
                                #self.logger.debug("stop_option {}".format(stop_option))
                                return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                    else:
                        speed_option = self.speed_maneuver(obstacles, obstacles_preds)
                        if speed_option is not None:
                            self.logger.debug("speed_option {}".format(speed_option))
                            return speed_option
                        else:
                            #self.logger.debug("stop_option {}".format(stop_option))
                            return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                elif closest_front_vehid is not None and back_vehid is None:
                    if front_vehid is not None:
                        follow_option = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v, obstacles, obstacles_preds)
                        if follow_option:
                            return follow_option
                    self.logger.debug("cannot catch front vehicle")

                    #new_obstacles_preds = {}
                    #for vehid, preds in obstacles_preds.items():
                    #    new_obstacles_preds[vehid] = []
                    #    for pred, score in preds:
                    #        new_obstacles_preds[vehid].append((pred, 0.6))
                    #speed_option = self.speed_maneuver(obstacles, new_obstacles_preds)
                    #if speed_option is not None:
                    #    self.logger.debug("speed_option {}".format(speed_option))
                    #    return speed_option
                    #else:
                    #    self.logger.debug("stop_option {}".format(stop_option))
                    #    return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                    speed_option = self.speed_maneuver(obstacles, obstacles_preds)
                    if speed_option is not None:
                        self.logger.debug("speed_option {}".format(speed_option))
                        return speed_option
                    else:
                        #self.logger.debug("stop_option {}".format(stop_option))
                        return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)

                    #stop_option = self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                    #if stop_option is not None:
                    #    self.logger.debug("stop_option {}".format(stop_option))
                    #    return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                    #else:
                    #    speed_option = self.speed_maneuver(obstacles, obstacles_preds)
                    #    self.logger.debug("speed_option {}".format(speed_option))
                    #    return speed_option
                elif back_vehid is not None and closest_front_vehid is None:
                    self.logger.debug("overtake back vehicle")
                    speed_option = self.speed_maneuver(obstacles, obstacles_preds)
                    if speed_option is not None:
                        self.logger.debug("speed_option {}".format(speed_option))
                        return speed_option
                    else:
                        #self.logger.debug("stop_option {}".format(stop_option))
                        return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                elif self.enough_gap:
                    self.logger.debug("a good gap between two vehicles")
                    if front_vehid is not None:
                        follow_option = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v, obstacles, obstacles_preds)
                        self.logger.debug("follow_option {}".format(follow_option))
                        if follow_option:
                            return follow_option
                    speed_option = self.speed_maneuver(obstacles, obstacles_preds)
                    if speed_option is not None:
                        self.logger.debug("speed_option {}".format(speed_option))
                        return speed_option
                    else:
                        #self.logger.debug("stop_option {}".format(stop_option))
                        return self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                else:
                    self.logger.debug("other cases, stop first")
                    stop_option = self.stop_maneuver(s0, sS, obstacles, obstacles_preds)
                    self.logger.debug("stop_option {}".format(stop_option))
                    if stop_option:
                        return stop_option
                    elif front_vehid is not None:
                        follow_option = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v, obstacles, obstacles_preds)
                        self.logger.debug("follow_option {}".format(follow_option))
                        if follow_option:
                            return follow_option
                        else:
                            return self.speed_maneuver(obstacles, obstacles_preds)
                    else:
                        return self.speed_maneuver(obstacles, obstacles_preds)

        else:
            return self.speed_maneuver(obstacles, obstacles_preds)

    def mainroad_task(self,  obstacles, obstacles_preds):
        s0 = self.veh.path[self.veh.ind.i].s + (self.veh.path[self.veh.ind.i+1].s - self.veh.path[self.veh.ind.i].s)*self.veh.ind.t
        self.logger.debug("Ego veh: {} s: {}".format(self.veh.id, s0))
        front_head_s, front_tail_s, front_v, front_vehid = self.find_front_veh(s0, obstacles, obstacles_preds)
        self.logger.debug("Front veh: {}, Head S: {} Tail S:{} V:{}".format(front_vehid, front_head_s, front_tail_s, front_v))
        self.veh.front_vehid = front_vehid
        self.veh.merge_front_vehid = None
        self.veh.back_vehid = None

        if front_vehid:
            follow_option = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v, obstacles, obstacles_preds)
            self.logger.debug("follow_option {}".format(follow_option))
            if follow_option:
                return follow_option
            else:
                return self.speed_maneuver(obstacles, obstacles_preds)
        else:
            return self.speed_maneuver(obstacles, obstacles_preds)


    def follow_maneuver(self, s0, front_vehid, front_s, front_v, obstacles, obstacles_preds):
        # at most 3 seconds
        #dT = self.config["dt"]*self.config["delta_step"]
        #maxT = self.config["dt"]*self.config["prediction_steps"]
        dT = self.config["dt"]*self.config["delta_step"]*2
        maxT = self.config["dt"]*self.config["prediction_steps"]*2

        #candidate_Ts = np.arange(dT*4, maxT+0.1, dT)
        #candidate_Ts = np.array([0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. ])
        #candidate_ks = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        candidate_Ts = np.array([0.8,  1.2, 1.6, 2. , 2.4, 2.8,
                                  3.2, 3.6,  4. ,  4.4,  4.8,
                                  5.2, 5.6,  6. ])
        candidate_ks = np.array([20,  30,  40,  50,  60,  70,
                                 80,  90,  100,  110,  120,
                                 130,  140, 150])

        #candidate_gaps = [self.veh._length*i*0.25 for i in range(10)]
        #for i in range(1, 10):
        #    candidate_gaps.append(-self.veh._length*i*0.25)
        candidate_gaps = [0.5*i for i in range(5)]
        for i in range(1, 15):
            candidate_gaps.append(-0.5*i)


        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []
        lt = candidate_Ts[0]
        future_front_s = front_s
        future_front_v = front_v
        front_veh = obstacles[front_vehid]
        for t, step in zip(candidate_Ts, candidate_ks):
            #print(t, int(t/self.config["dt"]))
            future_time_step = self.veh.current_time_step + step#+ int(t/self.config["dt"])
            future_state = front_veh.state_at_time(future_time_step)
            if future_state is not None:
                obs_tail_pos = self._get_obs_tail_pos(front_veh, future_state)
                future_front_proj, future_front_s = project_on_path(obs_tail_pos, self.veh.path, self.veh.ind.i, self.veh.ind.i+50)
                future_front_v = np.cos(future_front_proj.phi) * future_state.velocity
                future_front_a = future_state.acceleration
                # check if the front veh leave the path
                if np.abs(future_front_proj.d - self.veh.current_d) > 0.5*self.veh._width + 0.5*front_veh.obstacle_shape.width + 1.0:
                    lt = t
                    continue
            else:
                # assume constant speed
                future_front_s = future_front_v*(t-lt) + future_front_s
                future_front_a = 0.0
            gap = self.config["mini_gap"]+self.config["time_head"]*future_front_v
            self.logger.debug("  ==FOLLOW: front veh: {} s: {} at t: {} v: {} a:{}".format(front_vehid, future_front_s, t, future_front_v, future_front_a))

            target_T.append(t)
            target_s.append(future_front_s - gap - 0.5*self.veh._length - s0)
            #target_s.append(future_front_s - s0)

            target_v.append(future_front_v)
            target_a.append(future_front_a)
            target_d.append(self.veh.desired_d)
            lt = t
        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        if len(target_T) > 0:
            #if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
            #    offset_d = np.array([0.0, -0.1, 0.1])
            #else:
            #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])
            if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
                offset_d = np.array([0.0])
            else:
                offset_d = np.array([0.0,  -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])


            action = dict(target_T=np.array(target_T),
                  target_s=np.array(target_s),
                  target_v=np.array(target_v),
                  target_a=np.array(target_a),
                  offset_s=np.array(candidate_gaps),
                  target_d=np.array(target_d),
                  offset_d=offset_d,
                  keep_velocity=0)
        else:
            return None

        self.logger.debug("==FOLLOW: target_t: {}".format(target_T))
        self.logger.debug("==FOLLOW: target_s: {}".format(target_s))
        self.logger.debug("==FOLLOW: target_v: {}".format(target_v))
        self.logger.debug("==FOLLOW: target_a: {}".format(target_a))
        self.logger.debug("==FOLLOW: offset_s: {}".format(candidate_gaps))
        if obstacles_preds is not None:
            ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))
        else:
            ret = opt(state, action, obstacles.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))
        '''
        for traj_idx in range(ret.num_traj):
            traj = ret.trajectories[traj_idx]
            if traj.feasible == 1:
                self.logger.debug("==FOLLOW: traj T: {} S:{} D:{} cost:{}".format(traj.final_t, traj.final_s,
                                                                         traj.final_d, traj.cost))
                                                                         '''
        # return the list of which action is valid
        if ret.success==1:
            return [0, ret.best_trajectory[0].final_t, ret.best_trajectory[0].final_s, ret.best_trajectory[0].final_sv, ret.best_trajectory[0].final_sa,
                    ret.best_trajectory[0].final_d, ret.best_trajectory[0].final_s-0.5*self.veh._length, ret.best_trajectory[0].final_s+0.5*self.veh._length]
        else:
            return None

    def speed_maneuver(self, obstacles, obstacles_preds):
        # at most 3 seconds
        #dT = self.config["dt"]*self.config["delta_step"]
        #maxT = self.config["dt"]*self.config["prediction_steps"]
        #candidate_Ts = np.arange(dT*4, maxT+0.1, dT)#np.arange(dT*4, maxT+dT, dT)
        dT = self.config["dt"]*self.config["delta_step"]*2
        maxT = self.config["dt"]*self.config["prediction_steps"]*2
        candidate_Ts = np.arange(dT*2, maxT+0.1, dT)#np.arange(dT*4, maxT+dT, dT)

        #candidate_Ts = np.array([0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. ])
        #candidate_ks = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])

        candidate_speeds = np.arange(0.0, 11.0, 1.0)
        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []
        for t in candidate_Ts:
            target_T.append(t)
            target_s.append(-1)
            target_v.append(10)
            target_a.append(0.0)
            target_d.append(self.veh.desired_d)
        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        self.logger.debug("==SPEED: target_t: {}".format(target_T))
        self.logger.debug("==SPEED: target_v: {}".format(target_v))
        self.logger.debug("==SPEED: target_a: {}".format(target_a))

        #if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
        #    offset_d = np.array([0.0, -0.1, 0.1])
        #else:
        #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])
        if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])

        action = dict(target_T=np.array(target_T),
                      target_s=np.array(target_s),
                      target_v=np.array(target_v),
                      target_a=np.array(target_a),
                      offset_s=candidate_speeds,
                      target_d=np.array(target_d),
                      offset_d=offset_d,
                      keep_velocity=1)
        if obstacles_preds is not None:
            ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))
        else:
            ret = opt(state, action, obstacles.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                  future_steps=list(range(0, self.config["prediction_steps"], 10)))
        #for traj_idx in range(ret.num_traj):
        #    traj = ret.trajectories[traj_idx]
            #if traj.feasible == 1:
            #    self.logger.debug("==SPEED: traj T: {} S:{} V:{} cost:{} D: {} N:{}".format(traj.final_t, traj.final_s, traj.final_sv, traj.cost, traj.final_d, traj.path_length))

        if ret.success==1:
            speed_option = [1, ret.best_trajectory[0].final_t, ret.best_trajectory[0].final_s, ret.best_trajectory[0].final_sv, ret.best_trajectory[0].final_sa,
                    ret.best_trajectory[0].final_d, ret.best_trajectory[0].final_sv-2.0, ret.best_trajectory[0].final_sv+2.0]
            self.logger.debug("speed_option {}".format(speed_option))
            return speed_option
        else:
            self.logger.warning("speed_option None")
            return None

    def merge_maneuver(self, front_vehid, front_tail_s, front_v, front_on_same_lane,
                             back_vehid, back_head_s, back_v, s0, dist2stop, merge_end, obstacles, obstacles_preds,
                             lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids):
        #dT = self.config["dt"] * self.config["delta_step"]
        #maxT = self.config["dt"] * self.config["prediction_steps"]
        #candidate_Ts = np.arange(dT*4, maxT+0.1, dT)# np.arange(dT*4, maxT+dT, dT)
        #candidate_Ts = np.array([0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. ])
        #candidate_ks = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        dT = self.config["dt"] * self.config["delta_step"]*2
        maxT = self.config["dt"] * self.config["prediction_steps"]*2

        candidate_Ts = np.array([0.8,  1.2, 1.6, 2. , 2.4, 2.8,
                                  3.2, 3.6,  4. ,  4.4,  4.8,
                                  5.2, 5.6,  6. ])
        candidate_ks = np.array([20,  30,  40,  50,  60,  70,
                                 80,  90,  100,  110,  120,
                                 130,  140, 150])

        conflict_lanelet = self.env.lanelet_network.find_lanelet_by_id(conflict_lanelet_ids[0])
        future_front_s = front_tail_s
        future_front_v = front_v
        future_back_s = back_head_s
        future_back_v = back_v
        lt = candidate_Ts[0]
        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []

        if front_vehid is not None and back_vehid is not None:
            self.logger.debug("==MERGE: Check both front {} and back {}".format(front_vehid, back_vehid))
            # both front and back vehicle
            front_veh = obstacles[front_vehid]
            back_veh = obstacles[back_vehid]
            for t, step in zip(candidate_Ts, candidate_ks):
                future_time_step = self.veh.current_time_step + step #int(t/self.config["dt"])
                future_front_state = front_veh.state_at_time(future_time_step)
                future_back_state = back_veh.state_at_time(future_time_step)

                check_front_obs, future_front_s, future_front_v, future_front_a = self._check_future_front_obs(front_veh, future_front_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_front_s, future_front_v, lt, t, merge_end)
                check_back_obs, future_back_s, future_back_v, future_back_a = self._check_future_back_obs(back_veh, future_back_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_back_s, future_back_v, lt, t, merge_end)
                lt = t

                avg_v = (future_front_v + future_back_v)*0.5
                #time_head = 1.5
                front_des_gap = self.config["mini_gap"]+self.config["time_head"]*avg_v + avg_v*(avg_v-future_front_v)/(2*np.sqrt(2*6))
                back_des_gap = self.config["mini_gap"]+self.config["time_head"]*future_back_v + future_back_v*(future_back_v-avg_v)/(2*np.sqrt(2*6))
                #candidate_gaps = [(i*0.5-2)*front_des_gap for i in range(6)]
                self.logger.debug("  ==MERGE: check: {} front veh: {} s: {} at t: {} v: {} a:{}".format(check_front_obs, front_vehid, future_front_s, t, future_front_v, future_front_a))
                self.logger.debug("  ==MERGE: check: {} back veh: {} s: {} at t: {} v: {} a:{}".format(check_back_obs, back_vehid, future_back_s, t, future_back_v, future_back_a))
                #if check_front_obs and future_front_s >= max(dist2stop, s0+0.5*self.veh._length+ front_des_gap):
                if future_front_s >= max(dist2stop, s0+0.5*self.veh._length+ front_des_gap):
                    # in the merging area
                    self.logger.debug("  ==MERGE: front veh:{} in merging area".format(front_vehid))
                    self.logger.debug("  ==MERGE: back veh:{} gap{}-{} ".format(back_vehid, future_front_s-front_des_gap, future_back_s+back_des_gap+self.veh._length))

                    #if not check_back_obs or future_front_s - front_des_gap > future_back_s + back_des_gap + self.veh._length:
                    if future_front_s - front_des_gap > future_back_s + back_des_gap + self.veh._length:
                        self.logger.debug("  ==MERGE: veh:{}-veh:{} enough gap".format(front_vehid, back_vehid))
                        self.enough_gap = True
                        # enough gap
                        target_T.append(t)
                        target_s.append(future_front_s - front_des_gap - 0.5*self.veh._length - s0)
                        target_v.append((future_front_v + future_back_v)*0.5)
                        target_a.append(future_front_a)
                        target_d.append(self.veh.desired_d)
        elif front_vehid is not None:
            # only front vehicle
            front_veh = obstacles[front_vehid]
            self.logger.debug("==MERGE: Check front {}".format(front_vehid))
            for t, step in zip(candidate_Ts, candidate_ks):
                future_time_step = self.veh.current_time_step + step#int(t/self.config["dt"])
                future_front_state = front_veh.state_at_time(future_time_step)
                check_front_obs, future_front_s, future_front_v, future_front_a = self._check_future_front_obs(front_veh, future_front_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_front_s, future_front_v, lt, t, merge_end)
                lt = t
                front_des_gap = self.config["mini_gap"]+self.config["time_head"]*future_front_v
                self.logger.debug("  ==MERGE: front veh: {} s: {} at t: {} v: {} a:{}".format(front_vehid, future_front_s, t, future_front_v, future_front_a))

                #candidate_gaps = [(i*0.5-2)*front_des_gap for i in range(6)]

                if check_front_obs and future_front_s >= max(dist2stop, s0+0.5*self.veh._length+front_des_gap):
                    # in the merging area
                    target_T.append(t)
                    target_s.append(future_front_s - front_des_gap - 0.5*self.veh._length - s0)
                    target_v.append((future_front_v+self.veh.current_sv)*0.5)
                    target_a.append(future_front_a)
                    target_d.append(self.veh.desired_d)
        else:
            # only back vehicle
            back_veh = obstacles[back_vehid]
            self.logger.debug("==MERGE: Check back {}".format(back_vehid))
            for t, step in zip(candidate_Ts, candidate_ks):
                future_time_step = self.veh.current_time_step + step#int(t/self.config["dt"])
                future_back_state = back_veh.state_at_time(future_time_step)
                check_back_obs, future_back_s, future_back_v, future_back_a = self._check_future_back_obs(back_veh, future_back_state, lanelet_occupancies,
                                                 after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_back_s, future_back_v, lt, t, merge_end)
                lt = t
                back_des_gap = self.config["mini_gap"]+self.config["time_head"]*future_back_v
                #candidate_gaps = [i*0.5*back_des_gap for i in range(5)]
                self.logger.debug("  ==MERGE: back veh: {} s: {} at t: {} v: {} a:{}".format(back_vehid, future_back_s, t, future_back_v, future_back_a))
                if check_back_obs and future_back_s + back_des_gap + self.veh._length >= dist2stop:
                    self.logger.debug("  ==MERGE: back veh:{} is relevant".format(back_vehid))
                    # enough gap
                    target_T.append(t)
                    target_s.append(future_back_s + back_des_gap + 0.5*self.veh._length- s0)
                    target_v.append((future_back_v+self.veh.current_sv)*0.5)
                    target_a.append(future_back_a)
                    target_d.append(self.veh.desired_d)

        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        self.logger.debug("==MERGE: target_t: {}".format(target_T))
        self.logger.debug("==MERGE: target_s: {}".format(target_s))
        self.logger.debug("==MERGE: target_a: {}".format(target_a))
        #candidate_gaps = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]
        candidate_gaps = [0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0]

        self.logger.debug("==MERGE: offset_s: {}".format(candidate_gaps))
        if len(target_T) > 0:
            #if self.veh.current_sv <= 4:# and self.action[1] < 2.6:
            #    offset_d = np.array([0.0, -0.1, 0.1])
            #else:
            #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])

            action = dict(target_T=np.array(target_T),
                          target_s=np.array(target_s),
                          target_v=np.array(target_v),
                          target_a=np.array(target_a),
                          offset_s=np.array(candidate_gaps),
                          target_d=np.array(target_d),
                          offset_d=np.array([0]),
                          keep_velocity=0)
            if obstacles_preds is not None:
                ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                          future_steps=list(range(0, self.config["prediction_steps"], 10)))
            else:
                ret = opt(state, action, obstacles.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                          future_steps=list(range(0, self.config["prediction_steps"], 10)))
        else:
            return None
        if ret.success==1:
            return [3, ret.best_trajectory[0].final_t, ret.best_trajectory[0].final_s, ret.best_trajectory[0].final_sv, ret.best_trajectory[0].final_sa,
                    ret.best_trajectory[0].final_d, ret.best_trajectory[0].final_s-0.5*self.veh._length, ret.best_trajectory[0].final_s+0.5*self.veh._length]
        else:
            return None

    def stop_maneuver(self, s0, dist2stop, obstacles, obstacles_preds):
        #candidate_Ts = np.arange(dT*4, 3.2, 0.2)
        #dT = self.config["dt"] * self.config["delta_step"]
        #maxT = self.config["dt"] * self.config["prediction_steps"]
        #candidate_Ts = np.arange(dT*4, maxT+0.1, dT)#np.arange(dT*4, maxT+dT, dT)
        dT = self.config["dt"] * self.config["delta_step"]*2
        maxT = self.config["dt"] * self.config["prediction_steps"] *2
        candidate_Ts = np.arange(dT*2, maxT+0.1, dT)#np.arange(dT*4, maxT+dT, dT)

        #candidate_gaps = [-0.1*i for i in range(30)]
        #candidate_gaps = [-0.2*i for i in range(10)]
        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []
        for t in candidate_Ts:
            target_T.append(t)
            #target_s.append(dist2stop-0.5*self.veh._length-s0)
            target_s.append(-1)
            target_v.append(0.0)
            target_a.append(0.0)
            #target_d.append(0.0)
            target_d.append(self.veh.desired_d)

        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        self.logger.debug("==STOP: target_t: {}".format(target_T))
        self.logger.debug("==STOP: target_s: {}".format(target_s))
        self.logger.debug("==STOP: target_v: {}".format(target_v))
        self.logger.debug("==STOP: target_a: {}".format(target_a))
        #self.logger.debug("==STOP: offset_s: {}".format(candidate_gaps))
        self.logger.debug("==STOP: dist2stop: {}".format(dist2stop))

        if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])


        action = dict(target_T=np.array(target_T),
              target_s=np.array(target_s),
              target_v=np.array(target_v),
              target_a=np.array(target_a),
              offset_s=np.array([0.0]),
              #offset_s=np.array(candidate_gaps),
              target_d=np.array(target_d),
              offset_d=offset_d,
              keep_velocity=1)
        if obstacles_preds is not None:
            ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                  future_steps=list(range(0, self.config["prediction_steps"], 10)))
        else:
            ret = opt(state, action, obstacles.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))
        self.logger.debug("==STOP: num traj {}".format(ret.num_traj))

        closest_stop = -float('inf')
        closest_opt = None
        for traj_idx in range(ret.num_traj):
            traj = ret.trajectories[traj_idx]
            if traj.feasible == 1:
                stop_at = s0 + traj.final_s + 0.5*self.veh._length
                self.logger.debug("==STOP: traj T: {} S:{} D:{} cost:{} SA:{}".format(traj.final_t, traj.final_s,
                                                                                traj.final_d, traj.cost, stop_at))
                if stop_at >= dist2stop - 0.5*self.veh._length and stop_at <= dist2stop + 1.5:
                    self.logger.debug("==STOP success {}".format(stop_at))
                    if closest_stop < stop_at:
                        closest_stop = stop_at
                        closest_opt = [2, traj.final_t, traj.final_s, traj.final_sv, traj.final_sa,
                                   traj.final_d, max(traj.final_t-0.2, 0.4), traj.final_t+0.2]

        return closest_opt
        '''
        self.logger.debug("==STOP success {}".format(ret.success))
        if ret.success == 1:
            self.logger.debug("==STOP: best traj N: {}".format(ret.best_trajectory[0].path_length))
            self.logger.debug("==STOP: best traj T: {}".format(ret.best_trajectory[0].t[ret.best_trajectory[0].path_length-1]))
            self.logger.debug("==STOP: best traj S: {}".format(ret.best_trajectory[0].s[ret.best_trajectory[0].path_length-1]))
            self.logger.debug("==STOP: best traj V: {}".format(ret.best_trajectory[0].speed_s[ret.best_trajectory[0].path_length-1]))
            return [2, ret.best_trajectory[0].final_t, ret.best_trajectory[0].final_s, ret.best_trajectory[0].final_sv, ret.best_trajectory[0].final_sa,
                    ret.best_trajectory[0].final_d, max(ret.best_trajectory[0].final_t-0.2, 0.2), ret.best_trajectory[0].final_t+0.2]
        else:
            return None
'''


class DecentralPolicy(RulebasedPolicy):
    def __init__(self, config, veh, env, lr=0.01, int_cooling=1.0, single_reference=False):
        super(DecentralPolicy, self).__init__(
            config, veh, env, single_reference=single_reference
        )

        self.action_candidates = []
        self.action_dists = []
        self.speed_candidates = []
        self.costs = []
        self.task = -1
        self.lr = lr
        self.int_cooling = self.cooling = int_cooling

    def action(self, obstacles, obstacles_preds=None, connected_preds=None):
        # check whether on entrance road
        lanelet_id = self.veh.current_state.posF.ind[1]
        if lanelet_id in self.env.source_lanelet_ids:
            self.logger.debug("ENTRANCE TASK on lane {}".format(lanelet_id))
            return self.entrance_task(obstacles, obstacles_preds, connected_preds)
        else:
            self.logger.debug("MAIN TASK on lane {}".format(lanelet_id))
            return self.mainroad_task(obstacles, obstacles_preds, connected_preds)

    def update_candidates(self):
        E = np.sum(self.action_dists * self.costs) # np.arrays
        H = -np.sum(self.action_dists* np.log(self.action_dists))
        self.logger.debug("UPDATE: D {} C {} E {} H {}".format(self.action_dists, self.costs, E, H))

        for i in range(len(self.action_dists)):
            adv = (self.costs[i] - E)/self.cooling
            up = adv + H + np.log(self.action_dists[i])
            print(adv,  H + np.log(self.action_dists[i]))

            self.action_dists[i] = self.action_dists[i] - self.lr * up
            self.logger.debug("UPDATE: adv {} up {}".format(adv, up))


        # normal
        self.action_dists = np.exp(self.action_dists)/sum(np.exp(self.action_dists))
        # cooling
        self.cooling = self.cooling * 0.99
        self.cooling = max(self.cooling, 0.01)

    def entrance_task(self, obstacles, obstacles_preds, connected_preds):
        self.task = 0
        s0 = self.veh.path[self.veh.ind.i].s + (self.veh.path[self.veh.ind.i+1].s - self.veh.path[self.veh.ind.i].s)*self.veh.ind.t
        self.logger.debug("Ego veh: {} s: {}".format(self.veh.id, s0))
        front_head_s, front_tail_s, front_v, front_vehid = self.find_front_veh(s0, obstacles, obstacles_preds)
        self.logger.debug("Front veh: {}, Head S: {} Tail S:{} V:{}".format(front_vehid, front_head_s, front_tail_s, front_v))
        for i in range(len(self.veh.path)):
            if self.veh.stopline_occupancies[i] == 1.0:
                sS = self.veh.path[i].s
            if self.veh.center_lanelet_occupancies[i] != self.veh.center_lanelet_occupancies[0]:
                sE = self.veh.path[i].s
                break
        else:
            self.logger.warning("Merge end not get")
        merge_front_head_s, merge_front_tail_s, merge_front_v, merge_front_vehid, \
         back_head_s, back_tail_s, back_v, back_vehid, \
         lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids = self.find_merge_vehs(s0, sE, obstacles, obstacles_preds, front_vehid, front_head_s)

        self.logger.debug("Merge veh: {}, Head S: {} Tail S:{} V:{}".format(merge_front_vehid, merge_front_head_s, merge_front_tail_s, merge_front_v))
        self.logger.debug("Back veh: {}, Head S: {} Tail S:{} V:{}".format(back_vehid, back_head_s, back_tail_s, back_v))

        # reset the distribution
        if front_vehid != self.veh.front_vehid or merge_front_vehid != self.veh.merge_front_vehid or back_vehid != self.veh.back_vehid:
            self.logger.debug("Reset ")
            #print("reset")
            self.action_candidates = []
            self.speed_candidates = []
            self.costs = []
            self.action_dists = np.ones(8+6)/(8+6)
            self.cooling = self.int_cooling#0.5
            # reset speed maneuvers samples
        self.veh.front_vehid = front_vehid
        self.veh.merge_front_vehid = merge_front_vehid
        self.veh.back_vehid = back_vehid
        front_on_same_lane = True
        if merge_front_vehid is not None and merge_front_tail_s<front_tail_s:
            self.logger.info("merge veh:{} over front veh: {} stop: {} merge_end: {}".format(merge_front_vehid, front_vehid, sS, sE))
            closest_front_vehid = merge_front_vehid
            closest_front_tail_s = merge_front_tail_s
            closest_front_v = merge_front_v
            front_on_same_lane = False
        else:
            closest_front_vehid = front_vehid
            closest_front_tail_s = front_tail_s
            closest_front_v = front_v

        if len(self.action_candidates) == 0:
            # init speed maneuvers
            self.action_candidates, self.costs = self.speed_maneuver(obstacles, obstacles_preds, connected_preds)
        else:
            self.action_candidates, self.costs = self.speed_maneuver(obstacles, obstacles_preds, connected_preds, candidate_speeds=self.speed_candidates)

        if front_vehid:
            #print("has follow ", front_vehid)
            act, cost = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v, obstacles, obstacles_preds, connected_preds)
            self.action_candidates.append(act)
            self.costs.append(cost)
        else:
            self.action_candidates.append(None)
            self.costs.append(100)

        if closest_front_vehid or back_vehid:
            #print("has merge ", closest_front_vehid, back_vehid)
            act, cost = self.merge_maneuver(closest_front_vehid, closest_front_tail_s, closest_front_v, front_on_same_lane,
                                                   back_vehid, back_head_s, back_v, s0, sS, sE, obstacles, obstacles_preds,
                                                   lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids, connected_preds)
            self.action_candidates.append(act)
            self.costs.append(cost)
        else:
            self.action_candidates.append(None)
            self.costs.append(100)

        act, cost = self.stop_maneuver(s0, sS, obstacles, obstacles_preds, connected_preds)
        self.action_candidates.append(act)
        self.costs.append(cost)
        self.costs = np.array(self.costs)
        self.update_candidates()

        #sorted_index = np.argsort(best_costs)
        #return self.action_candidates[np.argmax(self.action_dists)]
        names = []
        for s in self.speed_candidates:
            names.append("KS@{}".format(s))
        names.extend(["Follow", "Merge", "Stop"])

        return (self.action_candidates, self.action_dists, names)


    def mainroad_task(self, obstacles, obstacles_preds, connected_preds):
        s0 = self.veh.path[self.veh.ind.i].s + (self.veh.path[self.veh.ind.i+1].s - self.veh.path[self.veh.ind.i].s)*self.veh.ind.t
        self.logger.debug("Ego veh: {} s: {}".format(self.veh.id, s0))
        front_head_s, front_tail_s, front_v, front_vehid = self.find_front_veh(s0, obstacles, obstacles_preds)
        self.logger.debug("Front veh: {}, Head S: {} Tail S:{} V:{}".format(front_vehid, front_head_s, front_tail_s, front_v))

        # reset the distribution
        #
        if front_vehid != self.veh.front_vehid or self.task != 1:
            self.logger.debug("Reset ")
            self.action_candidates = []
            self.speed_candidates = []
            self.costs = []
            self.action_dists = np.ones(6+6)/(6+6)
            self.cooling = self.int_cooling#0.5
            # reset speed maneuvers samples
        self.task = 1

        self.veh.front_vehid = front_vehid
        self.veh.merge_front_vehid = None
        self.veh.back_vehid = None

        if len(self.action_candidates) == 0:
            # init speed maneuvers
            self.action_candidates, self.costs = self.speed_maneuver(obstacles, obstacles_preds, connected_preds)
        else:
            # update speed maneuvers
            self.action_candidates, self.costs = self.speed_maneuver(obstacles, obstacles_preds, connected_preds, candidate_speeds=self.speed_candidates)

        if front_vehid:
            act, cost = self.follow_maneuver(s0, front_vehid, front_tail_s, front_v,  obstacles, obstacles_preds, connected_preds)
            self.action_candidates.append(act)
            self.costs.append(cost)
        else:
            self.action_candidates.append(None)
            self.costs.append(100)
        self.costs = np.array(self.costs)
        self.update_candidates()

        #return self.action_candidates[np.argmax(self.action_dists)]
        names = []
        for s in self.speed_candidates:
            names.append("KS@{}".format(s))
        names.append("Follow")
        return (self.action_candidates, self.action_dists, names)

    def speed_maneuver(self, obstacles, obstacles_preds, connected_preds, candidate_speeds=None):
        # at most 3 seconds
        dT = self.config["dt"]*self.config["delta_step"]*2
        maxT = self.config["dt"]*self.config["prediction_steps"]*2
        candidate_Ts = np.arange(dT*2, maxT+0.1, dT)#np.arange(dT*4, maxT+dT, dT)

        connected_vehid = list(connected_preds.keys())[0]
        if connected_vehid in obstacles_preds:
            old_connected_veh_pred = obstacles_preds[connected_vehid]
        else:
            old_connected_veh_pred = []
        obstacles_preds[connected_vehid] = []

        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []
        for t in candidate_Ts:
            target_T.append(t)
            target_s.append(-1)
            target_v.append(10)
            target_a.append(0.0)
            target_d.append(self.veh.desired_d)
        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]

        self.logger.debug("==SPEED: cand v: {}".format(candidate_speeds))

        #self.logger.debug("==SPEED: target_t: {}".format(target_T))
        #self.logger.debug("==SPEED: target_v: {}".format(target_v))
        #self.logger.debug("==SPEED: target_a: {}".format(target_a))

        if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, 1.0, 2.0, 3.0, 3.5])

        action = dict(target_T=np.array(target_T),
                      target_s=np.array(target_s),
                      target_v=np.array(target_v),
                      target_a=np.array(target_a),
                      offset_s=np.arange(0.0, 11.0, 1.0),
                      target_d=np.array(target_d),
                      offset_d=offset_d,
                      keep_velocity=1)

        ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))

        best_trajs = [None for _ in range(11)]
        best_costs = [100.0 for _ in range(11)]
        for traj_idx in range(ret.num_traj):
            traj = ret.trajectories[traj_idx]
            if traj.feasible == 1:
                final_v = int(round(traj.final_sv))
                #assert traj.path_length == 76
                cost = self.opt_cost(traj, connected_preds[connected_vehid], 1, 10)
                if cost < best_costs[final_v]:
                    best_costs[final_v] = cost
                    best_trajs[final_v] = traj

        cand_trajs, cand_costs = [] , []
        #print("speed", best_costs)
        self.speed_candidates = []
        for i in range(5+6):
            cand_trajs.append(best_trajs[i])
            cand_costs.append(best_costs[i])
            self.speed_candidates.append(i)

        #if candidate_speeds is None:
        #    sorted_index = np.argsort(best_costs)
        #    self.speed_candidates = []
        #    for i in range(5+6):
        #        cand_trajs.append(best_trajs[sorted_index[i]])
        #        cand_costs.append(best_costs[sorted_index[i]])
        #        self.speed_candidates.append(sorted_index[i])
        '''
        else:
            num_invalid = 0
            for v in candidate_speeds:
                cand_trajs.append(best_trajs[v])
                cand_costs.append(best_costs[v])
                if best_costs[v] >= 100:
                    num_invalid += 1
            #print(cand_costs, candidate_speeds, num_invalid)
            if num_invalid > 0:
                cand_trajs, cand_costs = [], []
                sorted_index = np.argsort(best_costs)
                self.speed_candidates = []

                for i in range(5):
                    cand_trajs.append(best_trajs[sorted_index[i]])
                    cand_costs.append(best_costs[sorted_index[i]])
                    self.speed_candidates.append(sorted_index[i])
                self.action_dists = np.ones(len(self.action_dists))/len(self.action_dists)
                self.cooling = 0.5
        '''
        #print("cand cost", cand_costs)
        obstacles_preds[connected_vehid] = old_connected_veh_pred
        return cand_trajs, cand_costs

    def follow_maneuver(self, s0, front_vehid, front_s, front_v, obstacles, obstacles_preds, connected_preds):
        # at most 3 seconds
        dT = self.config["dt"]*self.config["delta_step"]*2
        maxT = self.config["dt"]*self.config["prediction_steps"]*2
        #candidate_Ts = np.arange(dT*4, maxT+0.1, dT)
        candidate_Ts = np.array([0.8,  1.2, 1.6, 2. , 2.4, 2.8,
                                  3.2, 3.6,  4. ,  4.4,  4.8,
                                  5.2, 5.6,  6. ])
        candidate_ks = np.array([20,  30,  40,  50,  60,  70,
                                 80,  90,  100,  110,  120,
                                 130,  140, 150])

        connected_vehid = list(connected_preds.keys())[0]
        if connected_vehid in obstacles_preds:
            old_connected_veh_pred = obstacles_preds[connected_vehid]
        else:
            old_connected_veh_pred = []
        obstacles_preds[connected_vehid] = []

        candidate_gaps = [0.5*i for i in range(5)]
        for i in range(1, 15):
            candidate_gaps.append(-0.5*i)
        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []
        lt = candidate_Ts[0]
        future_front_s = front_s
        future_front_v = front_v
        front_veh = obstacles[front_vehid]
        for t, step in zip(candidate_Ts, candidate_ks):
            #print(t, int(t/self.config["dt"]))
            future_time_step = self.veh.current_time_step + step#+ int(t/self.config["dt"])
            future_state = front_veh.state_at_time(future_time_step)
            if future_state is not None:
                obs_tail_pos = self._get_obs_tail_pos(front_veh, future_state)
                future_front_proj, future_front_s = project_on_path(obs_tail_pos, self.veh.path, self.veh.ind.i, self.veh.ind.i+50)
                future_front_v = np.cos(future_front_proj.phi) * future_state.velocity
                future_front_a = future_state.acceleration
                # check if the front veh leave the path
                if np.abs(future_front_proj.d - self.veh.current_d) > 0.5*self.veh._width + 0.5*front_veh.obstacle_shape.width + 1.0:
                    lt = t
                    continue
            else:
                # assume constant speed
                future_front_s = future_front_v*(t-lt) + future_front_s
                future_front_a = 0.0
            gap = self.config["mini_gap"]+self.config["time_head"]*future_front_v
            self.logger.debug("  ==FOLLOW: front veh: {} s: {} at t: {} v: {} a:{}".format(front_vehid, future_front_s, t, future_front_v, future_front_a))

            target_T.append(t)
            target_s.append(future_front_s - gap - 0.5*self.veh._length - s0)
            #target_s.append(future_front_s - s0)

            target_v.append(future_front_v)
            target_a.append(future_front_a)
            target_d.append(self.veh.desired_d)
            lt = t
        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        if len(target_T) > 0:

            if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
                offset_d = np.array([0.0])
            else:
                offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, 1.0, 2.0, 3.0, 3.5])


            action = dict(target_T=np.array(target_T),
                  target_s=np.array(target_s),
                  target_v=np.array(target_v),
                  target_a=np.array(target_a),
                  offset_s=np.array(candidate_gaps),
                  target_d=np.array(target_d),
                  offset_d=offset_d,
                  keep_velocity=0)
        else:
            return None, 100

        ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))

        obstacles_preds[connected_vehid] = old_connected_veh_pred

        # return the list of which action is valid
        if ret.success==1:
            #assert ret.best_trajectory[0].path_length == 76
            cost = self.opt_cost(ret.best_trajectory[0], connected_preds[connected_vehid], 1, 10)
            return ret.best_trajectory[0], cost
        else:
            return None, 100

    def merge_maneuver(self, front_vehid, front_tail_s, front_v, front_on_same_lane,
                             back_vehid, back_head_s, back_v, s0, dist2stop, merge_end, obstacles, obstacles_preds,
                             lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids, connected_preds):
        dT = self.config["dt"] * self.config["delta_step"]*2
        maxT = self.config["dt"] * self.config["prediction_steps"]*2
        #candidate_Ts = np.arange(dT*4, maxT+0.1, dT)# np.arange(dT*4, maxT+dT, dT)
        candidate_Ts = np.array([0.8,  1.2, 1.6, 2. , 2.4, 2.8,
                                  3.2, 3.6,  4. ,  4.4,  4.8,
                                  5.2, 5.6,  6. ])
        candidate_ks = np.array([20,  30,  40,  50,  60,  70,
                                 80,  90,  100,  110,  120,
                                 130,  140, 150])

        connected_vehid = list(connected_preds.keys())[0]
        if connected_vehid in obstacles_preds:
            old_connected_veh_pred = obstacles_preds[connected_vehid]
        else:
            old_connected_veh_pred = []
        obstacles_preds[connected_vehid] = []

        conflict_lanelet = self.env.lanelet_network.find_lanelet_by_id(conflict_lanelet_ids[0])
        future_front_s = front_tail_s
        future_front_v = front_v
        future_back_s = back_head_s
        future_back_v = back_v
        lt = candidate_Ts[0]
        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []

        if front_vehid is not None and back_vehid is not None:
            self.logger.debug("==MERGE: Check both front {} and back {}".format(front_vehid, back_vehid))
            # both front and back vehicle
            front_veh = obstacles[front_vehid]
            back_veh = obstacles[back_vehid]
            for t, step in zip(candidate_Ts, candidate_ks):
                future_time_step = self.veh.current_time_step + step #int(t/self.config["dt"])
                future_front_state = front_veh.state_at_time(future_time_step)
                future_back_state = back_veh.state_at_time(future_time_step)

                check_front_obs, future_front_s, future_front_v, future_front_a = self._check_future_front_obs(front_veh, future_front_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_front_s, future_front_v, lt, t, merge_end)
                check_back_obs, future_back_s, future_back_v, future_back_a = self._check_future_back_obs(back_veh, future_back_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_back_s, future_back_v, lt, t, merge_end)
                lt = t

                avg_v = (future_front_v + future_back_v)*0.5
                #time_head = 1.5
                front_des_gap = self.config["mini_gap"]+self.config["time_head"]*avg_v + avg_v*(avg_v-future_front_v)/(2*np.sqrt(2*6))
                back_des_gap = self.config["mini_gap"]+self.config["time_head"]*future_back_v + future_back_v*(future_back_v-avg_v)/(2*np.sqrt(2*6))
                #candidate_gaps = [(i*0.5-2)*front_des_gap for i in range(6)]
                self.logger.debug("  ==MERGE: check: {} front veh: {} s: {} at t: {} v: {} a:{}".format(check_front_obs, front_vehid, future_front_s, t, future_front_v, future_front_a))
                self.logger.debug("  ==MERGE: check: {} back veh: {} s: {} at t: {} v: {} a:{}".format(check_back_obs, back_vehid, future_back_s, t, future_back_v, future_back_a))
                #if check_front_obs and future_front_s >= max(dist2stop, s0+0.5*self.veh._length+ front_des_gap):
                if future_front_s >= max(dist2stop, s0+0.5*self.veh._length+ front_des_gap):
                    # in the merging area
                    self.logger.debug("  ==MERGE: front veh:{} in merging area".format(front_vehid))
                    self.logger.debug("  ==MERGE: back veh:{} gap{}-{} ".format(back_vehid, future_front_s-front_des_gap, future_back_s+back_des_gap+self.veh._length))

                    #if not check_back_obs or future_front_s - front_des_gap > future_back_s + back_des_gap + self.veh._length:
                    if future_front_s - front_des_gap > future_back_s + back_des_gap + self.veh._length:
                        self.logger.debug("  ==MERGE: veh:{}-veh:{} enough gap".format(front_vehid, back_vehid))
                        self.enough_gap = True
                        # enough gap
                        target_T.append(t)
                        target_s.append(future_front_s - front_des_gap - 0.5*self.veh._length - s0)
                        target_v.append((future_front_v + future_back_v)*0.5)
                        target_a.append(future_front_a)
                        target_d.append(self.veh.desired_d)
        elif front_vehid is not None:
            # only front vehicle
            front_veh = obstacles[front_vehid]
            self.logger.debug("==MERGE: Check front {}".format(front_vehid))
            for t, step in zip(candidate_Ts, candidate_ks):
                future_time_step = self.veh.current_time_step + step#int(t/self.config["dt"])
                future_front_state = front_veh.state_at_time(future_time_step)
                check_front_obs, future_front_s, future_front_v, future_front_a = self._check_future_front_obs(front_veh, future_front_state, lanelet_occupancies, after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_front_s, future_front_v, lt, t, merge_end)
                lt = t
                front_des_gap = self.config["mini_gap"]+self.config["time_head"]*future_front_v
                self.logger.debug("  ==MERGE: front veh: {} s: {} at t: {} v: {} a:{}".format(front_vehid, future_front_s, t, future_front_v, future_front_a))

                #candidate_gaps = [(i*0.5-2)*front_des_gap for i in range(6)]

                if check_front_obs and future_front_s >= max(dist2stop, s0+0.5*self.veh._length+front_des_gap):
                    # in the merging area
                    target_T.append(t)
                    target_s.append(future_front_s - front_des_gap - 0.5*self.veh._length - s0)
                    target_v.append((future_front_v+self.veh.current_sv)*0.5)
                    target_a.append(future_front_a)
                    target_d.append(self.veh.desired_d)
        else:
            # only back vehicle
            back_veh = obstacles[back_vehid]
            self.logger.debug("==MERGE: Check back {}".format(back_vehid))
            for t, step in zip(candidate_Ts, candidate_ks):
                future_time_step = self.veh.current_time_step + step#int(t/self.config["dt"])
                future_back_state = back_veh.state_at_time(future_time_step)
                check_back_obs, future_back_s, future_back_v, future_back_a = self._check_future_back_obs(back_veh, future_back_state, lanelet_occupancies,
                                                 after_merge_lanelet_ids, conflict_lanelet_ids,
                                                 future_back_s, future_back_v, lt, t, merge_end)
                lt = t
                back_des_gap = self.config["mini_gap"]+self.config["time_head"]*future_back_v
                #candidate_gaps = [i*0.5*back_des_gap for i in range(5)]
                self.logger.debug("  ==MERGE: back veh: {} s: {} at t: {} v: {} a:{}".format(back_vehid, future_back_s, t, future_back_v, future_back_a))
                if check_back_obs and future_back_s + back_des_gap + self.veh._length >= dist2stop:
                    self.logger.debug("  ==MERGE: back veh:{} is relevant".format(back_vehid))
                    # enough gap
                    target_T.append(t)
                    target_s.append(future_back_s + back_des_gap + 0.5*self.veh._length- s0)
                    target_v.append((future_back_v+self.veh.current_sv)*0.5)
                    target_a.append(future_back_a)
                    target_d.append(self.veh.desired_d)

        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        self.logger.debug("==MERGE: target_t: {}".format(target_T))
        self.logger.debug("==MERGE: target_s: {}".format(target_s))
        self.logger.debug("==MERGE: target_a: {}".format(target_a))
        #candidate_gaps = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]
        candidate_gaps = [0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0]

        self.logger.debug("==MERGE: offset_s: {}".format(candidate_gaps))
        if len(target_T) > 0:
            #if self.veh.current_sv <= 4:# and self.action[1] < 2.6:
            #    offset_d = np.array([0.0, -0.1, 0.1])
            #else:
            #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])

            action = dict(target_T=np.array(target_T),
                          target_s=np.array(target_s),
                          target_v=np.array(target_v),
                          target_a=np.array(target_a),
                          offset_s=np.array(candidate_gaps),
                          target_d=np.array(target_d),
                          offset_d=np.array([0]),
                          keep_velocity=0)
            if obstacles_preds is not None:
                ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                          future_steps=list(range(0, self.config["prediction_steps"], 10)))
            else:
                ret = opt(state, action, obstacles.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                          future_steps=list(range(0, self.config["prediction_steps"], 10)))
            obstacles_preds[connected_vehid] = old_connected_veh_pred
        else:
            obstacles_preds[connected_vehid] = old_connected_veh_pred
            return None, 100
        if ret.success==1:
            cost = self.opt_cost(ret.best_trajectory[0], connected_preds[connected_vehid], 1, 10)
            return ret.best_trajectory[0], cost
        else:
            return None, 100

    def stop_maneuver(self, s0, dist2stop, obstacles, obstacles_preds, connected_preds):
        #candidate_Ts = np.arange(dT*4, 3.2, 0.2)
        dT = self.config["dt"] * self.config["delta_step"]*2
        maxT = self.config["dt"] * self.config["prediction_steps"] *2
        candidate_Ts = np.arange(dT*2, maxT+0.1, dT)#np.arange(dT*4, maxT+dT, dT)
        connected_vehid = list(connected_preds.keys())[0]
        if connected_vehid in obstacles_preds:
            old_connected_veh_pred = obstacles_preds[connected_vehid]
        else:
            old_connected_veh_pred = []
        obstacles_preds[connected_vehid] = []
        #candidate_gaps = [-0.1*i for i in range(30)]
        #candidate_gaps = [-0.2*i for i in range(10)]
        target_T, target_s, target_v, target_a, target_d= [], [], [], [], []
        for t in candidate_Ts:
            target_T.append(t)
            #target_s.append(dist2stop-0.5*self.veh._length-s0)
            target_s.append(-1)
            target_v.append(0.0)
            target_a.append(0.0)
            #target_d.append(0.0)
            target_d.append(self.veh.desired_d)

        state = [self.veh.current_state.position[0], self.veh.current_state.position[1],
                 self.veh.current_state.orientation, self.veh.current_sv,
                 self.veh.current_sa, self.veh.current_dv, self.veh.current_da, self.veh.current_time_step,
                 self.veh._length, self.veh._width]
        self.logger.debug("==STOP: target_t: {}".format(target_T))
        self.logger.debug("==STOP: target_s: {}".format(target_s))
        self.logger.debug("==STOP: target_v: {}".format(target_v))
        self.logger.debug("==STOP: target_a: {}".format(target_a))
        #self.logger.debug("==STOP: offset_s: {}".format(candidate_gaps))
        self.logger.debug("==STOP: dist2stop: {}".format(dist2stop))

        if self.veh.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])


        action = dict(target_T=np.array(target_T),
              target_s=np.array(target_s),
              target_v=np.array(target_v),
              target_a=np.array(target_a),
              offset_s=np.array([0.0]),
              #offset_s=np.array(candidate_gaps),
              target_d=np.array(target_d),
              offset_d=offset_d,
              keep_velocity=1)
        if obstacles_preds is not None:
            ret = opt(state, action, obstacles_preds.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                  future_steps=list(range(0, self.config["prediction_steps"], 10)))
        else:
            ret = opt(state, action, obstacles.values(), self.veh.route.reference_path_segments[self.veh.route.principle_reference.index(1)],
                      future_steps=list(range(0, self.config["prediction_steps"], 10)))
        self.logger.debug("==STOP: num traj {}".format(ret.num_traj))
        obstacles_preds[connected_vehid] = old_connected_veh_pred
        closest_stop = -float('inf')
        closest_traj = None
        for traj_idx in range(ret.num_traj):
            traj = ret.trajectories[traj_idx]
            if traj.feasible == 1:
                stop_at = s0 + traj.final_s + 0.5*self.veh._length
                self.logger.debug("==STOP: traj T: {} S:{} D:{} cost:{} SA:{}".format(traj.final_t, traj.final_s,
                                                                                traj.final_d, traj.cost, stop_at))
                if stop_at >= dist2stop - 0.5*self.veh._length and stop_at <= dist2stop + 1.5:
                    self.logger.debug("==STOP success {}".format(stop_at))
                    if closest_stop < stop_at:
                        closest_stop = stop_at
                        closest_traj = traj
        if closest_traj is None:
            return None , 100
        #assert closest_traj.path_length == 76
        cost = self.opt_cost(closest_traj, connected_preds[connected_vehid], 1, 10)
        return closest_traj, cost

    def opt_cost(self, traj, obstacle_pred, delta_t, steps):
        traj_cost = self.compute_traj_cost(traj)
        total_cost = 0
        if len(obstacle_pred) == 0:
            return traj_cost
        for pred, score in obstacle_pred:
            pred_c = self.compute_pred_cost(pred)
            collision_c = self.compute_col_cost(traj, pred, delta_t, steps)
            total_cost += score * (traj_cost + pred_c + collision_c)
        return total_cost

    def compute_traj_cost(self, traj):
        vel_cost = 0.0
        acc_cost = 0.0
        d_cost = np.abs(self.veh.desired_d-traj.final_d)
        N = 0
        for i in range(traj.path_length):
            velocity=np.sqrt(traj.speed_s[i]**2+traj.speed_d[i]**2)
            vel_cost += np.abs(10-velocity)
            acc_cost += np.abs(traj.acc_s[i])

            N += 1
        return (vel_cost + acc_cost)/N# + d_cost * 10

    def compute_pred_cost(self, pred):
        vel_cost = 0.0
        acc_cost = 0.0
        N = 0
        for i in range(0, len(pred.trajectory.state_list), 5):
            state = pred.trajectory.state_list[i]
            vel_cost += np.abs(10-state.velocity)
            acc_cost += np.abs(state.acceleration)
            N += 1
        return (vel_cost + acc_cost)/N

    def compute_col_cost(self, traj, pred, delta_t, steps):
        col_cost = 0
        for t in range(0, traj.path_length, steps):
            occupancy = pred.occupancy_at_time_step(self.env.time_step+t*delta_t)
            if occupancy is not None:

                v = occupancy.shape.vertices
                col_dist = run_dist(traj.x[t], traj.y[t], traj.heading[t], self.veh._length, self.veh._width, \
                                    v[0,0], v[0,1], v[1,0], v[1,1], v[2,0], v[2,1], v[3,0], v[3,1])
                if col_dist < 1.5:
                    #print("col_dist", col_dist, t)
                    col_cost = 25
                    break
        return col_cost
