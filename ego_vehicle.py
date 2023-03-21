import warnings
from typing import Dict, List, Union
import numpy as np
import copy
import math
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, GoalRegion
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle ,ObstacleType
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet
from commonroad.scenario.laneletcomplement import CurvePt, CurveIndex, LaneLetCurve, VecSE2, lerp_curve, get_curve_index, lerp_curve_with_ind
from route import RoutePlanner
from frenet_planning_wrapper import opt
from commonroad.scenario.lanelet import LaneletNetwork

def make_path(vecs:FrenetState, lanelet_network:LaneletNetwork, width):
    dss = []
    ss = [0]
    selected_vecs = [vecs[0]]
    c_pt = 0
    for i_pt in range(1, len(vecs)):
        dx = vecs[i_pt].posG.x - vecs[c_pt].posG.x
        dy = vecs[i_pt].posG.y - vecs[c_pt].posG.y

        ds = np.sqrt(dx**2 + dy**2)
        if ds > 0.5:
            dss.append(np.sqrt(dx**2 + dy**2))
            ss.append(np.sqrt(dx**2 + dy**2))
            selected_vecs.append(vecs[i_pt])
            c_pt = i_pt
        elif i_pt == len(vecs)-1 and ds != 0.0:
            dss.append(np.sqrt(dx**2 + dy**2))
            ss.append(np.sqrt(dx**2 + dy**2))
            selected_vecs.append(vecs[i_pt])
    dss.append(dss[-1])
    ss = np.cumsum(ss)

    ks = []
    for i_pt in range(len(selected_vecs)-1):
        ks.append((selected_vecs[i_pt+1].posG.th - selected_vecs[i_pt].posG.th)/dss[i_pt])
    ks.append(ks[-1])
    kds = []
    for i_pt in range(len(selected_vecs)-1):
        kds.append((ks[i_pt+1] - ks[i_pt])/dss[i_pt])
    kds.append(kds[-1])
    curve = []
    lanelet_occupancies = []
    stopline_distance2 = []
    start_lanelet = lanelet_network.find_lanelet_by_id(selected_vecs[0].posF.ind[1])
    if len(start_lanelet.predecessor) == 0:
        merge_lanelet_ids = [start_lanelet.successor[0]]
        merge_lanelet = lanelet_network.find_lanelet_by_id(merge_lanelet_ids[0])
        if merge_lanelet.adj_left is not None:
            merge_lanelet_ids.append(merge_lanelet.adj_left)
        elif merge_lanelet.adj_right is not None:
            merge_lanelet_ids.append(merge_lanelet.adj_right)
    else:
        merge_lanelet_ids = []
    for i_pt in range(len(selected_vecs)):
        curve.append(CurvePt(selected_vecs[i_pt].posG, ss[i_pt], ks[i_pt], kds[i_pt]))
        lanelet = lanelet_network.find_lanelet_by_id(selected_vecs[i_pt].posF.ind[1])
        dist_left = lanelet.left_dist[selected_vecs[i_pt].posF.ind[0].i]
        dist_right = lanelet.right_dist[selected_vecs[i_pt].posF.ind[0].i]
        if selected_vecs[i_pt].posF.d+0.5*width > dist_left and lanelet.adj_left is not None:
            lanelet_occupancies.append(set([selected_vecs[i_pt].posF.ind[1], lanelet.adj_left]))
        elif selected_vecs[i_pt].posF.d-0.5*width < -dist_right and lanelet.adj_right is not None:
            lanelet_occupancies.append(set([selected_vecs[i_pt].posF.ind[1], lanelet.adj_right]))
        else:
            lanelet_occupancies.append(set([selected_vecs[i_pt].posF.ind[1]]))

        if lanelet.stop_line is not None:
            x,y = lanelet.stop_line.start
            stopline_distance2.append((selected_vecs[i_pt].posG.x-x)**2 + (selected_vecs[i_pt].posG.y-y)**2)
        else:
            stopline_distance2.append(1000)
    if min(stopline_distance2) == 1000:
        stopline_occupancies = [0 for _ in range(len(curve))]
    else:
        ind = np.argmin(stopline_distance2)
        stopline_occupancies = [0 for _ in range(len(curve))]
        stopline_occupancies[ind] = 1
    if len(merge_lanelet_ids) != 0:
        merge_inds = []
        for i_pt in range(len(lanelet_occupancies)):
            if len(lanelet_occupancies[i_pt].intersection(merge_lanelet_ids)) == 1:
                merge_inds.append(i_pt)
            elif len(lanelet_occupancies[i_pt].intersection(merge_lanelet_ids)) == 2:
                for ind in merge_inds:
                    lanelet_occupancies[ind] = set(merge_lanelet_ids)
                break
    return curve, lanelet_occupancies, stopline_occupancies

def project_on_path(posG:VecSE2, path:List[CurvePt], low_ind, high_ind):
    low_ind = max(low_ind-5, 0)
    high_ind = min(high_ind+5, len(path))
    closest_ind = posG.index_closest_to_point(path[low_ind:high_ind])+low_ind
    while closest_ind >= high_ind-1 and high_ind < len(path):
        high_ind += 1
        closest_ind = posG.index_closest_to_point(path[low_ind:high_ind])+low_ind
    proj = posG.proj_on_curve(path[low_ind:high_ind], clamped=False)
    s = lerp_curve_with_ind(path[low_ind:high_ind], proj.ind)
    return proj, s

def project_on_lanelet(posG: VecSE2, lanelet: LaneLetCurve):
    proj = posG.proj_on_curve(lanelet.center_curve)
    return proj, lerp_curve_with_ind(lanelet.center_curve, proj.ind)

def move_along_path(ind:CurveIndex, path:List[CurvePt], delta_s):
    next_ind, next_s = get_curve_index(ind, path, delta_s)
    if next_ind.t > 5:
        return None, None, None
    footpoint = lerp_curve(path[next_ind.i], path[next_ind.i+1], next_ind.t)
    position = np.array([footpoint.pos.x, footpoint.pos.y])
    #footpoint.pos.th = footpoint.pos._mod2pi(footpoint.pos.th)
    orientation = footpoint.pos._mod2pi(footpoint.pos.th)
    return next_ind, position, orientation

class GIDMVehicle(object):
    def __init__(self, id, states,
                 width, length, lanelet_network, sources, sinks,
                 amax=2.0, amin=-8.0, bconf=2.0, vstar=10, delta=4,
                 T = 1.5):
        self.id = id
        self._width = width
        self._length = length

        self.amax = amax
        self.amin = amin
        self.bconf = bconf
        self.v_star = vstar
        self.delta = delta
        self.front_s_max = 80
        self.back_s_max = 40
        self.mini_gap = 2.0
        self.long_shift = True
        self.blending = True
        self.cah = True
        self.T = T
        self.c = 0.5
        self.delta_dist_merge = 30
        self.TTM_min = 2.5
        self.sigma = 1.0

        self.check_merge = True
        # initial_state
        self.v = states[0].velocity
        self.a = 0.0
        self.s = 0.0
        self.path, self.center_lanelet_occupancies, self.stopline_occupancies = make_path(states, lanelet_network, width)
        self.ind = CurveIndex(0,0)
        #self.center_lanelet_laterials = [s.posF.d for s in states]
        #self.center_lanelet_occupancies = []
        #self.future_occupancy_lanelet_s = []
        self.action = None
        #self.shapes = [Rectangle(self._length, self._width, center=s.position, orientation=s.orientation) for s in states]
        #self.lanelet_occupancies = [lanelet_network.find_lanelet_by_shape(s) for s in self.shapes]
        self.lanelet_network = lanelet_network
        self.sources = sources
        self.sinks = sinks

        self.front_vehid = None
        self.front_v = 0.0
        self.front_a = 0.0
        self.front_s = float('inf')

        self.merge_front_vehid = None
        self.merge_front_v = 0.0
        self.merge_front_a = 0.0
        self.merge_front_s = float('inf')

        self.front_on_same_lane = True
        self.front_w = 1.0

        self.back_vehid = None
        self.back_v = 0.0
        self.back_a = 0.0
        self.back_s = float('inf')
        self.back_w = 1.0

        self.cool = 0.99
        self.TTM = None
        self.arrived = False

    def update_state(self, delta_s, v, acc, t):
        ind, position, orientation = move_along_path(self.ind, self.path, delta_s)
        if ind is None:
            self.arrived = True
            return None
        self.ind = ind
        self.v = v
        self.a = acc
        state = FrenetState(position=position,
                           orientation=orientation,
                           velocity=v,
                           acceleration=acc,
                           time_step=t+1)
        state.set_posF(self.lanelet_network, list(self.center_lanelet_occupancies[self.ind.i]))
        return state

    def _get_conflict_lanelets(self, set_lanelet_ids):
        # pass the stop line
        conflict_occupancy_lanelets = set()
        if len(set_lanelet_ids) == 2: # both
            for llid in set_lanelet_ids:
                ll = self.lanelet_network.find_lanelet_by_id(llid)
                sucid = ll.successor[0]
                suc = self.lanelet_network.find_lanelet_by_id(sucid)
                for prdid in suc.predecessor:
                    if prdid != llid:
                        conflict_occupancy_lanelets.add(sucid)
        else:
            llid = list(set_lanelet_ids)[0]
            ll = self.lanelet_network.find_lanelet_by_id(llid)
            if ll.adj_right is None or (ll.adj_right is None and ll.adj_left is None): # consider only right most
                sucid = ll.successor[0]
                suc = self.lanelet_network.find_lanelet_by_id(sucid)
                for prdid in suc.predecessor:
                    if prdid != llid:
                        conflict_occupancy_lanelets.add(sucid)
            else: #both
                for lllid in [llid, ll.adj_right]:
                    lll = self.lanelet_network.find_lanelet_by_id(lllid)
                    sucid = lll.successor[0]
                    suc = self.lanelet_network.find_lanelet_by_id(sucid)
                    for prdid in suc.predecessor:
                        if prdid != lllid:
                            conflict_occupancy_lanelets.add(sucid)
        return conflict_occupancy_lanelets

    def update_neighbors(self, obstacles):
        # check direct front vehicles along the path
        self.lane_change_merge = False
        self.s = s0 = self.path[self.ind.i].s + (self.path[self.ind.i+1].s - self.path[self.ind.i].s)*self.ind.t
        front_s = float('inf')
        front_v = 0
        front_a = 0
        front_veh = None
        for i in range(self.ind.i, len(self.path)):
            s = self.path[i].s
            if s - s0 > self.front_s_max:
                break
            conflict_occupancy_lanelets = set()
            # future occupancy
            future_occupancy_lanelets = self.center_lanelet_occupancies[i]
            #if self.stopline_occupancies[i] == 1:
            #    conflict_occupancy_lanelets = self._get_conflict_lanelets(future_occupancy_lanelets)

            for obstacle in obstacles:
                if len(future_occupancy_lanelets.intersection(obstacle.initial_shape_lanelet_ids)) > 0 or \
                   len(conflict_occupancy_lanelets.intersection(obstacle.initial_shape_lanelet_ids)) > 0:
                    obs_head_pos = obstacle.obstacle_shape.length*0.5*np.array([np.cos(obstacle.initial_state.orientation),
                                                                       np.sin(obstacle.initial_state.orientation)]) + obstacle.initial_state.position
                    obs_head_pos = VecSE2(obs_head_pos[0], obs_head_pos[1], obstacle.initial_state.orientation)
                    obs_head_proj, obs_head_s = project_on_path(obs_head_pos, self.path, self.ind.i, i)
                    # head-to-head distance to determine front yes/no
                    if obs_head_s < s0 + 0.5*self._length:
                        continue #no
                    obs_tail_pos = -obstacle.obstacle_shape.length*0.5*np.array([np.cos(obstacle.initial_state.orientation),
                                                                        np.sin(obstacle.initial_state.orientation)]) + obstacle.initial_state.position
                    obs_tail_pos = VecSE2(obs_tail_pos[0], obs_tail_pos[1], obstacle.initial_state.orientation)
                    obs_tail_proj, obs_tail_s = project_on_path(obs_tail_pos, self.path, self.ind.i, i)
                    # lateral distance to determine on path yes/no
                    if min(np.abs(obs_head_proj.d), np.abs(obs_tail_proj.d)) > 0.5*self._width + 0.5*obstacle.obstacle_shape.width + 0.8: #TODO
                        continue #no
                    # bumper-to-bumper distance to determine closeness
                    delta_s = max(0.0, obs_tail_s - (s0+0.5*self._length))
                    if delta_s < front_s:
                        front_s = delta_s
                        front_v =  np.cos(obs_head_proj.phi) * obstacle.initial_state.velocity
                        front_a = obstacle.initial_state.acceleration
                        front_veh = obstacle.obstacle_id
            if front_veh is not None:
                break

        merge_front_s = float('inf')
        merge_front_v = 0
        merge_front_a = 0
        merge_front_veh = None

        merge_back_s = float('inf')
        merge_back_v = 0
        merge_back_a = 0
        merge_back_veh = None
        current_lanelet_ids = self.center_lanelet_occupancies[self.ind.i]
        current_lanelet = self.lanelet_network.find_lanelet_by_id(list(current_lanelet_ids)[0])
        merge_pt = len(self.path)
        if len(current_lanelet.predecessor) == 0 and self.check_merge: # a source in roundabout
            # check merge vehicles from conflict lanelet
            for i in range(self.ind.i, len(self.path)):
                s = self.path[i].s
                future_occupancy_lanelets = self.center_lanelet_occupancies[i]
                if len(future_occupancy_lanelets.intersection(current_lanelet_ids)) == 0: # merge point
                    #if self.v == 0 and s - s0 > self.mini_gap+self._length:
                    #    break
                    #elif (s - s0)/self.v > self.TTM_min:
                    #    break
                    if len(future_occupancy_lanelets) == 2: # consider both
                        for lanelet_id in future_occupancy_lanelets:
                            pos_end = self.path[i].pos
                            ii = i
                            while lanelet_id in self.center_lanelet_occupancies[ii]:
                                pos_end = self.path[ii].pos
                                ii += 1
                                if ii == len(self.center_lanelet_occupancies):
                                    break

                            obs_front_s,\
                            obs_front_v,\
                            obs_front_a,\
                            obs_front_veh,\
                            obs_back_s,\
                            obs_back_v,\
                            obs_back_a,\
                            obs_back_veh = self.check_merge_lane(lanelet_id, obstacles, s0, s, self.path[i].pos, pos_end, self.long_shift)
                            #if self.id ==148 or self.id == 167:
                            #    print("both", obs_front_veh, obs_front_s, obs_front_v, self.id)
                            if obs_front_s < merge_front_s:
                                merge_front_s = obs_front_s
                                merge_front_v = obs_front_v
                                merge_front_a = obs_front_a
                                merge_front_veh = obs_front_veh
                            if obs_back_s < merge_back_s:
                                merge_back_s = obs_back_s
                                merge_back_v = obs_back_v
                                merge_back_a = obs_back_a
                                merge_back_veh = obs_back_veh
                    else:
                        lanelet_id = list(future_occupancy_lanelets)[0]
                        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
                        if lanelet.adj_right is None or (lanelet.adj_right is None and lanelet.adj_left is None): # consider only right most
                            #if self.id == 32:
                            #    print("right")
                            pos_end = self.path[i].pos
                            ii = i
                            while lanelet_id in self.center_lanelet_occupancies[ii]:
                                pos_end = self.path[ii].pos
                                ii += 1
                                if ii == len(self.center_lanelet_occupancies):
                                    break

                            obs_front_s,\
                            obs_front_v,\
                            obs_front_a,\
                            obs_front_veh,\
                            obs_back_s,\
                            obs_back_v,\
                            obs_back_a,\
                            obs_back_veh = self.check_merge_lane(lanelet_id, obstacles, s0, s, self.path[i].pos, pos_end, self.long_shift)
                            #if self.id ==148 or self.id == 167:
                            #    print("right", obs_back_veh, obs_back_s, merge_back_s, merge_back_veh, self.id)

                            if obs_front_s < merge_front_s:
                                merge_front_s = obs_front_s
                                merge_front_v = obs_front_v
                                merge_front_a = obs_front_a
                                merge_front_veh = obs_front_veh
                            if obs_back_s < merge_back_s:
                                merge_back_s = obs_back_s
                                merge_back_v = obs_back_v
                                merge_back_a = obs_back_a
                                merge_back_veh = obs_back_veh
                        else: # consider both
                            #if self.id == 32:
                            #    print("both 2")
                            for llid in [lanelet_id, lanelet.adj_right]:
                                pos_end = self.path[i].pos
                                ii = i
                                while llid in self.center_lanelet_occupancies[ii]:
                                    pos_end = self.path[ii].pos
                                    ii += 1
                                    if ii == len(self.center_lanelet_occupancies):
                                        break
                                obs_front_s,\
                                obs_front_v,\
                                obs_front_a,\
                                obs_front_veh,\
                                obs_back_s,\
                                obs_back_v,\
                                obs_back_a,\
                                obs_back_veh = self.check_merge_lane(llid, obstacles, s0, s, self.path[i].pos, pos_end, self.long_shift)
                                #if self.id ==148 or self.id == 167:
                                #    print("both 2", obs_back_veh, obs_back_s, merge_back_s, merge_back_veh, self.id)
                                if obs_front_s < merge_front_s:
                                    merge_front_s = obs_front_s
                                    merge_front_v = obs_front_v
                                    merge_front_a = obs_front_a
                                    merge_front_veh = obs_front_veh
                                if obs_back_s < merge_back_s:
                                    merge_back_s = obs_back_s
                                    merge_back_v = obs_back_v
                                    merge_back_a = obs_back_a
                                    merge_back_veh = obs_back_veh

                    merge_pt = i
                    break
        # check lane change along the neighbor lanelet
        for i in range(self.ind.i, len(self.path)):
            s = self.path[i].s
            #if s - s0 > self.front_s_max or i >= merge_pt:
            #    break
            if s - s0 > self.front_s_max:
                break
            # future occupancy
            future_occupancy_lanelets = self.center_lanelet_occupancies[i]
            if len(future_occupancy_lanelets) > 1:
                # merging to adjacent lanelet
                merging_lanelet_id = list(future_occupancy_lanelets)[1]
                if i > self.ind.i:
                    last_lanelet_id = list(self.center_lanelet_occupancies[i-1])[0]
                    last_lanelet = self.lanelet_network.find_lanelet_by_id(last_lanelet_id)
                    if merging_lanelet_id == last_lanelet_id: # current merging lanelet == last lanelet
                        merging_lanelet_id = list(future_occupancy_lanelets)[0]
                    elif merging_lanelet_id in last_lanelet.successor: # current merging lanelet == successor of last lanelet
                        merging_lanelet_id = list(future_occupancy_lanelets)[0]
                else: # if currently on middle of two lane, ignore
                    break
                # backward looking from the merging point (it is only for merging)
                pos_end = self.path[i].pos
                ii = i
                while merging_lanelet_id in self.center_lanelet_occupancies[ii]:
                    pos_end = self.path[ii].pos
                    ii += 1
                obs_front_s,\
                obs_front_v,\
                obs_front_a,\
                obs_front_veh,\
                obs_back_s,\
                obs_back_v,\
                obs_back_a,\
                obs_back_veh = self.check_merge_lane(merging_lanelet_id, obstacles, s0, s, self.path[i].pos, pos_end, self.long_shift)
                #if self.id ==148 or self.id == 167:
                #    print("merge", obs_back_veh, obs_back_s, merge_back_s, merge_back_veh, self.id)
                if obs_front_s < merge_front_s:
                    merge_front_s = obs_front_s
                    merge_front_v = obs_front_v
                    merge_front_a = obs_front_a
                    merge_front_veh = obs_front_veh
                    merge_pt = i
                    self.lane_change_merge = True
                if obs_back_s < merge_back_s:
                    merge_back_s = obs_back_s
                    merge_back_v = obs_back_v
                    merge_back_a = obs_back_a
                    merge_back_veh = obs_back_veh
                    merge_pt = i
                break

        self.front_vehid = front_veh
        self.front_v = front_v
        self.front_a = front_a
        self.front_s = front_s

        self.merge_front_vehid = merge_front_veh
        self.merge_front_v = merge_front_v
        self.merge_front_a = merge_front_a
        self.merge_front_s = merge_front_s

        self.back_vehid = merge_back_veh
        self.back_v = merge_back_v
        self.back_a = merge_back_a
        self.back_s = merge_back_s

        if self.blending and merge_pt < len(self.path):
            self.TTM = (self.path[merge_pt].s - s0 - 0.5*self._length)/(self.v+1e-5)
            stopline_pt = np.argmax(self.stopline_occupancies)
            if self.path[stopline_pt].s - s0 - 0.5*self._length > 10:
                self.front_w = self.back_w = 0.0
            elif self.path[stopline_pt].s - s0 - 0.5*self._length < 0 and self.path[stopline_pt].s - s0 - 0.5*self._length > -5:
                self.front_w = self.back_w = 1.0
            elif self.TTM >= self.TTM_min:
                self.front_w = self.back_w = np.exp(-(self.TTM-self.TTM_min)**2/(2*self.sigma**2))
            else:
                self.front_w = self.back_w = 1.0
        else:
            self.TTM = None
            self.front_w = self.back_w = 1.0
        self.merge_pt = merge_pt



    def check_merge_lane(self, merging_lanelet_id, obstacles, s0, s, pos_start, pos_end, long_shift):
        backward_lanelet_id = merging_lanelet_id
        backward_lanelet = self.lanelet_network.find_lanelet_by_id(backward_lanelet_id)
        merge_lanelet = backward_lanelet
        proj0, dist0 = project_on_lanelet(pos_start, backward_lanelet)
        projE, distE = project_on_lanelet(pos_end, backward_lanelet)
        merge_front_s = merge_front_s_raw = float('inf')
        merge_front_v = 0
        merge_front_a = 0
        merge_front_veh = None
        merge_back_s = merge_back_s_raw = float('inf')
        merge_back_v = 0
        merge_back_a = 0
        merge_back_veh = None
        while dist0 <= self.back_s_max:# + s - s0:
            for obstacle in obstacles:
                if isinstance(obstacle, StaticObstacle):
                    continue
                #if obstacle.obstacle_id == 30 and self.id ==38:
                #    print(dist0, backward_lanelet_id, obstacle.initial_shape_lanelet_ids)
                on_source_lanelet = False
                if len(merge_lanelet.predecessor) > 0:
                    for llid in obstacle.initial_center_lanelet_ids:
                        #ll = self.lanelet_network.find_lanelet_by_id(llid)
                        if llid in self.sources:
                            on_source_lanelet = True
                            break
                if on_source_lanelet:
                    continue
                if backward_lanelet_id in obstacle.initial_shape_lanelet_ids:
                    # obstacle in merging_lanelet
                    obs_tail_pos = -obstacle.obstacle_shape.length*0.5*np.array([np.cos(obstacle.initial_state.orientation),
                                                                        np.sin(obstacle.initial_state.orientation)]) + obstacle.initial_state.position
                    obs_tail_pos = VecSE2(obs_tail_pos[0], obs_tail_pos[1], obstacle.initial_state.orientation)
                    obs_tail_proj, obs_tail_dist = project_on_lanelet(obs_tail_pos, backward_lanelet)
                    dist_merge_tail = dist0 - obs_tail_dist
                    # tail to merge-pt distance to determine merge yes/no
                    if dist_merge_tail <= 0:
                        continue #no
                    obs_head_pos = obstacle.obstacle_shape.length*0.5*np.array([np.cos(obstacle.initial_state.orientation),
                                                                       np.sin(obstacle.initial_state.orientation)]) + obstacle.initial_state.position
                    # check a head pos prediction
                    head_lanelets = self.lanelet_network.find_lanelet_by_position([obs_head_pos])[0]
                    if len(head_lanelets) == 1 and head_lanelets[0] in self.sinks:
                        continue
                    obs_head_pos = VecSE2(obs_head_pos[0], obs_head_pos[1], obstacle.initial_state.orientation)
                    obs_head_proj, obs_head_dist = project_on_lanelet(obs_head_pos, backward_lanelet)
                    # lateral distance to determine on path yes/no
                    #if obstacle.obstacle_id == 30 and self.id ==38:
                    #    print(min(np.abs(obs_head_proj.d-proj0.d-np.sin(proj0.phi)*0.5*self._length), np.abs(obs_tail_proj.d-proj0.d-np.sin(proj0.phi)*0.5*self._length)) , 0.5*self._width + 0.5*obstacle.obstacle_shape.width + 1.5)
                    if min(np.abs(obs_head_proj.d-proj0.d-np.sin(proj0.phi)*0.5*self._length), \
                           np.abs(obs_tail_proj.d-proj0.d-np.sin(proj0.phi)*0.5*self._length), \
                           np.abs(obs_head_proj.d-projE.d-np.sin(projE.phi)*0.5*self._length), \
                           np.abs(obs_tail_proj.d-projE.d-np.sin(projE.phi)*0.5*self._length))  > 0.5*self._width + 0.5*obstacle.obstacle_shape.width + 1.0: #TODO
                        continue #no
                    dist_merge_head = dist0 - obs_head_dist
                    merge_v = np.cos(obs_head_proj.phi)*obstacle.initial_state.velocity
                    merge_a = obstacle.initial_state.acceleration
                    # head-to-head distance to determine front/back
                    max_dist_shift, W_shift, dist_shift = 0, 0, 0
                    obs_head_s = s - dist_merge_head
                    obs_tail_s = s - dist_merge_tail
                    if long_shift: #TODO
                        dist_merge_min = np.abs(0.5*self.v**2/self.amin)+0.5*self.v
                        dist_merge_max = dist_merge_min+self.delta_dist_merge
                        if s - s0 < dist_merge_min:
                            W_shift = 0
                        elif s - s0 > dist_merge_max:
                            W_shift = 1
                        else:
                            W_shift = (s-s0-dist_merge_min)/(dist_merge_max-dist_merge_min)
                        delta_v = self.v - merge_v
                        max_dist_shift = 0.5*(self.mini_gap + self.T * merge_v - (merge_v * delta_v) / (2 * math.sqrt(self.amax * self.bconf)))
                        #max_dist_shift += ((s-s0)/(self.v+1e-5) - dist_merge_tail/(merge_v+1e-5))*merge_v*0.2
                        dist_shift = max_dist_shift*W_shift
                        #obs_head_s = s - dist_merge_head + dist_shift
                        #obs_tail_s = s - dist_merge_tail + dist_shift
                    #else:
                    #    obs_head_s = s - dist_merge_head
                    #    obs_tail_s = s - dist_merge_tail
                    #if self.id ==148 or self.id == 167:
                    #    print(obs_head_s, obs_tail_s, s0, max_dist_shift, W_shift, obstacle.obstacle_id, self.id)
                    if obs_head_s >= s0+0.5*self._length: #front
                        # bumper-to-bumper distance to determine closeness
                        #delta_s = max(0.0, obs_tail_s - (s0+0.5*self._length))
                        delta_s = obs_tail_s - (s0+0.5*self._length)
                        #if self.id ==148 or self.id == 167:
                        #    print("mf", delta_s, merge_front_s, merge_front_veh, obstacle.obstacle_id, self.id)
                        if delta_s < merge_front_s_raw:
                            merge_front_s = max(0.0, delta_s)
                            merge_front_s_raw = delta_s
                            merge_front_v = merge_v
                            merge_front_a = merge_a
                            merge_front_veh = obstacle.obstacle_id
                    elif obs_head_s+dist_shift >= s0+0.5*self._length: #front after shift
                        delta_s = obs_tail_s + dist_shift - (s0+0.5*self._length)
                        if delta_s < merge_front_s_raw:
                            merge_front_s = max(0.0, delta_s)
                            merge_front_s_raw = delta_s
                            merge_front_v = merge_v
                            merge_front_a = merge_a
                            merge_front_veh = obstacle.obstacle_id
                    else: #back
                        # bumper-to-bumper distance to determine closeness
                        #delta_s = max(0.0, s0-0.5*self._length - obs_head_s)
                        delta_s = s0-0.5*self._length - obs_head_s - dist_shift
                        if delta_s <= 0.0:
                            if delta_s < merge_front_s_raw:
                                merge_front_s = max(0.0, delta_s)
                                merge_front_s_raw = delta_s
                                merge_front_v = merge_v
                                merge_front_a = merge_a
                                merge_front_veh = obstacle.obstacle_id
                        else:
                            if delta_s < merge_back_s_raw:
                                merge_back_s = max(0.0, delta_s)
                                merge_back_s_raw = delta_s
                                merge_back_v = merge_v
                                merge_back_a = merge_a
                                merge_back_veh = obstacle.obstacle_id
            if merge_front_veh is not None and merge_back_veh is not None:
                break
            if len(backward_lanelet.predecessor) == 0:
                break
            backward_lanelet_id = backward_lanelet.predecessor[0]
            backward_lanelet_ = self.lanelet_network.find_lanelet_by_id(backward_lanelet_id)
            if len(backward_lanelet_.predecessor) == 0:
                backward_lanelet_id = backward_lanelet.predecessor[1]
            backward_lanelet = self.lanelet_network.find_lanelet_by_id(backward_lanelet_id)
            dist0 += backward_lanelet.center_curve[-1].s
        return merge_front_s, merge_front_v, merge_front_a, merge_front_veh, merge_back_s, merge_back_v, merge_back_a, merge_back_veh

    def _s_star_front(self):
        delta_v =  self.v - self.front_v
        s_star = self.mini_gap + self.T * self.v + (self.v * delta_v) / (2 * math.sqrt(self.amax * self.bconf))
        s_star = max(0.0, s_star)

        delta_v =  self.v - self.merge_front_v
        s_star_merge = self.mini_gap + self.T * self.v + (self.v * delta_v) / (2 * math.sqrt(self.amax * self.bconf))
        s_star_merge = max(0.0, s_star_merge)

        return s_star, s_star_merge

    def _s_star_back(self):
        delta_v = self.v - self.back_v
        s_star = self.mini_gap + self.T * self.back_v - (self.back_v * delta_v) / (2 * math.sqrt(self.amax * self.bconf))
        s_star = max(0.0, s_star)
        return s_star

    def get_accel(self, obstacles):
        """
            calculate the acceleration of the vehicle based on surroundings
        """
        self.update_neighbors(obstacles)
        s_star_front, s_star_front_merge = self._s_star_front()
        s_star_back = self._s_star_back()

        free_drive = self.amax * (1 - math.pow(self.v/self.v_star, self.delta))
        if self.front_s == float('inf'):
            front_int = 0.0
            self.ttc = float('inf')
        elif self.front_s == 0.0:
            front_int = float('inf')
            self.ttc = 0.0
        else:
            front_int = self.amax * math.pow(s_star_front/self.front_s, 2)
            delta_v = self.v - self.front_v
            if delta_v <= 0:
                self.ttc = float('inf')
            else:
                self.ttc = self.front_s/delta_v

        if self.merge_front_s == float('inf'):
            front_merge_int = 0.0
            self.merge_ttc = float('inf')

        elif self.merge_front_s == 0.0:
            front_merge_int = float('inf')
            self.merge_ttc = 0.0
            if self.front_w <= 1e-4:
                front_merge_int = 0.0
                self.merge_ttc = float('inf')

        else:
            front_merge_int = self.amax * math.pow(s_star_front_merge/self.merge_front_s, 2) * self.front_w
            delta_v = self.v - self.merge_front_v
            if delta_v <= 0:
                self.merge_ttc = float('inf')
            else:
                self.merge_ttc = self.front_s/delta_v

        if front_int >= front_merge_int:
            self.front_on_same_lane = True
        else:
            front_int = front_merge_int
            self.front_on_same_lane = False
            '''
            delta_v = self.v - self.merge_front_v
            if delta_v <= 0:
                self.ttc = float('inf')
            else:
                self.tcc = self.merge_front_s/delta_v
            '''

        if self.back_s == float('inf'):
            back_int = 0.0
        elif self.back_s == 0.0:
            back_int = self.amax
        else:
            back_int = min(self.amax, self.amax * math.pow(s_star_back/self.back_s, 2) * self.back_w)
        accel = free_drive - front_int + self.c * back_int

        if self.cah:
            if self.front_on_same_lane:
                if self.front_s == float('inf'):
                    a_tilda = delta_v = 0.0
                    lead_v = self.v
                    lead_s = float('inf')
                else:
                    a_tilda = min(self.a, self.front_a)
                    delta_v = self.v - self.front_v
                    lead_v = self.front_v
                    lead_s = self.front_s
            else:
                if self.merge_front_s == float('inf'):
                    a_tilda = delta_v = 0.0
                    lead_v = self.v
                    lead_s = float('inf')
                else:
                    a_tilda = min(self.a, self.merge_front_a)
                    delta_v = self.v - self.merge_front_v
                    lead_v = self.merge_front_v
                    lead_s = self.merge_front_s
            if lead_v*delta_v <= -2*lead_s*a_tilda:
                if lead_v**2 - 2*lead_s*a_tilda == 0:
                    accel_cah = self.amax
                else:
                    accel_cah = self.v**2*a_tilda/(lead_v**2 - 2*lead_s*a_tilda)
            else:
                if lead_s == 0.0:
                    accel_cah = -float('inf')
                else:
                    accel_cah = a_tilda - delta_v**2/2/lead_s*(delta_v>0)
            if accel < accel_cah:
                accel = (1 - self.cool)*accel + self.cool*(accel_cah + self.bconf*np.tanh((accel-accel_cah)/self.bconf))

        accel = min(max(self.amin, accel), self.amax)


        return accel


class EgoVehicle(object):
    """
    Interface object for ego vehicle.
    How to use: After each simulation step, get current state with EgoVehicle.current_state()
    and set planned trajectory with EgoVehicle.set_planned_trajectory(planned_trajectory).
    """

    def __init__(self, id, initial_state: State, goal_state:State, width: float = 2.0, length: float = 5.0,
                  lanelet_network: LaneletNetwork = None):
        self.id = id
        self._width = width
        self._length = length
        self._initial_state = initial_state
        self.current_lanelet = initial_state.posF.ind[1]
        self._state_dict: Dict[State] = dict()  # collects driven states
        self._all_planned_trajectories = dict()
        self._all_planned_dists = dict()

        self.shape = Rectangle(self.length, self.width, center=np.array([0, 0]), orientation=0.0)
        self._driven_trajectory = None  # returns trajectory object of driven
        self._planned_trajectories: Dict[int, List[State]] = {}  # collects trajectories from planner for every time step
        self._current_time_step = initial_state.time_step
        self.delta_steps = 1
        self.plan_steps = 5
        self.prediction_steps = 75
        self.dt = 0.04
        self.front_s_max = 80
        initial_state.yaw_rate = 0.0
        initial_state.slip_angle = 0.0
        goal = GoalRegion([State(position=Rectangle(5, 4.5,
                                            center=goal_state.position,
                                            orientation=goal_state.orientation),
                           time_step=Interval(initial_state.time_step,goal_state.time_step+300))],
                           lanelets_of_goal_position = {0:[goal_state.posF.ind[1]]})
        self.planning_problem = PlanningProblem(id, initial_state, goal)
        self.arrived = self.planning_problem.goal.is_reached(initial_state)
        self.lanelet_network = lanelet_network
        route_planner = RoutePlanner(lanelet_network, self.planning_problem)
        candidate_holder = route_planner.plan_routes()
        self.route = candidate_holder.retrieve_first_route()
        self.path = self.route.reference_paths[self.route.principle_reference.index(1)]
        self.center_lanelet_occupancies = self.route.reference_lanelets[self.route.principle_reference.index(1)]
        self.stopline_occupancies = self.route.reference_path_segments[self.route.principle_reference.index(1)][:, -1]
        initial_proj, _ = project_on_path(initial_state.posG, self.path, 0, 50)
        self.ind = initial_proj.ind
        self.desired_d = self.current_d = initial_proj.d
        self.current_sv = initial_state.velocity
        self.current_sa = 0.0
        self.current_dv = 0.0
        self.current_da = 0.0
        self.current_j = 0.0
        self.ttc = float('inf')
        self.planned_infos = []
        self.action = None
        self.front_vehid = None
        self.merge_front_vehid = None
        self.back_vehid = None

        #alternative_candidate_holder = route_planner.plan_alternative_routes()
        #self.list_alt_routes, _ = alternative_candidate_holder.retrieve_all_routes()
        self.list_ids_neighbor = []

    def apply_action(self, action):
        #self.mode, self.des_s, self.des_v, self.des_a, self.des_T, self.des_d = action
        self.action = action
        self._current_plan_step = 0

    def follow(self, obstacles, obstacles_preds):
        target_T = [self.action[1]]
        target_s = [self.action[2]]
        target_v = [self.action[3]]
        target_a = [self.action[4]]
        #target_d = [self.action[5]]
        #target_d = [0.0]
        target_d = [self.desired_d]

        offset_s = []
        offset_s_minus = (self.action[2] - self.action[6]) / 10
        offset_s_plus = (self.action[7] - self.action[2]) / 10
        for i in range(11):
            offset_s.append(-offset_s_minus*i)
        for i in range(1, 11):
            offset_s.append(offset_s_plus*i)

        state = [self.current_state.position[0], self.current_state.position[1],
                 self.current_state.orientation, self.current_sv,
                 self.current_sa, self.current_dv, self.current_da, self.current_time_step,
                 self._length, self._width]
        #if self.current_sv <= 2:# and self.action[1] < 2.6:
        #    offset_d = np.array([0.0, -0.1, 0.1])
        #else:
        #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])
        if self.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])

        action = dict(target_T=np.array(target_T),
              target_s=np.array(target_s),
              target_v=np.array(target_v),
              target_a=np.array(target_a),
              offset_s=np.array(offset_s),
              target_d=np.array(target_d),
              offset_d=offset_d,
              keep_velocity=0)
        #print("follow", action)

        if obstacles_preds is not None:
            return opt(state, action, obstacles_preds.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                       future_steps=list(range(0, self.prediction_steps, 10)))
        else:
            return opt(state, action, obstacles.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                       future_steps=list(range(0, self.prediction_steps, 10)))

    def stop(self, obstacles, obstacles_preds):
        target_T, target_s, target_v, target_a, target_d = [], [], [], [], []
        dT_minus = (self.action[1] - self.action[6])/10
        dT_plus = (self.action[7] - self.action[1])/10
        for i in range(11):
            target_T.append(self.action[1] - dT_minus*i)
            target_s.append(self.action[2])
            target_v.append(self.action[3])
            target_a.append(self.action[4])
            #target_d.append(self.action[5])
            #target_d.append(0.0)
            target_d.append(self.desired_d)
        for i in range(1, 11):
            target_T.append(self.action[1] + dT_plus*i)
            target_s.append(self.action[2])
            target_v.append(self.action[3])
            target_a.append(self.action[4])
            #target_d.append(self.action[5])
            #target_d.append(0.0)
            target_d.append(self.desired_d)

        state = [self.current_state.position[0], self.current_state.position[1],
                 self.current_state.orientation, self.current_sv,
                 self.current_sa, self.current_dv, self.current_da, self.current_time_step,
                 self._length, self._width]
        #if self.current_sv <= 2:# and np.min(target_T) < 2.6:
        #    offset_d = np.array([0.0])
        #else:
        #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])
        if self.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])

        action = dict(target_T=np.array(target_T),
              target_s=np.array(target_s),
              target_v=np.array(target_v),
              target_a=np.array(target_a),
              offset_s=np.array([0.0]),
              target_d=np.array(target_d),
              offset_d=offset_d,
              keep_velocity=1)
        if obstacles_preds is not None:
            return opt(state, action, obstacles_preds.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                        future_steps=list(range(0, self.prediction_steps, 10)))
        else:
            return opt(state, action, obstacles.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                        future_steps=list(range(0, self.prediction_steps, 10)))

    def keep_speed(self, obstacles, obstacles_preds):
        target_T = [self.action[1]]
        target_s = [self.action[2]]
        target_v = [self.action[3]]
        target_a = [self.action[4]]
        #target_d = [self.action[5]]
        #target_d = [0.0]
        target_d = [self.desired_d]

        offset_s = []
        offset_s_minus = (self.action[3] - self.action[6]) / 10
        offset_s_plus = (self.action[7] - self.action[3]) / 10
        for i in range(11):
            offset_s.append(self.action[3] - offset_s_minus*i)
        for i in range(1, 11):
            offset_s.append(self.action[3] + offset_s_plus*i)
        state = [self.current_state.position[0], self.current_state.position[1],
                 self.current_state.orientation, self.current_sv,
                 self.current_sa, self.current_dv, self.current_da, self.current_time_step,
                 self._length, self._width]
        #if self.current_sv <= 2:# and self.action[1] < 2.6:
        #    offset_d = np.array([0.0, -0.1, 0.1])
        #else:
        #    offset_d = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5])
        if self.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])
        action = dict(target_T=np.array(target_T),
                      target_s=np.array(target_s),
                      target_v=np.array(target_v),
                      target_a=np.array(target_a),
                      offset_s=np.array(offset_s),
                      target_d=np.array(target_d),
                      offset_d=offset_d,
                      keep_velocity=1)
        if obstacles_preds is not None:
            return opt(state, action, obstacles_preds.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                    future_steps=list(range(0, self.prediction_steps, 10)))
        else:
            return opt(state, action, obstacles.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                    future_steps=list(range(0, self.prediction_steps, 10)))

    def merge(self, obstacles, obstacles_preds):
        target_T = [self.action[1]]
        target_s = [self.action[2]]
        target_v = [self.action[3]]
        target_a = [self.action[4]]
        #target_d = [self.action[5]]
        #target_d = [0.0]
        target_d = [self.desired_d ]
        offset_s = []
        offset_s_minus = (self.action[2] - self.action[6]) / 10
        offset_s_plus = (self.action[7] - self.action[2]) / 10
        for i in range(11):
            offset_s.append(-offset_s_minus*i)
        for i in range(1, 11):
            offset_s.append(offset_s_plus*i)

        state = [self.current_state.position[0], self.current_state.position[1],
                 self.current_state.orientation, self.current_sv,
                 self.current_sa, self.current_dv, self.current_da, self.current_time_step,
                 self._length, self._width]
        if self.current_sv <= 2:# and self.action[1] < 2.6:
            offset_d = np.array([0.0])
        else:
            offset_d = np.array([0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5])
        action = dict(target_T=np.array(target_T),
                      target_s=np.array(target_s),
                      target_v=np.array(target_v),
                      target_a=np.array(target_a),
                      offset_s=np.array(offset_s),
                      target_d=np.array(target_d),
                      offset_d=offset_d,
                      keep_velocity=0)
        if obstacles_preds is not None:
            return opt(state, action, obstacles_preds.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                       future_steps=list(range(0, self.prediction_steps, 10)))
        else:
            return opt(state, action, obstacles.values(), self.route.reference_path_segments[self.route.principle_reference.index(1)],
                   future_steps=list(range(0, self.prediction_steps, 10)))

    def update_front_veh(self, obstacles):
        self.current_s = s0 = self.path[self.ind.i].s + (self.path[self.ind.i+1].s - self.path[self.ind.i].s)*self.ind.t
        front_s = float('inf')
        front_v = 0
        front_a = 0
        front_veh = None
        for i in range(self.ind.i, len(self.path)):
            s = self.path[i].s
            if s - s0 > self.front_s_max:
                break
            # future occupancy
            future_occupancy_lanelets = [self.center_lanelet_occupancies[i]]
            if self.current_d + 0.5*self._width >= 3.0:
                lanelet = self.lanelet_network.find_lanelet_by_id(future_occupancy_lanelets[0])
                if lanelet.adj_left is not None:
                    future_occupancy_lanelets.append(lanelet.adj_left)
            '''
            if self.current_d - 0.5*self._width <= -3.0:
                lanelet = self.lanelet_network.find_lanelet_by_id(future_occupancy_lanelets[0])
                if lanelet.adj_right is not None:
                    future_occupancy_lanelets.append(lanelet.adj_right)
            '''
            future_occupancy_lanelets = set(future_occupancy_lanelets)
            for obstacle in obstacles:
                if len(future_occupancy_lanelets.intersection(obstacle.initial_shape_lanelet_ids)) > 0:
                    obs_head_pos = obstacle.obstacle_shape.length*0.5*np.array([np.cos(obstacle.initial_state.orientation),
                                                                       np.sin(obstacle.initial_state.orientation)]) + obstacle.initial_state.position
                    obs_head_pos = VecSE2(obs_head_pos[0], obs_head_pos[1], obstacle.initial_state.orientation)
                    obs_head_proj, obs_head_s = project_on_path(obs_head_pos, self.path, self.ind.i, i)
                    # head-to-head distance to determine front yes/no
                    if obs_head_s < s0 + 0.5*self._length:
                        continue #no
                    obs_tail_pos = -obstacle.obstacle_shape.length*0.5*np.array([np.cos(obstacle.initial_state.orientation),
                                                                        np.sin(obstacle.initial_state.orientation)]) + obstacle.initial_state.position
                    obs_tail_pos = VecSE2(obs_tail_pos[0], obs_tail_pos[1], obstacle.initial_state.orientation)
                    obs_tail_proj, obs_tail_s = project_on_path(obs_tail_pos, self.path, self.ind.i, i)
                    # lateral distance to determine on path yes/no
                    if min(np.abs(obs_head_proj.d), np.abs(obs_tail_proj.d)) > 0.5*self._width + 0.5*obstacle.obstacle_shape.width + 0.8: #TODO
                        continue #no
                    # bumper-to-bumper distance to determine closeness
                    delta_s = max(0.0, obs_tail_s - (s0+0.5*self._length))
                    if delta_s < front_s:
                        front_s = delta_s
                        front_v =  np.cos(obs_head_proj.phi) * obstacle.initial_state.velocity
                        front_a = obstacle.initial_state.acceleration
                        front_veh = obstacle.obstacle_id
            if front_veh is not None:
                break
        self.front_s = front_s
        self.front_veh = front_veh
        self.front_v = front_v
        self.front_a = front_a

        if self.front_s == float('inf'):
            self.ttc = float('inf')
        elif self.front_s == 0.0:
            self.ttc = 0.0
        else:
            delta_v = self.current_sv - self.front_v
            if delta_v <= 0:
                self.ttc = float('inf')
            else:
                self.ttc = self.front_s/delta_v

    def update_connected_state(self,  obstacles, obstacles_preds):
        if self._current_plan_step % self.plan_steps == 0:
            self.update_front_veh(obstacles.values())
            best_action = None
            best_dist = 0.0
            for act, dist in zip(self.action[0], self.action[1]):
                if dist > best_dist and act is not None:
                    best_action = act
                    best_dist = dist

            if best_action is None:
                #print("None", self.current_time_step, self.plan_steps)
                planned_trajectory = self._planned_trajectories[self.current_time_step-self.plan_steps][self.plan_steps:]
                self.set_planned_trajectory(planned_trajectory)
                self.set_all_planned_trajectories([])
                self._current_plan_step = 1
                self._current_time_step += 1
                return self.current_state

            planned_trajectory = []
            self.planned_infos = []

            self.s_d = []

            for i in range(1, best_action.path_length):
                next_state =FrenetState(position = np.array([best_action.x[i], best_action.y[i]]),
                                   orientation=best_action.heading[i],
                                   velocity=np.sqrt(best_action.speed_s[i]**2+best_action.speed_d[i]**2),
                                   acceleration=best_action.acc_s[i],
                                   acceleration_y=best_action.acc_d[i],
                                   time_step=self.current_time_step+i,
                                   yaw_rate=0.0,
                                   slip_angle=best_action.slipangle[i])
                #if self.action[0] == 0:
                #    print("FO", i,  frenet_return.best_trajectory[0].s[i], frenet_return.best_trajectory[0].speed_s[i], frenet_return.best_trajectory[0].acc_s[i])
                self.s_d.append([best_action.s[i], best_action.d[i]])
                phi = np.arctan2(best_action.d[i]-best_action.d[i-1],
                                 best_action.s[i]-best_action.s[i-1])
                if i == 1:
                    ind, _ = get_curve_index(self.ind, self.path, best_action.s[i]-best_action.s[i-1])
                    self.ind = ind # TODO path ind not lanelet ind
                    self.current_d = best_action.d[i]
                    self.current_sv = best_action.speed_s[i]
                    self.current_sa = best_action.acc_s[i]
                    self.current_dv = best_action.speed_d[i]
                    self.current_da = best_action.acc_d[i]
                    self.current_j = best_action.jerk[i]
                else:
                    ind, _ = get_curve_index(ind, self.path, best_action.s[i]-best_action.s[i-1])
                    self.planned_infos.append([ind,
                                               best_action.d[i],
                                               best_action.speed_s[i],
                                               best_action.acc_s[i],
                                               best_action.speed_d[i],
                                               best_action.acc_d[i],
                                               best_action.jerk[i]])
                lanelet_list = [self.center_lanelet_occupancies[ind.i]]
                lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_list[0])
                if lanelet.adj_left:
                    lanelet_list.append(lanelet.adj_left)
                if lanelet.adj_right:
                    lanelet_list.append(lanelet.adj_right)
                next_state.set_posF(self.lanelet_network, lanelet_list)
                #next_state.posF = Frenet(tuple_form=(ind.i,ind.t,self.center_lanelet_occupancies[ind.i],frenet_return.best_trajectory[0].s[i],frenet_return.best_trajectory[0].d[i],phi))
                planned_trajectory.append(next_state)
            self.set_planned_trajectory(planned_trajectory)

            alter_trajectories = []
            alter_dists = []
            for act, dist in zip(self.action[0], self.action[1]):
                if act is None:
                    continue
                traj = []
                for i in range(1, act.path_length):
                    next_state =FrenetState(position = np.array([act.x[i], act.y[i]]),
                                       orientation=act.heading[i],
                                       velocity=np.sqrt(act.speed_s[i]**2+act.speed_d[i]**2),
                                       acceleration=act.acc_s[i],
                                       acceleration_y=act.acc_d[i],
                                       time_step=self.current_time_step+i,
                                       yaw_rate=0.0,
                                       slip_angle=act.slipangle[i])
                    traj.append(next_state)
                alter_trajectories.append(traj)
                alter_dists.append(dist)
            self.set_all_planned_trajectories(alter_trajectories)
            self.set_all_planned_dists(alter_dists)
            self._current_plan_step = 1

        else:
            self.add_state(self._planned_trajectories[self.current_time_step-self._current_plan_step][self._current_plan_step % self.plan_steps])
            info = self.planned_infos.pop(0)
            self.ind = info[0]
            self.current_d = info[1]
            self.current_sv = info[2]
            self.current_sa = info[3]
            self.current_dv = info[4]
            self.current_da = info[5]
            self.current_j = info[6]
            self._current_plan_step += 1
        self._current_time_step += 1

        #self.arrived = self.planning_problem.goal.is_reached(self.current_state)
        #if self.arrived:
        #    return None
        return self.current_state


    def update_state(self, obstacles, obstacles_preds):
        if self._current_plan_step % self.plan_steps == 0: # replan every few steps
            self.update_front_veh(obstacles.values())
            if self.action is None:
                #print("None", self.current_time_step, self.plan_steps)
                planned_trajectory = self._planned_trajectories[self.current_time_step-self.plan_steps][self.plan_steps:]
                self.set_planned_trajectory(planned_trajectory)
                self.set_all_planned_trajectories([])
                self._current_plan_step = 1
                self._current_time_step += 1
                return self.current_state

            if self.action[0] == 0: # follow
                frenet_return = self.follow(obstacles, obstacles_preds)
            elif self.action[0] == 1: # velocity
                frenet_return = self.keep_speed(obstacles, obstacles_preds)
            elif self.action[0] == 2: # stop
                frenet_return = self.stop(obstacles, obstacles_preds)
            elif self.action[0] == 3: # merge
                frenet_return = self.merge(obstacles, obstacles_preds)

            assert frenet_return.success >=0, "collision"

            if frenet_return.success == 0:
                #print("None", self.current_time_step, self.plan_steps)
                planned_trajectory = self._planned_trajectories[self.current_time_step-self.plan_steps][self.plan_steps:]
                self.set_planned_trajectory(planned_trajectory)
                self.set_all_planned_trajectories([])
                self._current_plan_step = 1
                self._current_time_step += 1
                return self.current_state

            planned_trajectory = []
            self.planned_infos = []

            self.s_d = []

            for i in range(1, frenet_return.best_trajectory[0].path_length):
                #if frenet_return.best_trajectory[0].speed_s[i]

                next_state =FrenetState(position = np.array([frenet_return.best_trajectory[0].x[i], frenet_return.best_trajectory[0].y[i]]),
                                   orientation=frenet_return.best_trajectory[0].heading[i],
                                   velocity=np.sqrt(frenet_return.best_trajectory[0].speed_s[i]**2+frenet_return.best_trajectory[0].speed_d[i]**2),
                                   acceleration=frenet_return.best_trajectory[0].acc_s[i],
                                   acceleration_y=frenet_return.best_trajectory[0].acc_d[i],
                                   time_step=self.current_time_step+i,
                                   yaw_rate=0.0,
                                   slip_angle=frenet_return.best_trajectory[0].slipangle[i])
                #if self.action[0] == 0:
                #    print("FO", i,  frenet_return.best_trajectory[0].s[i], frenet_return.best_trajectory[0].speed_s[i], frenet_return.best_trajectory[0].acc_s[i])
                self.s_d.append([frenet_return.best_trajectory[0].s[i], frenet_return.best_trajectory[0].d[i]])
                phi = np.arctan2(frenet_return.best_trajectory[0].d[i]-frenet_return.best_trajectory[0].d[i-1],
                                 frenet_return.best_trajectory[0].s[i]-frenet_return.best_trajectory[0].s[i-1])
                if i == 1:
                    ind, _ = get_curve_index(self.ind, self.path, frenet_return.best_trajectory[0].s[i]-frenet_return.best_trajectory[0].s[i-1])
                    self.ind = ind # TODO path ind not lanelet ind
                    self.current_d = frenet_return.best_trajectory[0].d[i]
                    self.current_sv = frenet_return.best_trajectory[0].speed_s[i]
                    self.current_sa = frenet_return.best_trajectory[0].acc_s[i]
                    self.current_dv = frenet_return.best_trajectory[0].speed_d[i]
                    self.current_da = frenet_return.best_trajectory[0].acc_d[i]
                    self.current_j = frenet_return.best_trajectory[0].jerk[i]
                else:
                    ind, _ = get_curve_index(ind, self.path, frenet_return.best_trajectory[0].s[i]-frenet_return.best_trajectory[0].s[i-1])
                    self.planned_infos.append([ind,
                                               frenet_return.best_trajectory[0].d[i],
                                               frenet_return.best_trajectory[0].speed_s[i],
                                               frenet_return.best_trajectory[0].acc_s[i],
                                               frenet_return.best_trajectory[0].speed_d[i],
                                               frenet_return.best_trajectory[0].acc_d[i],
                                               frenet_return.best_trajectory[0].jerk[i]])
                lanelet_list = [self.center_lanelet_occupancies[ind.i]]
                lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_list[0])
                if lanelet.adj_left:
                    lanelet_list.append(lanelet.adj_left)
                if lanelet.adj_right:
                    lanelet_list.append(lanelet.adj_right)
                next_state.set_posF(self.lanelet_network, lanelet_list)
                #next_state.posF = Frenet(tuple_form=(ind.i,ind.t,self.center_lanelet_occupancies[ind.i],frenet_return.best_trajectory[0].s[i],frenet_return.best_trajectory[0].d[i],phi))
                planned_trajectory.append(next_state)
            self.set_planned_trajectory(planned_trajectory)
            alter_trajectories = []
            for k in range(frenet_return.num_traj):
                if frenet_return.trajectories[k].feasible == 0:
                    continue
                traj = []

                #print(frenet_return.trajectories[k].final_d)

                for i in range(1, frenet_return.trajectories[k].path_length):
                    #print("t ", frenet_return.trajectories[k].t[i], frenet_return.trajectories[k].s[i], frenet_return.trajectories[k].d[i])

                    next_state =FrenetState(position = np.array([frenet_return.trajectories[k].x[i], frenet_return.trajectories[k].y[i]]),
                                       orientation=frenet_return.trajectories[k].heading[i],
                                       velocity=np.sqrt(frenet_return.trajectories[k].speed_s[i]**2+frenet_return.trajectories[k].speed_d[i]**2),
                                       acceleration=frenet_return.trajectories[k].acc_s[i],
                                       acceleration_y=frenet_return.trajectories[k].acc_d[i],
                                       time_step=self.current_time_step+i,
                                       yaw_rate=0.0,
                                       slip_angle=frenet_return.trajectories[k].slipangle[i])
                    traj.append(next_state)
                alter_trajectories.append(traj)
            #print(alter_trajectories, frenet_return.num_traj)
            self.set_all_planned_trajectories(alter_trajectories)
            self._current_plan_step = 1

        else:
            self.add_state(self._planned_trajectories[self.current_time_step-self._current_plan_step][self._current_plan_step % self.plan_steps])
            info = self.planned_infos.pop(0)
            self.ind = info[0]
            self.current_d = info[1]
            self.current_sv = info[2]
            self.current_sa = info[3]
            self.current_dv = info[4]
            self.current_da = info[5]
            self.current_j = info[6]
            self._current_plan_step += 1
        self._current_time_step += 1

        #self.arrived = self.planning_problem.goal.is_reached(self.current_state)
        #if self.arrived:
        #    return None
        return self.current_state

    @property
    def pp_id(self) -> int:
        """
        :returns: associated planning problem id
        """
        return self.planning_problem.planning_problem_id if self.planning_problem is not None else None

    def set_neighbors(self, list_ids_neighbor: List[str]) -> None:
        self.list_ids_neighbor = list_ids_neighbor

    def set_all_planned_trajectories(self, all_state_list: List[List[State]]) -> None:
        self._all_planned_trajectories[self.current_time_step] = all_state_list

    def set_all_planned_dists(self, all_dists_list):
        self._all_planned_dists[self.current_time_step] = all_dists_list

    def set_planned_trajectory(self, planned_state_list: List[State]) -> None:
        """
        Sets planned trajectory beginning with current time step.

        :param planned_state_list: the planned trajectory

        """

        assert len(planned_state_list) >= self.delta_steps, \
            'planned_trajectory must contain at least {} states, but contains {}. (See delta_steps in sumo_config file)' \
                .format(self.delta_steps, len(planned_state_list))
        assert self._current_time_step+1 == planned_state_list[0].time_step, \
            'planned_trajectory must always start at time_step ({}) but starts at time_step {}' \
                .format(self._current_time_step + 1, planned_state_list[0].time_step)
        self._planned_trajectories[self.current_time_step] = planned_state_list
        self.add_state(planned_state_list[0])

    @property
    def get_planned_trajectory(self) -> List[State]:
        """Gets planned trajectory according to the current time step"""
        return self._planned_trajectories[self.current_time_step]

    def get_dynamic_obstacle(self, time_step: Union[int, None] = None) -> DynamicObstacle:
        """
        If time step is None, adds complete driven trajectory and returns the dynamic obstacles.
        If time step is int: starts from given step and adds planned trajectory and returns the dynamic obstacles.

        :param time_step: initial time step of vehicle
        :return: DynamicObstacle object of the ego vehicle.
        """
        if time_step is None:
            return DynamicObstacle(self.id, obstacle_type=ObstacleType.CAR,
                                   obstacle_shape=Rectangle(self.length, self.width, center=np.array([0, 0]),
                                                            orientation=0.0),
                                   initial_state=self.initial_state, prediction=self.driven_trajectory)
        elif isinstance(time_step, int):
            if time_step in self._state_dict:
                if time_step in self._planned_trajectories:
                    prediction = TrajectoryPrediction(Trajectory(self._planned_trajectories[time_step][0].time_step,
                                                                 self._planned_trajectories[time_step]),
                                                      self.shape)
                elif time_step-1 in self._planned_trajectories:
                    prediction = TrajectoryPrediction(Trajectory(self._planned_trajectories[time_step-1][0].time_step,
                                                                 self._planned_trajectories[time_step-1]),
                                                      self.shape)
                else:
                    prediction = None
                #print("time_step in state", time_step, self._planned_trajectories.keys(), prediction is None)
                return DynamicObstacle(self.id, obstacle_type=ObstacleType.CAR,
                                       obstacle_shape=Rectangle(self.length, self.width, center=np.array([0, 0]),
                                                                orientation=0.0),
                                       initial_state=self.get_state_at_timestep(time_step), prediction=prediction)
            elif time_step == self.initial_state.time_step:
                if time_step in self._planned_trajectories:
                    prediction = TrajectoryPrediction(Trajectory(self._planned_trajectories[time_step][0].time_step,
                                                                 self._planned_trajectories[time_step]),
                                                      self.shape)
                elif time_step-1 in self._planned_trajectories:
                    prediction = TrajectoryPrediction(Trajectory(self._planned_trajectories[time_step-1][0].time_step,
                                                                 self._planned_trajectories[time_step-1]),
                                                      self.shape)
                else:
                    prediction = None
                #print("time_step not in state", time_step, self._planned_trajectories.keys(), self.initial_state.time_step, prediction is None)
                return DynamicObstacle(self.id, obstacle_type=ObstacleType.CAR,
                                       obstacle_shape=Rectangle(self.length, self.width, center=np.array([0, 0]),
                                                                orientation=0.0),
                                       initial_state=self.get_state_at_timestep(time_step), prediction=prediction)
        else:
            raise ValueError('time needs to be type None or int')

    def get_planned_state(self, delta_step: int = 0):
        """
        Returns the planned state.

        :param delta_step: get planned state after delta steps

        """
        assert self.current_time_step in self._planned_trajectories,\
            f"No planned trajectory found at time step {self.current_time_step} for ego vehicle {self.id}! " \
            f"Use ego_vehicle.set_planned_trajectory() to set the trajectory. {self._planned_trajectories}"

        planned_state: State = copy.deepcopy(self._planned_trajectories[self.current_time_step][0])
        if self.delta_steps > 1:
            # linear interpolation
            for state in planned_state.attributes:
                curr_state = getattr(self.current_state, state)
                next_state = getattr(planned_state, state)
                setattr(planned_state, state,
                        curr_state + (delta_step + 1) / self.delta_steps * (next_state - curr_state))

        return planned_state

    @property
    def current_state(self) -> State:
        """
        Returns the current state.
        """
        if self.current_time_step == self.initial_state.time_step:
            return self.initial_state
        else:
            return self._state_dict[self.current_time_step]

    def get_state_at_timestep(self, time_step: int) -> State:
        """
        Returns the state according to the given time step.

        :param time_step: the state is returned according to this time step.
        """

        if time_step == self.initial_state.time_step:
            state = copy.deepcopy(self.initial_state)
            #state.time_step = 0
            return state
        else:
            state = self._state_dict[time_step]
            #state.time_step = 0
            return state

    @current_state.setter
    def current_state(self, current_state):
        raise PermissionError('current_state cannot be set manually, use set_planned_trajectory()')

    @property
    def current_time_step(self) -> int:
        """
        Returns current time step.
        """
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, current_time_step):
        raise PermissionError('current_state cannot be set manually, use set_planned_trajectory()')

    @property
    def goal(self) -> GoalRegion:
        """
        Returns the goal of the planning problem.
        """
        return self.planning_problem.goal

    def add_state(self, state: State) -> None:
        """
        Adds a state to the current state dictionary.

        :param state: the state to be added

        """
        self._state_dict[self._current_time_step + 1] = state

    @property
    def driven_trajectory(self) -> TrajectoryPrediction:
        """
        Returns trajectory prediction object for driven trajectory (mainly for plotting)

        """
        state_dict_tmp = {}
        for t, state in self._state_dict.items():
            state_dict_tmp[t] = state
            state_dict_tmp[t].time_step = t

        sorted_list = sorted(state_dict_tmp.keys())
        state_list = [state_dict_tmp[key] for key in sorted_list]
        return TrajectoryPrediction(Trajectory(self.initial_state.time_step + 1, state_list), self.shape)

    @driven_trajectory.setter
    def driven_trajectory(self, _):
        if hasattr(self, '_driven_trajectory'):
            warnings.warn('driven_trajectory of vehicle cannot be changed')
            return

    @property
    def width(self) -> float:
        """
        Returns the width of the ego vehicle.
        """
        return self._width

    @width.setter
    def width(self, width):
        if hasattr(self, '_width'):
            warnings.warn('width of vehicle cannot be changed')
            return

    @property
    def length(self) -> float:
        """
        Returns the length of the ego vehicle.
        """
        return self._length

    @length.setter
    def length(self, length):
        if hasattr(self, '_length'):
            warnings.warn('length of vehicle cannot be changed')
            return

    @property
    def initial_state(self) -> State:
        """
        Returns the initial state of the ego vehicle.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, _):
        if hasattr(self, '_initial_state'):
            warnings.warn('initial_state of vehicle cannot be changed')
            return
