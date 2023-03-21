# for RL and IRL and MP
import os
import os.path as osp
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle, Arc, Polygon
from copy import copy
import torch
import gc
from torch_geometric.data import Dataset, Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.utils import degree, add_self_loops
from commonroad.scenario.scenariocomplement import CoreFeatureExtractor2, MultiFeatureExtractor, ScenarioWrapper
from commonroad.scenario.scenariocomplement import MultiFeatureExtractor, CoreFeatureExtractor0, NeighborFeatureExtractor0, NeighborFeatureExtractor, ScenarioWrapper

from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle
from commonroad.common.file_reader_complement import LaneletCurveNetworkReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LineMarking, LaneletType
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet, move_along_curve

from commonroad.scenario.laneletcomplement import *
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.laneletcomplement import make_lanelet_curve_network
from commonroad.common.file_reader import CommonRoadFileReader

from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem, GoalRegion
from commonroad.scenario.trajectory import Trajectory, State

from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.scenario import Scenario

from ego_vehicle import EgoVehicle, GIDMVehicle
from route import RoutePlanner
from util import make_paths, load_ngsim_scenarios,  make_grids, ScenarioData, traj_collate
import networkx as nx
from config import draw_params_ego,  draw_params_mlego, draw_params_obstacle
from frenet_planning_wrapper import opt
from gym import spaces
from commonroad.prediction.prediction import TrajectoryPrediction, Occupancy

import pickle

class ReplayRoundabout(object):
    # Replay env with replay from dataset
    def __init__(self, config, seed=36, load_path=True):
        self.config = config
        self.vehicle_exts = CoreFeatureExtractor2(self.config)
        if load_path:
            if isinstance(config["data_dir"], str):
                self.root = root = config["data_dir"]
                self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
                self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
            else:
                self.root = root = config["data_dir"][0]
                self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
                self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
            self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
            self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
            self.trajectories, self.vehicleinfos, self.lanelet_networks = load_ngsim_scenarios(self.lanelet_paths, self.raw_paths)

        self.rng = np.random.RandomState(seed)
        self.observation_space = spaces.Dict({"veh_state":spaces.Box(low=0, high=1, shape=(4,)),
                                              "veh_shapes":spaces.Box(low=0, high=1, shape=(2,)),
                                              "veh_hist":spaces.Box(low=0, high=1, shape=(3,3)),
                                              "lane_ctrs":spaces.Box(low=0, high=1, shape=(2,)),
                                              "lane_pris":spaces.Box(low=0, high=1, shape=(2,)),
                                              "lane_vecs":spaces.Box(low=0, high=1, shape=(2,)),
                                              "lane_widths":spaces.Box(low=0, high=1, shape=(2,))}
                                            )
        self.action_space = spaces.Box(low=0, high=1, shape=(2,))
        # the lanelet grids around the IRL vehicle
        # the path is given and known (some times a lane change is mandatory)
        # we need a new representation of the path with adaptive nodes

    @property
    def raw_paths(self):
        files = self.raw_files
        return [osp.join(self.root+"/raw", f) for f in files]

    @property
    def lanelet_paths(self):
        files = self.map_files
        return [osp.join(self.root+"/raw", f) for f in files]

    def reset(self, info):
        raw_path, egoid, ts, te = info
        self.traj_idx = traj_idx = self.raw_paths.index(raw_path)
        self.lanelet_network = self.lanelet_networks[self.vehicleinfos[traj_idx][0]]
        vehids = list(self.trajectories[traj_idx][ts].keys())
        self.te = te
        self.background_vehicles = dict()
        self.ego_vehicles = dict()
        self.obstacles = defaultdict(lambda:dict())

        self.source_lanelet_ids, self.goal_lanelet_ids = [], []
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.successor) == 0:
                self.goal_lanelet_ids.append(lanelet.lanelet_id)
            if len(lanelet.predecessor) == 0:
                self.source_lanelet_ids.append(lanelet.lanelet_id)

        for t in range(ts, te+1):
            for vehid in self.trajectories[traj_idx][t].keys():
                ts_, te_ = self.vehicleinfos[traj_idx][1][vehid]["frames"]
                vels = [self.trajectories[traj_idx][tt][vehid].velocity for tt in range(ts_, te_+1)]
                vels = np.array(vels)
                acc = (vels[1:] - vels[:-1])/self.config["dt"]
                acc = np.append(acc, acc[-1])

                stopped = False
                if vels[0] == 0.0:
                    stop_steps = 0
                    hist_stop_steps = 0
                    tt = ts_+1
                    while tt <= te_:
                        if vels[tt-ts_]==0.0:
                            stop_steps += 1
                        else:
                            break
                        tt += 1
                    if stop_steps >= 2/self.config["dt"]:
                        stopped = True
                    if not stopped:
                        tt = ts_-1
                        while tt >= ts_:
                            if vels[tt-ts_]==0.0:
                                hist_stop_steps += 1
                                if stop_steps + hist_stop_steps >= 2/self.config["dt"]:
                                    stopped = True
                                    break
                            else:
                                break
                            tt -= 1
                if stopped:
                    continue
                shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                              self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                              center=self.trajectories[traj_idx][t][vehid].position,
                                              orientation=self.trajectories[traj_idx][t][vehid].orientation))
                initial_state = self.trajectories[traj_idx][t][vehid]
                initial_state.acceleration = acc[t-ts_]
                assert(initial_state.time_step == t)
                self.obstacles[t][vehid] = DynamicObstacle(obstacle_id=vehid,
                    obstacle_type=self.vehicleinfos[traj_idx][1][vehid]["type"],
                    initial_state=initial_state,
                    obstacle_shape=self.vehicleinfos[traj_idx][1][vehid]["shape"],
                    initial_center_lanelet_ids=set([self.trajectories[traj_idx][t][vehid].posF.ind[1]]),
                    initial_shape_lanelet_ids=set(shape_lanelet_ids))

        self.expert_cr_scenario = ScenarioWrapper(self.config["dt"], self.lanelet_network,
                                              self.vehicleinfos[traj_idx][0], self.obstacles, None)
        self.expert_cr_scenario.set_sensor_range(self.config["max_veh_radius"],
                                            self.config["max_veh_disp_front"],
                                            self.config["max_veh_disp_rear"],
                                            self.config["max_veh_disp_side"])
        self.expert_cr_scenario = ScenarioWrapper(self.config["dt"], self.lanelet_network,
                                              self.vehicleinfos[traj_idx][0], self.obstacles, None)
        self.expert_cr_scenario.scenario_id = "{}_{}".format(traj_idx, ts)
        self.expert_cr_scenario.set_sensor_range(self.config["max_veh_radius"],
                                            self.config["max_veh_disp_front"],
                                            self.config["max_veh_disp_rear"],
                                            self.config["max_veh_disp_side"])
        state = dict()
        return state

    def generate_MP_state(self, egoid, time_step, include_gt=False):
        record = True
        time_steps = list(range(time_step-self.config["horizon_steps"]+1, time_step+1, self.config["delta_step"]))
        pred_steps =  self.config["prediction_steps"]//self.config["delta_step"]

        all_grids,\
        ctrs, \
        vecs, \
        pris, \
        widths, \
        suc_edges, \
        pre_edges, left_edges, right_edges,\
        node2lane, lane2node  = make_grids(self.expert_cr_scenario,
                                           egoid,
                                           time_step,
                                           self.config["grid_length"],
                                           max_disp_front=self.config["max_grid_disp_front"],
                                           max_disp_rear=self.config["max_grid_disp_rear"],
                                           max_radius=self.config["max_grid_radius"],
                                           cross_dist=self.config["cross_dist"],
                                           num_scales=self.config["num_scales"])

        self.expert_cr_scenario.add_grids(all_grids)
        neighbors = self.expert_cr_scenario.get_neighbors(egoid,
                                                     time_step,
                                                     along_lanelet=self.config["veh_along_lane"],
                                                     max_radius=self.config["max_veh_radius"],
                                                     front=self.config["max_veh_disp_front"],
                                                     side=self.config["max_veh_disp_side"],
                                                     rear=self.config["max_veh_disp_rear"])
        self.vehicle_exts.set_origin(self.expert_cr_scenario, egoid, time_step)
        x_dict,y_dict,lx_dict,ly_dict,w_dict,edge_dict = {},{},{},{},{},{}
        ego_feat = self.vehicle_exts.get_features(self.expert_cr_scenario, egoid, time_step)
        ego_x = self.vehicle_exts.get_features_by_name(ego_feat, ['obs_xs', 'obs_ys',
                                                                  'obs_rel_xs','obs_rel_ys', 'masked',
                                                                  'length', 'width'])
        ego_lx = self.vehicle_exts.get_features_by_name(ego_feat, ['obs_lanelet'])
        ego_y = self.vehicle_exts.get_features_by_name(ego_feat, ['fut_xs', 'fut_ys', 'fut_rel_xs', 'fut_rel_ys', 'has_preds'])
        ego_ly = self.vehicle_exts.get_features_by_name(ego_feat, ['fut_lanelet'])
        #print(ego_x[4*len(time_steps):5*len(time_steps)], ego_y[-pred_steps:])
        if (np.sum(ego_x[4*len(time_steps):5*len(time_steps)]) == 0.0 or np.sum(ego_y[-pred_steps:]) == 0.0) and include_gt:
            record = False
        #max_vel_nei_x = np.max(self.vehicle_exts.get_features_by_name(ego_feat,['obs_rel_xs']))/(self.config["delta_step"]*self.config["dt"])
        #max_vel_nei_y = np.max(self.vehicle_exts.get_features_by_name(ego_feat,['obs_rel_ys']))/(self.config["delta_step"]*self.config["dt"])

        x_dict[egoid] = ego_x
        y_dict[egoid] = ego_y
        lx_dict[egoid] = ego_lx
        ly_dict[egoid] = ego_ly
        w_dict[egoid] = 1.0

        close_neighbors = []
        for name in self.vehicle_exts.extractors[2].neighbor_names:
            veh_id = int(self.vehicle_exts.get_features_by_name(ego_feat, [name+'_veh_id'])[0])
            if veh_id in neighbors:
                edge_dict[(egoid,veh_id)] = self.vehicle_exts.get_features_by_name(ego_feat, [name+'_rel_x',name+'_rel_y'])
                w_dict[veh_id] = 0.8
                close_neighbors.append(veh_id)
        remaining_neighbors = []
        connected_neighbors = []
        for neighbor_id in close_neighbors:
            neighbors_feat = self.vehicle_exts.get_features(self.expert_cr_scenario, neighbor_id, time_step)
            nei_x = self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_xs','obs_ys',
                                                                           'obs_rel_xs','obs_rel_ys', 'masked',
                                                                           'length', 'width'])
            nei_lx = self.vehicle_exts.get_features_by_name(neighbors_feat, ['obs_lanelet'])

            nei_y = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_xs','fut_ys', 'fut_rel_xs', 'fut_rel_ys', 'has_preds'])
            nei_ly = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_lanelet'])
            #print(nei_x[4*len(time_steps):5*len(time_steps)], nei_y[-pred_steps:])
            if (np.sum(nei_x[4*len(time_steps):5*len(time_steps)]) == 0.0 or np.sum(nei_y[-pred_steps:]) == 0.0) and include_gt:
                record = False

            #max_vel_nei_x = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_xs']))/(self.config["delta_step"]*self.config["dt"])
            #max_vel_nei_y = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_ys']))/(self.config["delta_step"]*self.config["dt"])

            x_dict[neighbor_id] = nei_x
            y_dict[neighbor_id] = nei_y

            lx_dict[neighbor_id] = nei_lx
            ly_dict[neighbor_id] = nei_ly
            for name in self.vehicle_exts.extractors[2].neighbor_names:
                veh_id = int(self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id'])[0])
                if veh_id in neighbors:
                    edge_dict[(neighbor_id, veh_id)] = self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_rel_x',name+'_rel_y'])
                    if veh_id not in close_neighbors and veh_id != egoid and veh_id not in remaining_neighbors:
                        remaining_neighbors.append(veh_id)
                        connected_neighbors.append(veh_id)

        for neighbor_id in neighbors:
            if neighbor_id not in close_neighbors and neighbor_id not in remaining_neighbors:
                remaining_neighbors.append(neighbor_id)
        for neighbor_id in remaining_neighbors:
            neighbors_feat = self.vehicle_exts.get_features(self.expert_cr_scenario, neighbor_id, time_step)
            for name in self.vehicle_exts.extractors[2].neighbor_names:
                veh_id = int(self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id'])[0])
                if veh_id == egoid or veh_id in close_neighbors or veh_id in remaining_neighbors:
                    edge_dict[(neighbor_id, veh_id)] = self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_rel_x',name+'_rel_y'])
                    if veh_id in remaining_neighbors and veh_id not in connected_neighbors:
                        connected_neighbors.append(veh_id)
                    if neighbor_id not in connected_neighbors:
                        connected_neighbors.append(neighbor_id)
        for neighbor_id in connected_neighbors:
            neighbors_feat = self.vehicle_exts.get_features(self.expert_cr_scenario, neighbor_id, time_step)
            nei_x = self.vehicle_exts.get_features_by_name(neighbors_feat, ['obs_xs','obs_ys',
                                                                            'obs_rel_xs','obs_rel_ys','masked',
                                                                            'length', 'width'])
            nei_lx = self.vehicle_exts.get_features_by_name(neighbors_feat, ['obs_lanelet'])

            nei_y = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_xs','fut_ys', 'fut_rel_xs', 'fut_rel_ys', 'has_preds'])
            nei_ly = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_lanelet'])

            #max_vel_nei_x = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_xs']))/(self.config["delta_step"]*self.config["dt"])
            #max_vel_nei_y = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_ys']))/(self.config["delta_step"]*self.config["dt"])
            #print(nei_x[4*len(time_steps):5*len(time_steps)], nei_y[-pred_steps:])
            if (np.sum(nei_x[4*len(time_steps):5*len(time_steps)]) == 0.0 or np.sum(nei_y[-pred_steps:]) == 0.0) and include_gt:
                record = False

            x_dict[neighbor_id] = nei_x
            y_dict[neighbor_id] = nei_y

            lx_dict[neighbor_id] = nei_lx
            ly_dict[neighbor_id] = nei_ly
            w_dict[neighbor_id] = 0.5
        if record == False or len(x_dict)==1:
            return None
        x, xseq, y, yseq, w, id, shape, has_preds = [], [], [], [], [], [], [], []
        lxseq, lyseq = [], []
        id_to_idx = {}
        sender, receiver, edge_attr = [], [], []


        for idx, vehid in enumerate(x_dict.keys()):
            xx = np.array(copy(x_dict[vehid]))
            shape.append(xx[-2:])
            yy = np.array(copy(y_dict[vehid])[:-pred_steps]).reshape((4, pred_steps)).T
            xseq.append(xx[:-2].reshape((5, len(time_steps))).T) # change
            lxseq.append(np.array(lx_dict[vehid]))
            yseq.append(yy)
            lyseq.append(np.array(ly_dict[vehid]))
            has_preds.append(copy(y_dict[vehid])[-pred_steps:])
            w.append(w_dict[vehid])
            id.append(vehid)
            id_to_idx[vehid] = idx
            edge_attr.append([0.0, 0.0])
            sender.append(idx)
            receiver.append(idx)
        for s, r in edge_dict.keys():
            edge_attr.append(edge_dict[(s,r)])
            sender.append(id_to_idx[s])
            receiver.append(id_to_idx[r])
        has_preds = torch.tensor(has_preds, dtype=torch.bool)
        veh_xseq = torch.tensor(xseq, dtype=torch.float)
        veh_yseq = torch.tensor(yseq, dtype=torch.float)

        veh_lxseq = torch.tensor(lxseq, dtype=torch.long)
        veh_lyseq = torch.tensor(lyseq, dtype=torch.long)
        #####################################
        num_preds = has_preds.size(1)
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
                  has_preds.device
               ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)

        lane_ctrs = ctrs
        veh_start_ctrs = veh_xseq[:,-1,:2]
        veh_start_lanelet = veh_lxseq[:,-1].numpy()
        if include_gt:
            veh_end_ctrs = veh_yseq[:,:,:2]
            veh_end_ctrs = veh_end_ctrs[row_idcs, last_idcs]
            veh_end_lanelet = veh_lyseq[row_idcs, last_idcs].numpy()

        node_DG = nx.DiGraph()
        path_DG = nx.DiGraph()
        #lanelet_DG = nx.DiGraph()
        for lanelet_id in lane2node.keys():
            lanelet = self.expert_cr_scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type or \
               LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                pri = 1
            else:
                pri = 2
            is_left = lanelet.adj_left is None
            path_DG.add_node(lanelet_id, pri=pri, left=is_left)
            for succ in lanelet.successor:
                if succ in lane2node:
                    path_DG.add_edge(lanelet_id, succ)
                    #lanelet_DG.add_edge(lanelet_id, succ)
            #if lanelet.adj_left is not None:
            #    lanelet_DG.add_edge(lanelet_id, lanelet.adj_left)
            #if lanelet.adj_right is not None:
            #    lanelet_DG.add_edge(lanelet_id, lanelet.adj_right)
        for n in range(lane_ctrs.size(0)):
            node_DG.add_node(n)
        flr_edge_index = torch.cat([suc_edges[0],
                                    left_edges,
                                    right_edges], -1)
        for j in range(flr_edge_index.size(-1)):
            length = torch.sqrt(((ctrs[flr_edge_index[0][j]]-ctrs[flr_edge_index[1][j]])**2).sum()).item()
            node_DG.add_edge(flr_edge_index[0][j].item(), flr_edge_index[1][j].item(),
                             length=length)

        dist = veh_start_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
        start_dist = torch.sqrt((dist**2).sum(2))
        start_mask = start_dist <= self.config["actor2map_dist"]
        start_dist_idcs = torch.nonzero(start_mask, as_tuple=False)
        sorted_start_dist, sorted_start_idcs = start_dist.sort(1, descending=False)
        if include_gt:
            dist = veh_end_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
            target_dist = torch.sqrt((dist**2).sum(2))
            target_mask = target_dist <=self.config["map2actor_dist"]
            target_dist_idcs = torch.nonzero(target_mask, as_tuple=False)
            sorted_target_dist, sorted_target_idcs = target_dist.sort(1, descending=False)

        # find the shortest paths
        obs_veh_index, fut_veh_index = [], []
        obs_lane_index, fut_lane_index = [], []
        #route_index = {r:[] for r in range(5)}
        #actor_routes = []
        #actor_targets = []
        veh_full_path = []

        veh_path_sender, veh_path_receiver = [], []
        path_node_sender, path_node_receiver = [], []
        path_num_nodes, path_node_node = [], []

        #delta_s, next_d = [], []
        for v in range(veh_start_ctrs.size(0)):
            start_vs = start_dist_idcs[:,0] == v
            if start_vs.sum() == 0:
                first_ns = list(sorted_start_idcs[v, :4].numpy())
            else:
                first_ns = list(start_dist_idcs[:, 1][start_vs].numpy())
            start_lanelet_cands = [node2lane[n] for n in first_ns]
            start_lanelet_cands = set(start_lanelet_cands)
            best_start_lanelet_id = veh_start_lanelet[v]

            if include_gt:
                last_ns = sorted_target_idcs[v, :8].numpy()
                last_dists = sorted_target_dist[v, :8].numpy()
                dest_lanelet_cands = [node2lane[n] for n in last_ns]
                dest_lanelet_cands = set(dest_lanelet_cands)
                # need to find the ground-truth start_lanelet, and dest_lanelet
                best_dest_lanelet_id = veh_end_lanelet[v]
                #print(best_dest_lanelet_id)
            #assert(best_start_lanelet_id is not None)
            #assert(best_dest_lanelet_id is not None)

            #if best_start_lanelet_id not in start_lanelet_cands or best_dest_lanelet_id not in dest_lanelet_cands:
            #    record = False

            # find the end lanelets each lanelet forms a path

            # start edges
            obs_veh_index.extend([v for _ in range(len(first_ns))])
            obs_lane_index.extend(first_ns)

            # compute all route lengths
            path_lengths = nx.multi_source_dijkstra_path_length(node_DG, set(first_ns), weight="length")
            #for n, d in zip(last_ns, last_dists):
            #    if n in path_lengths and path_lengths[n] < 40:
            #       actor_targets.append(n)
            #       break
            #else:
            #    print(v, last_ns, last_dists)
            #    raise
            # path edges
            #r = {kk:[] for kk in range(5)}
            #
            if include_gt:
                possible_lanelets = [best_start_lanelet_id, best_dest_lanelet_id]
            else:
                possible_lanelets = [best_start_lanelet_id]
            #print(possible_lanelets)
            for n, path_len in path_lengths.items():
                lid = node2lane[n]
                #if path_len < 10:#/self.config["grid_length"]:
                    #r[0].append([v, n])
                    #if n == actor_targets[-1]:
                    #    actor_routes.append(0)
                #    if lid not in possible_lanelets:
                #        possible_lanelets.append(lid)
                #elif path_len < 20:#/self.config["grid_length"]:
                #    r[1].append([v, n])
                    #if n == actor_targets[-1]:
                    #    actor_routes.append(1)
                #    if lid not in possible_lanelets:
                #        possible_lanelets.append(lid)
                #elif path_len < 30:#/self.config["grid_length"]:
                #    r[2].append([v, n])
                    #if n == actor_targets[-1]:
                    #    actor_routes.append(2)
                #    if lid not in possible_lanelets:
                #        possible_lanelets.append(lid)
                #elif path_len < 40:#/self.config["grid_length"]:
                if path_len < 40 and lid not in possible_lanelets:
                    #r[3].append([v, n])
                    #if n == actor_targets[-1]:
                    #    actor_routes.append(3)
                    #if lid not in possible_lanelets:
                    possible_lanelets.append(lid)
                #elif path_len < 50:#/self.config["grid_length"]:
                #    r[4].append([v, n])
                    #if n == actor_targets[-1]:
                    #    actor_routes.append(4)

            #assert(len(r[0]) > 0) # the closest
            """
            route_index[0].extend(r[0])
            if len(r[1]) == 0:
                assert(len(r[2]) ==0 and len(r[3]) ==0 and len(r[4]) ==0)
                route_index[1].extend(r[0][-2:])
                route_index[2].extend(r[0][-2:])
                route_index[3].extend(r[0][-2:])
                route_index[4].extend(r[0][-2:])
            elif len(r[2]) == 0:
                route_index[1].extend(r[1])
                assert(len(r[3]) ==0 and len(r[4]) ==0)
                route_index[2].extend(r[1][-2:])
                route_index[3].extend(r[1][-2:])
                route_index[4].extend(r[1][-2:])
            elif len(r[3]) == 0:
                route_index[1].extend(r[1])
                route_index[2].extend(r[2])
                assert(len(r[4]) ==0)
                route_index[3].extend(r[2][-2:])
                route_index[4].extend(r[2][-2:])
            elif len(r[4]) == 0:
                route_index[1].extend(r[1])
                route_index[2].extend(r[2])
                route_index[3].extend(r[3])
                route_index[4].extend(r[3][-2:])
            else:
                route_index[1].extend(r[1])
                route_index[2].extend(r[2])
                route_index[3].extend(r[3])
                route_index[4].extend(r[4])
            """
            #possible starts and possible targets
            #possible_sources = []
            possible_targets = []


            for lid in possible_lanelets:

                lanelet = self.expert_cr_scenario.lanelet_network.find_lanelet_by_id(lid)
                #is_source = True
                #for pred in lanelet.predecessor:
                #    if pred in possible_lanelets:
                #        is_source = False
                #        break
                is_target = True
                for succ in lanelet.successor:
                    if succ in possible_lanelets:
                        is_target = False
                        break
                #if is_source:
                #    possible_sources.append(lid)
                if is_target:
                    possible_targets.append(lid)

            possible_sources = [best_start_lanelet_id]
            lanelet_leftmost = self.expert_cr_scenario.lanelet_network.find_lanelet_by_id(best_start_lanelet_id)
            while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                lanelet_leftmost = self.expert_cr_scenario.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
                if lanelet_leftmost.lanelet_id not in possible_sources:
                    possible_sources.append(lanelet_leftmost.lanelet_id)
            lanelet_rightmost = self.expert_cr_scenario.lanelet_network.find_lanelet_by_id(best_start_lanelet_id)
            while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                lanelet_rightmost = self.expert_cr_scenario.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
                if lanelet_rightmost.lanelet_id not in possible_sources:
                    possible_sources.append(lanelet_rightmost.lanelet_id)

            # let's get all possible paths
            possible_paths = []
            for sou in possible_sources:
                shortest_lanelet_paths = nx.single_source_dijkstra_path(path_DG, sou)
                for tar in possible_targets:
                    if tar in shortest_lanelet_paths:
                        is_path = True
                        for llid in shortest_lanelet_paths[tar]:
                            if llid not in possible_lanelets:
                                is_path = False
                        if is_path:
                            possible_paths.append(shortest_lanelet_paths[tar])
            has_target = False
            target_path_inds = []
            for pind, path in enumerate(possible_paths):
                # a candidate path
                if include_gt:
                    if best_dest_lanelet_id in path:
                        #veh_full_path.append([1])
                        target_path_inds.append(pind)
                        has_target = True
                #else:
                    #veh_full_path.append([0])
                src, tar = path[0], path[-1]
                veh_path_sender.append(v)
                veh_path_receiver.append(len(path_num_nodes))
                num_path_nodes = 0
                for llid in path:
                    for nid in lane2node[llid]:
                        path_node_sender.append(len(path_num_nodes))
                        path_node_receiver.append(nid)
                        num_path_nodes += 1
                path_node_node_sender, path_node_node_receiver = [], []
                for pn in range(num_path_nodes):
                    # a path is forward connected
                    for ppn in range(pn, num_path_nodes):
                        path_node_node_sender.append(pn)
                        path_node_node_receiver.append(ppn)
                path_num_nodes.append(num_path_nodes)
                path_node_node.append(torch.tensor([path_node_node_sender,
                                                    path_node_node_receiver], dtype=torch.long))
            if include_gt:
                assert(has_target)
                if len(target_path_inds) == 1:
                    target_path_ind = target_path_inds[0]
                else:
                    select_main_inds = []
                    for pind in target_path_inds:
                        path = possible_paths[pind]
                        if path_DG.nodes[path[-1]]["pri"] == 2:
                            select_main_inds.append(pind)

                    if len(select_main_inds) == 1:
                        target_path_ind = select_main_inds[0]
                    elif len(select_main_inds) == 0:
                        select_left_inds = []
                        for pind in target_path_inds:
                            path = possible_paths[pind]
                            if path_DG.nodes[path[-1]]["left"]:
                                select_left_inds.append(pind)
                        if len(select_left_inds) == 0:
                            target_path_ind = target_path_inds[0]
                        else:
                            target_path_ind = select_left_inds[0]
                    else:
                        select_left_inds = []
                        for pind in select_main_inds:
                            path = possible_paths[pind]
                            if path_DG.nodes[path[-1]]["left"]:
                                select_left_inds.append(pind)
                        if len(select_left_inds) == 0:
                            target_path_ind = target_path_inds[0]
                        else:
                            target_path_ind = select_left_inds[0]
                for pind in range(len(possible_paths)):
                    if pind == target_path_ind:
                        veh_full_path.append([1])
                    else:
                        veh_full_path.append([0])

                target_path = possible_paths[target_path_ind]
                """
                vertices = []
                for i, lid in enumerate(target_path):
                    #lanelet = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lid)
                    for n in lane2node[lid]:
                        vertices.append(ctrs[n].numpy())
                vertices = np.array(vertices)
                target_curve = make_curve(vertices)

                ds, nd = [], []
                for i in range(veh_yseq[v,:,:2].size(0)):
                    if i == 0:
                        start_pos = veh_start_ctrs[v].numpy()
                        tar_pos = veh_yseq[v, i, :2].numpy()
                    else:
                        start_pos = veh_yseq[v, i-1, :2].numpy()
                        tar_pos = veh_yseq[v, i, :2].numpy()
                    if has_preds[v, i]:
                        start_pos = VecSE2(start_pos[0], start_pos[1], 0.0)
                        tar_pos = VecSE2(tar_pos[0], tar_pos[1], 0.0)
                        start_proj = start_pos.proj_on_curve(target_curve, clamped=False)
                        tar_proj = tar_pos.proj_on_curve(target_curve, clamped=False)
                        ds.append(lerp_curve_with_ind(target_curve, tar_proj.ind) - \
                                  lerp_curve_with_ind(target_curve, start_proj.ind))
                        nd.append(tar_proj.d)
                    else:
                        ds.append(0.0)
                        nd.append(0.0)
                delta_s.append(ds)
                next_d.append(nd)
                """
        ####################################################
        #last_ds = [d[-1] for d in next_d]
        #if np.max(np.abs(next_d)) > 6:
        #    record = False

        #if record == False:
        #    return None

        lane_start = torch.tensor([obs_veh_index,obs_lane_index], dtype=torch.long)
        #lane_path = {k:torch.tensor(v, dtype=torch.long).transpose(1, 0) for k,v in route_index.items()}
        #veh_target = torch.tensor(actor_targets, dtype=torch.long).unsqueeze(1)
        #veh_path = torch.tensor(actor_routes, dtype=torch.long).unsqueeze(1)
        #if include_gt:
        #    veh_full_path = torch.tensor(veh_full_path, dtype=torch.float)
            #delta_s = torch.tensor(delta_s, dtype=torch.float)
            #next_d = torch.tensor(next_d, dtype=torch.float)
            #veh_yfrenet = torch.stack([delta_s, next_d], -1)
            #traj_fre = veh_yfrenet.permute(1,0,2)

        path_num_nodes = torch.tensor(path_num_nodes, dtype=torch.long)
        veh_path_edge = torch.tensor([veh_path_sender, veh_path_receiver], dtype=torch.long)
        path_node_edge = torch.tensor([path_node_sender, path_node_receiver], dtype=torch.long)

        lane_ids = torch.tensor([node2lane[n] for n in range(ctrs.size(0))], dtype=torch.long)
        start_positions = veh_xseq[:,-1,:2].numpy()
        last_positions = veh_xseq[:,-2,:2].numpy()

        num_paths = path_num_nodes.size(0)
        #path_node = path_node_edge

        ########################################################################################
        traj_pred_rel, target_poses,  initial_poses, delta_s, next_d = [], [], [], [], []
        for pidx in range(num_paths):
            vv = lane_ctrs[path_node_edge[1][path_node_edge[0]==pidx]]
            cur = make_curve(vv.numpy())
            vidx = veh_path_edge[0, pidx]

            sp = VecSE2(start_positions[vidx, 0], start_positions[vidx, 1], 0.0)
            if len(cur) > 10:
                bound_ind = len(cur)//3
            else:
                bound_ind = len(cur)
            closest_ind = sp.index_closest_to_point(cur[:bound_ind])
            while closest_ind >= bound_ind-1:
                bound_ind +=1
                closest_ind = sp.index_closest_to_point(cur[:bound_ind])

            s_proj = sp.proj_on_curve(cur[:bound_ind], clamped=False)
            s_s = lerp_curve_with_ind(cur, s_proj.ind)

            lp = VecSE2(last_positions[vidx, 0], last_positions[vidx, 1], 0.0)
            l_proj = lp.proj_on_curve(cur[:bound_ind], clamped=False)
            l_s = lerp_curve_with_ind(cur, l_proj.ind)

            ds = s_s - l_s
            dd = s_proj.d - l_proj.d
            dT = self.config["dt"] * self.config["delta_step"]
            initial_poses.append([ds/dT, s_proj.d, dd/dT])

        lane_start = torch.tensor(initial_poses, dtype=torch.float32)

        if include_gt:
            veh_full_path = torch.tensor(veh_full_path, dtype=torch.float)
            veh_path_idcs = torch.arange(0, num_paths).unsqueeze(1)[veh_full_path == 1].unsqueeze(1)
            vertices = [lane_ctrs[path_node_edge[1][path_node_edge[0]==pidx]] for pidx in veh_path_idcs]
            for idx in range(veh_yseq.size(0)):
                target_curve = make_curve(vertices[idx].numpy())
                start_pos = VecSE2(start_positions[idx, 0], start_positions[idx, 1], 0.0)
                if len(target_curve) > 10:
                    bound_ind = len(target_curve)//3
                else:
                    bound_ind = len(target_curve)
                closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])
                while closest_ind >= bound_ind-1:
                    bound_ind +=1
                    closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])
                start_proj = start_pos.proj_on_curve(target_curve[:bound_ind], clamped=False)
                #print(idx, start_pos, start_proj.d, start_proj.ind)

                footpoint = lerp_curve(target_curve[start_proj.ind.i], target_curve[start_proj.ind.i+1], start_proj.ind.t)
                th = footpoint.pos.th + 0.5*np.pi
                next_posG = VecSE2(footpoint.pos.x + start_proj.d*np.cos(th), footpoint.pos.y + start_proj.d*np.sin(th), footpoint.pos.th + 0.0)
                #print("convert", next_posG, footpoint)

                start_s = lerp_curve_with_ind(target_curve, start_proj.ind)

                last_pos = VecSE2(last_positions[idx, 0], last_positions[idx, 1], 0.0)
                last_proj = last_pos.proj_on_curve(target_curve[:bound_ind], clamped=False)
                last_s = lerp_curve_with_ind(target_curve, last_proj.ind)

                ds, nd = [], []
                current_bound_ind = bound_ind

                inds = []
                for t in range(veh_yseq.size(1)):
                    if t == 0:
                        sta_pos = start_positions[idx]
                        tar_pos = veh_yseq[idx, t, :2].numpy()
                    else:
                        sta_pos = veh_yseq[idx, t-1, :2].numpy()
                        tar_pos = veh_yseq[idx, t, :2].numpy()
                    if has_preds[idx, t]:
                        sta_pos = VecSE2(sta_pos[0], sta_pos[1], 0.0)
                        tar_pos = VecSE2(tar_pos[0], tar_pos[1], 0.0)
                        sta_proj = sta_pos.proj_on_curve(target_curve[:current_bound_ind], clamped=False)
                        tar_proj = tar_pos.proj_on_curve(target_curve[:current_bound_ind], clamped=False)

                        inds.append(sta_proj.ind)

                        assert(sta_proj.ind.i < current_bound_ind-1)
                        assert(tar_proj.ind.i < current_bound_ind-1)
                        current_bound_ind = max(current_bound_ind, min(tar_proj.ind.i+5, len(target_curve)))
                        ds.append(lerp_curve_with_ind(target_curve, tar_proj.ind) - \
                                  lerp_curve_with_ind(target_curve, sta_proj.ind))
                        nd.append(tar_proj.d)
                    else:
                        ds.append(ds[-1])
                        nd.append(nd[-1])
                delta_s.append(ds)
                next_d.append(nd)

                #delta_s = torch.tensor(ds, dtype=torch.float)
                #if delta_s.sum().item() < -1:
                #    record = False
                #assert delta_s.sum().item() > -1, "{}, {}".format(delta_s, inds)
                #next_d = torch.tensor(nd, dtype=torch.float)
                #traj_fre[:, idx] = torch.stack([delta_s, next_d], -1)

                #s = traj_fre[:, idx, 0].sum().item()
                #ds = traj_fre[-1, idx, 0].item()
                #d = traj_fre[-1, idx, 1].item()
                #dd = (traj_fre[-1, idx, 1] - traj_fre[-2, idx, 1]).item()
                s = np.sum(ds)
                ss = ds[-1]
                d = nd[-1]
                dd = nd[-1] - nd[-2]
                target_poses.append([s, ss/(self.config["dt"]*self.config["delta_step"]),
                                     d, dd/(self.config["dt"]*self.config["delta_step"])])

                traj = []
                start_ind = start_proj.ind
                for t in range(veh_yseq.size(1)):
                    #next_ind, next_pos = move_along_curve(start_ind, target_curve, traj_fre[t, idx, 0].cpu().item(), traj_fre[t, idx, 1].cpu().item())
                    next_ind, next_pos = move_along_curve(start_ind, target_curve, ds[t], nd[t])
                    traj.append([next_pos.x-start_pos.x, next_pos.y-start_pos.y])
                    start_ind = next_ind
                    start_pos = next_pos
                traj_pred_rel.append(traj)

            #veh_yfrenet = torch.stack([delta_s, next_d], -1)
            veh_target = torch.tensor(target_poses, dtype=torch.float)
            traj_pred_rel = torch.tensor(traj_pred_rel, dtype=torch.float)
            delta_s = torch.tensor(delta_s, dtype=torch.float).unsqueeze(-1)
            next_d = torch.tensor(next_d, dtype=torch.float).unsqueeze(-1)
            veh_yfre = torch.cat([delta_s, next_d, traj_pred_rel], -1)
            #veh_yfre = torch.cat([traj_fre.permute(1,0,2), traj_pred_rel], dim=-1)

        if include_gt:
            data = ScenarioData(#veh_t=torch.tensor(time_steps, dtype=torch.long),
                                                veh_xseq=veh_xseq,
                                                veh_shape=torch.tensor(shape, dtype=torch.float),
                                                veh_yseq=veh_yseq,
                                                veh_yfre = veh_yfre,
                                                #veh_path=veh_path,
                                                veh_full_path=veh_full_path,
                                                veh_target=veh_target,
                                                veh_has_preds=has_preds,
                                                veh_edge_index=torch.tensor([sender,receiver], dtype=torch.long),
                                                veh_edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                                veh_id=torch.tensor(id, dtype=torch.float),

                                                veh_path_edge_index=veh_path_edge,
                                                path_lane_edge_index=path_node_edge,
                                                path_num_nodes=path_num_nodes,
                                                path_node_node_edge_index=path_node_node,
                                                lane_id = lane_ids,
                                                lane_ctrs = ctrs,
                                                lane_vecs = vecs,
                                                lane_pris = pris,
                                                lane_widths = widths,
                                                lane_suc_edge_index = suc_edges,
                                                lane_pre_edge_index = pre_edges,
                                                lane_left_edge_index = left_edges,
                                                lane_right_edge_index = right_edges,
                                                lane_start = lane_start,
                                                #lane_path = lane_path,
                                                )
        else:
            data = ScenarioData(#veh_t=torch.tensor(time_steps, dtype=torch.long),
                                                veh_xseq=veh_xseq,
                                                veh_shape=torch.tensor(shape, dtype=torch.float),
                                                veh_edge_index=torch.tensor([sender,receiver], dtype=torch.long),
                                                veh_edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                                veh_id=torch.tensor(id, dtype=torch.float),

                                                veh_path_edge_index=veh_path_edge,
                                                path_lane_edge_index=path_node_edge,
                                                path_num_nodes=path_num_nodes,
                                                path_node_node_edge_index=path_node_node,
                                                lane_id = lane_ids,
                                                lane_ctrs = ctrs,
                                                lane_vecs = vecs,
                                                lane_pris = pris,
                                                lane_widths = widths,
                                                lane_suc_edge_index = suc_edges,
                                                lane_pre_edge_index = pre_edges,
                                                lane_left_edge_index = left_edges,
                                                lane_right_edge_index = right_edges,
                                                lane_start = lane_start,
                                                #lane_path = lane_path,
                                                )
        return data

    def step(self, action):
        pass

    def reward(self):
        pass

    def render(self, h=5, w=5, add_centroid=True, fill_lanelet=True, draw_scene=True, show_lane=False):
        pass


class GIDMReplayRoundabout(ReplayRoundabout):
    def __init__(self, config, seed=36, load_path=True, motion_predictor=None, connected=False, dummy=False):
        super(GIDMReplayRoundabout, self).__init__(
            config,
            seed,
            load_path
        )

        self.vehicle_exts = MultiFeatureExtractor([CoreFeatureExtractor0(self.config),
                                          NeighborFeatureExtractor0(self.config),
                                          NeighborFeatureExtractor()])
        self.motion_predictor = motion_predictor
        self.connected = connected
        self.dummy = dummy

    def custom_reset(self):
        raw_path = './data/rounD/raw/rounD_04_dump.pk'
        ts = 2585 + 15
        te = 2998
        self.traj_idx = traj_idx = self.raw_paths.index(raw_path)
        self.lanelet_network = self.lanelet_networks[self.vehicleinfos[traj_idx][0]]
        egoids_list = [67, 68]
        self.vehicleinfos[traj_idx][1][68]["shape"]._length = 4.7383
        self.vehicleinfos[traj_idx][1][68]["shape"]._width = 2.0369

        vehids = [54, 55, 57, 60, 63, 67, 68]

        self.te = te
        self.background_vehicles = dict()
        self.ego_vehicles = dict()
        self.obstacles = defaultdict(lambda:dict())
        self.obstacles_preds = defaultdict(lambda:dict())
        self.connected_preds = defaultdict(lambda:dict())

        self.idm_params = pickle.load(open("./data/rounD/processed_new/rounD_params.pk", "rb"))
        self.idm_params_rng = np.random.default_rng(12345)

        self.source_lanelet_ids, self.goal_lanelet_ids = [], []
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.successor) == 0:
                self.goal_lanelet_ids.append(lanelet.lanelet_id)
            if len(lanelet.predecessor) == 0:
                self.source_lanelet_ids.append(lanelet.lanelet_id)

        self.waiting_list = []
        self.egoids_list = egoids_list
        for vehid in vehids:
            ts_, te_ = self.vehicleinfos[traj_idx][1][vehid]["frames"]
            stopped, stop_steps = self.is_stopped(vehid, ts, ts_, te_)
            if stopped:
                shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                              self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                              center=self.trajectories[traj_idx][ts][vehid].position,
                                              orientation=self.trajectories[traj_idx][ts][vehid].orientation))
                for t in range(ts-int(2/self.config['dt']), ts+stop_steps):
                    initial_state=copy(self.trajectories[traj_idx][t][vehid])
                    initial_state.time_step = t
                    initial_state.acceleration = 0.0
                    self.obstacles[t][vehid] = StaticObstacle(obstacle_id=vehid,
                        obstacle_type=self.vehicleinfos[traj_idx][1][vehid]["type"],
                        initial_state=initial_state,
                        obstacle_shape=self.vehicleinfos[traj_idx][1][vehid]["shape"],
                        initial_shape_lanelet_ids=set(shape_lanelet_ids))
                continue
            states = [self.trajectories[self.traj_idx][t][vehid] for t in range(ts, te_+1)]
            if len(states) == 1:
                continue
            if self.lanelet_network.find_lanelet_by_id(states[0].posF.ind[1]) is None:
                continue
            if vehid in egoids_list:

                self.ego_vehicles[vehid] = EgoVehicle(vehid, copy(states[0]), copy(states[-1]),
                                        self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                        self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                        self.lanelet_network)
                if self.dummy:
                    ego_veh = GIDMVehicle(vehid, states,
                                                   self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                                   self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                                   self.lanelet_network, self.source_lanelet_ids, self.goal_lanelet_ids)
                    ego_veh.path = self.ego_vehicles[vehid].path
                    ego_veh.center_lanelet_occupancies = [set([l]) for l in self.ego_vehicles[vehid].center_lanelet_occupancies]
                    ego_veh.stopline_occupancies = self.ego_vehicles[vehid].stopline_occupancies
                    ego_veh.ind = self.ego_vehicles[vehid].ind
                    self.ego_vehicles[vehid] = ego_veh

            else:
                if self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.TRUCK:
                    part = self.idm_params['tru']
                elif self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.MOTORCYCLE:
                    part = self.idm_params['mot']
                elif self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.BUS:
                    part = self.idm_params['bus']
                else:
                    part = self.idm_params['car']
                ind = self.idm_params_rng.integers(low=0, high=len(part), size=1)[0]
                vstar, dmin, Tmin, amax, bconf, ttm, sig, _ = part[ind]
                self.background_vehicles[vehid] = GIDMVehicle(vehid, states,
                                               self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                               self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                               self.lanelet_network, self.source_lanelet_ids, self.goal_lanelet_ids,
                                               vstar = vstar,
                                               T = Tmin,
                                               amax = amax,
                                               )
            for t in range(max(ts_, ts-self.config["horizon_steps"]), ts+1):
            #for t in range(ts, ts+1):
                shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                              self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                              center=self.trajectories[traj_idx][t][vehid].position,
                                              orientation=self.trajectories[traj_idx][t][vehid].orientation))
                initial_state=copy(self.trajectories[traj_idx][t][vehid])
                initial_state.time_step = t
                initial_state.acceleration = 0.0
                self.obstacles[t][vehid] = DynamicObstacle(obstacle_id=vehid,
                    obstacle_type=self.vehicleinfos[traj_idx][1][vehid]["type"],
                    initial_state=initial_state,
                    obstacle_shape=self.vehicleinfos[traj_idx][1][vehid]["shape"],
                    initial_center_lanelet_ids=set([self.trajectories[traj_idx][t][vehid].posF.ind[1]]),
                    initial_shape_lanelet_ids=set(shape_lanelet_ids))

        self.expert_cr_scenario = ScenarioWrapper(self.config["dt"], self.lanelet_network,
                                              self.vehicleinfos[traj_idx][0], self.obstacles, None)
        self.expert_cr_scenario.scenario_id = "{}_{}".format(traj_idx, ts)
        self.expert_cr_scenario.set_sensor_range(self.config["max_veh_radius"],
                                            self.config["max_veh_disp_front"],
                                            self.config["max_veh_disp_rear"],
                                            self.config["max_veh_disp_side"])
        self.time_step = ts
        if self.motion_predictor:
            self.update_motion_prediction()
        else:
            self.update_simple_motion_prediction()

        if self.connected:
            for egoid in self.egoids_list:
                self.connected_preds[egoid] = self.obstacles_preds[self.time_step][egoid]

        for vehid, veh in self.background_vehicles.items():
            if veh.arrived:
                continue
            obs = []
            for obs_id, ob in self.obstacles[self.time_step].items():
                if obs_id==vehid or isinstance(ob, StaticObstacle):
                    continue
                obs.append(ob)
            veh.update_neighbors(obs)
        states = dict()
        return states


    def reset(self, info, egoids_list=[]):
        if isinstance(info, str):
            trajdata = pickle.load(open(info, "rb"))
            ts, te = trajdata["ranges"]
            self.te = te
            self.traj_idx = traj_idx = 0
            fp = trajdata["mapfile"]
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in lanelet_network.lanelets}
            self.lanelet_network = make_lanelet_curve_network(lanelet_network, speed_limits)
            self.trajectories = [trajdata["state"]]
            self.vehicleinfos = [(fp, trajdata["infos"])]
        elif isinstance(info, list):
            raw_path, egoid, ts, te = info
            self.traj_idx = traj_idx = self.raw_paths.index(raw_path)
            self.lanelet_network = self.lanelet_networks[self.vehicleinfos[traj_idx][0]]
            egoids_list = [egoid]
        else:
            data, traj_idx = info
            self.traj_idx = traj_idx
            vehs_t = data.veh_t.numpy()
            ts, te = int(vehs_t[0]), int(vehs_t[-1]+self.config['num_preds']*self.config['delta_step'])
            self.lanelet_network = self.lanelet_networks[self.vehicleinfos[traj_idx][0]]

        # presimulation
        presim = True
        while presim:
            if len(egoids_list) == 0:
                break
            for egoid in egoids_list:
                state = self.trajectories[self.traj_idx][ts][egoid]
                if state.posF.ind[0].i==0 and state.posF.ind[0].t == 0:
                    presim = True
                    break
            else:
                presim = False
                break
            ts += self.config['delta_step']

        vehids = list(self.trajectories[traj_idx][ts].keys())
        self.te = te
        self.background_vehicles = dict()
        self.ego_vehicles = dict()
        self.obstacles = defaultdict(lambda:dict())
        self.obstacles_preds = defaultdict(lambda:dict())
        self.idm_params = pickle.load(open("./data/rounD/rounD_params.pk", "rb"))
        self.idm_params_rng = np.random.default_rng(12345)


        self.source_lanelet_ids, self.goal_lanelet_ids = [], []
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.successor) == 0:
                self.goal_lanelet_ids.append(lanelet.lanelet_id)
            if len(lanelet.predecessor) == 0:
                self.source_lanelet_ids.append(lanelet.lanelet_id)

        self.waiting_list = []
        self.egoids_list = egoids_list
        for vehid in vehids:
            ts_, te_ = self.vehicleinfos[traj_idx][1][vehid]["frames"]
            stopped, stop_steps = self.is_stopped(vehid, ts, ts_, te_)
            if stopped:
                shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                              self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                              center=self.trajectories[traj_idx][ts][vehid].position,
                                              orientation=self.trajectories[traj_idx][ts][vehid].orientation))
                for t in range(ts-int(2/self.config['dt']), ts+stop_steps):
                    initial_state=copy(self.trajectories[traj_idx][t][vehid])
                    initial_state.time_step = t
                    initial_state.acceleration = 0.0
                    self.obstacles[t][vehid] = StaticObstacle(obstacle_id=vehid,
                        obstacle_type=self.vehicleinfos[traj_idx][1][vehid]["type"],
                        initial_state=initial_state,
                        obstacle_shape=self.vehicleinfos[traj_idx][1][vehid]["shape"],
                        initial_shape_lanelet_ids=set(shape_lanelet_ids))
                continue
            states = [self.trajectories[self.traj_idx][t][vehid] for t in range(ts, te_+1)]
            if len(states) == 1:
                continue
            if self.lanelet_network.find_lanelet_by_id(states[0].posF.ind[1]) is None:
                continue
            if vehid in egoids_list:
                self.ego_vehicles[vehid] = EgoVehicle(vehid, copy(states[0]), copy(states[-1]),
                                        self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                        self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                        self.lanelet_network)
            else:
                if self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.TRUCK:
                    part = self.idm_params['tru']
                elif self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.MOTORCYCLE:
                    part = self.idm_params['mot']
                elif self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.BUS:
                    part = self.idm_params['bus']
                else:
                    part = self.idm_params['car']
                ind = self.idm_params_rng.integers(low=0, high=len(part), size=1)[0]
                vstar, dmin, Tmin, amax, bconf, ttm, sig, _ = part[ind]
                self.background_vehicles[vehid] = GIDMVehicle(vehid, states,
                                               self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                               self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                               self.lanelet_network, self.source_lanelet_ids, self.goal_lanelet_ids,
                                               vstar = vstar,
                                               T = Tmin,
                                               amax = amax,
                                               )
            for t in range(max(ts_, ts-self.config["horizon_steps"]), ts+1):
            #for t in range(ts, ts+1):
                shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                              self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                              center=self.trajectories[traj_idx][t][vehid].position,
                                              orientation=self.trajectories[traj_idx][t][vehid].orientation))
                initial_state=copy(self.trajectories[traj_idx][t][vehid])
                initial_state.time_step = t
                initial_state.acceleration = 0.0
                self.obstacles[t][vehid] = DynamicObstacle(obstacle_id=vehid,
                    obstacle_type=self.vehicleinfos[traj_idx][1][vehid]["type"],
                    initial_state=initial_state,
                    obstacle_shape=self.vehicleinfos[traj_idx][1][vehid]["shape"],
                    initial_center_lanelet_ids=set([self.trajectories[traj_idx][t][vehid].posF.ind[1]]),
                    initial_shape_lanelet_ids=set(shape_lanelet_ids))

        ###########
        # test obstacle
        '''
        for t in range(240):
            initial_state = FrenetState(position = np.array([117.08875, -54.56]),
                                           orientation=2.77,
                                           velocity=0,
                                           acceleration=0,
                                           acceleration_y=0,
                                           time_step=ts-120+t,
                                           yaw_rate=0.0,
                                           slip_angle=0.0)
            initial_state.set_posF(self.lanelet_network, [105])
            o = StaticObstacle(obstacle_id=-1,
                                    obstacle_type=self.vehicleinfos[traj_idx][1][11]["type"],
                                    initial_state=initial_state,
                                    obstacle_shape=self.vehicleinfos[traj_idx][1][11]['shape'],
                                    initial_shape_lanelet_ids=set([105]))

            self.obstacles[ts-120+t][-1] = o
        '''
        ##########
        self.expert_cr_scenario = ScenarioWrapper(self.config["dt"], self.lanelet_network,
                                              self.vehicleinfos[traj_idx][0], self.obstacles, None)
        self.expert_cr_scenario.scenario_id = "{}_{}".format(traj_idx, ts)
        self.expert_cr_scenario.set_sensor_range(self.config["max_veh_radius"],
                                            self.config["max_veh_disp_front"],
                                            self.config["max_veh_disp_rear"],
                                            self.config["max_veh_disp_side"])
        self.time_step = ts
        if self.motion_predictor:
            self.update_motion_prediction()
        else:
            self.update_simple_motion_prediction()

        for vehid, veh in self.background_vehicles.items():
            if veh.arrived:
                continue
            obs = []
            for obs_id, ob in self.obstacles[self.time_step].items():
                if obs_id==vehid or isinstance(ob, StaticObstacle):
                    continue
                obs.append(ob)
            veh.update_neighbors(obs)
        states = dict()
        return states

    def step(self, actions):
        # apply actions
        accs = dict()
        if self.dummy:
            for vehid, veh in self.ego_vehicles.items():
                if veh.arrived:
                    continue
                if veh.front_on_same_lane:
                    front_vehid = veh.front_vehid
                else:
                    front_vehid = veh.merge_front_vehid
                obs = []
                for obs_id, ob in self.obstacles[self.time_step].items():
                    if obs_id==vehid or isinstance(ob, StaticObstacle):
                        continue
                    # check dead lock
                    if obs_id in self.background_vehicles and front_vehid is not None:
                        obs_veh = self.background_vehicles[obs_id]
                        if obs_veh.front_on_same_lane:
                            obs_front_vehid = obs_veh.front_vehid
                        else:
                            obs_front_vehid = obs_veh.merge_front_vehid
                        if front_vehid == obs_id and obs_front_vehid == vehid:
                            veh.front_vehid = veh.merge_front_vehid = None
                            #if not veh.front_on_same_lane:
                            #    veh.check_merge = False
                            continue
                    obs.append(ob)
                if veh.check_merge:
                    for llid in self.obstacles[self.time_step][vehid].initial_shape_lanelet_ids:
                        if llid not in self.source_lanelet_ids:
                            veh.check_merge = False
                            break
                accs[vehid] = veh.get_accel(obs)
        else:
            for egoid, ego in self.ego_vehicles.items():
                ego.apply_action(actions[egoid])


        for vehid, veh in self.background_vehicles.items():
            if veh.arrived:
                continue
            if veh.front_on_same_lane:
                front_vehid = veh.front_vehid
            else:
                front_vehid = veh.merge_front_vehid
            obs = []
            for obs_id, ob in self.obstacles[self.time_step].items():
                if obs_id==vehid or isinstance(ob, StaticObstacle):
                    continue
                # check dead lock
                if obs_id in self.background_vehicles and front_vehid is not None:
                    obs_veh = self.background_vehicles[obs_id]
                    if obs_veh.front_on_same_lane:
                        obs_front_vehid = obs_veh.front_vehid
                    else:
                        obs_front_vehid = obs_veh.merge_front_vehid
                    if front_vehid == obs_id and obs_front_vehid == vehid:
                        veh.front_vehid = veh.merge_front_vehid = None
                        #if not veh.front_on_same_lane:
                        #    veh.check_merge = False
                        continue
                obs.append(ob)
            if veh.check_merge:
                for llid in self.obstacles[self.time_step][vehid].initial_shape_lanelet_ids:
                    if llid not in self.source_lanelet_ids:
                        veh.check_merge = False
                        break
            accs[vehid] = veh.get_accel(obs) # get the GIDM acc
        for t in range(self.config["sim_delta_step"]):
            for vehid, veh in self.background_vehicles.items():
                if veh.arrived:
                    continue
                #if vehid == 40:
                #    print(acc, veh.front_vehid, veh.merge_front_vehid, veh.front_s, veh.front_v, veh.v, veh.front_on_same_lane)
                self.update_vehicle(veh, accs[vehid]) # update the state of each vehicle
            for egoid, ego in self.ego_vehicles.items():
                if ego.arrived:
                    continue
                if self.dummy:
                    self.update_vehicle(ego, accs[egoid])
                else:
                    self.update_ego_vehicle(ego) # update the state of each ego
            self.time_step += 1
            # add new vehicles
            vehids = list(self.trajectories[self.traj_idx][self.time_step].keys())
            for vehid in vehids:
                if vehid == 73:
                    continue
                if vehid not in self.background_vehicles.keys() and \
                vehid not in self.ego_vehicles.keys() and \
                vehid not in self.obstacles[self.time_step].keys() and \
                vehid not in self.waiting_list: # not on waiting list
                   self.waiting_list.append(vehid)

        new_waiting_list = []
        for vehid in self.waiting_list:
            # TODO check if the veh can be inserted (location not occupanied)
            ts_, te_ = self.vehicleinfos[self.traj_idx][1][vehid]["frames"]
            states = [self.trajectories[self.traj_idx][tt][vehid] for tt in range(ts_, te_+1)]

            if self.lanelet_network.find_lanelet_by_id(states[0].posF.ind[1]) is None:
                continue
            if states[0].posF.ind[1] not in self.source_lanelet_ids and vehid not in self.obstacles[self.time_step-1].keys():
                continue
            if self.lanelet_occupied(states[0], self.vehicleinfos[self.traj_idx][1][vehid]["shape"].length) and vehid not in self.obstacles[self.time_step-1].keys():
                new_waiting_list.append(vehid)
                continue
            # TODO add ego vehicles
            initial_state = copy(states[0])
            initial_state.time_step = self.time_step
            initial_state.acceleration = 0.0
            shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[self.traj_idx][1][vehid]["shape"].length,
                                                  self.vehicleinfos[self.traj_idx][1][vehid]["shape"].width,
                                                  center=initial_state.position,
                                                  orientation=initial_state.orientation))
            on_source_lanelet = True
            if vehid not in self.obstacles[self.time_step-1].keys():
                for lanelet_id in shape_lanelet_ids:
                    if lanelet_id not in self.source_lanelet_ids:
                        on_source_lanelet = False
                        break
            if not on_source_lanelet:
                continue
            if vehid in self.egoids_list:
                initial_state, goal_state = states[0], states[-1]
                self.ego_vehicles[vehid] = EgoVehicle(vehid, initial_state, goal_state,
                                            self.vehicleinfos[traj_idx][1][vehid]["shape"].width,
                                            self.vehicleinfos[traj_idx][1][vehid]["shape"].length,
                                            self.lanelet_network)
            else:
                if self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.TRUCK:
                    part = self.idm_params['tru']
                elif self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.MOTORCYCLE:
                    part = self.idm_params['mot']
                elif self.vehicleinfos[self.traj_idx][1][vehid]["type"] == ObstacleType.BUS:
                    part = self.idm_params['bus']
                else:
                    part = self.idm_params['car']
                ind = self.idm_params_rng.integers(low=0, high=len(part), size=1)[0]
                vstar, dmin, Tmin, amax, bconf, ttm, sig, _ = part[ind]
                self.background_vehicles[vehid] = GIDMVehicle(vehid, states,
                                                       self.vehicleinfos[self.traj_idx][1][vehid]["shape"].width,
                                                       self.vehicleinfos[self.traj_idx][1][vehid]["shape"].length,
                                                       self.lanelet_network, self.source_lanelet_ids, self.goal_lanelet_ids,
                                                       vstar = vstar,
                                                       T = Tmin,
                                                       amax = amax,)
            self.obstacles[self.time_step][vehid] = DynamicObstacle(obstacle_id=vehid,
                        obstacle_type=self.vehicleinfos[self.traj_idx][1][vehid]["type"],
                        initial_state=initial_state,
                        obstacle_shape=self.vehicleinfos[self.traj_idx][1][vehid]["shape"],
                        initial_center_lanelet_ids=set([initial_state.posF.ind[1]]),
                        initial_shape_lanelet_ids=set(shape_lanelet_ids))

        self.waiting_list = new_waiting_list
        if self.motion_predictor:
            self.update_motion_prediction()
            self.update_simple_motion_prediction()

        else:
            self.update_simple_motion_prediction()
        if self.connected:
            self.update_communication()
        states = None
        dones = None
        infos = None
        return states, dones, infos

    def update_communication(self):
        for egoid in self.egoids_list:
            self.connected_preds[egoid] = []
            if self.ego_vehicles[egoid].arrived:
                continue
            for traj, dist in zip(self.ego_vehicles[egoid]._all_planned_trajectories[self.time_step-self.config['delta_step']],
                                  self.ego_vehicles[egoid]._all_planned_dists[self.time_step-self.config['delta_step']]):
                planned_states = traj[4:]
                self.connected_preds[egoid].append( (TrajectoryPrediction(
                    Trajectory(self.time_step, planned_states),
                    self.vehicleinfos[self.traj_idx][1][egoid]["shape"]
                ), dist)
                )

    def update_motion_prediction(self):
        # Option One: Focus on one ego vehicle , some vehicle may be out of range
        # Option Two: Cover all vehicles
        egoid = None
        for vehid in self.egoids_list:
            if not self.ego_vehicles[vehid].arrived:
                egoid = vehid
                break
        #egoid = 68
        #if self.ego_vehicles[self.egoids_list[0]].arrived:
        if egoid is None:
            return None
        inputs = self.generate_MP_state(egoid, self.time_step, include_gt=False)
        #if inputs.veh_id.size(0) == 1:
        if inputs is None:
            return None
        inputs = traj_collate([inputs])
        with torch.no_grad():
            predictions = self.motion_predictor.predict(inputs)
        #ego_posG = self.ego_vehicles[self.egoids_list[0]].current_state.posG
        ego_posG = self.obstacles[self.time_step][egoid].initial_state.posG
        s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
        # update each obstacles future occupancies
        last_pos = predictions["last_pos"]
        num_agents = len(predictions["veh_ids"])
        dT = self.config["dt"]*self.config["delta_step"]
        #self.obstacles_preds[self.time_step] = {}
        for aidx in range(num_agents):
            vehid = predictions["veh_ids"][aidx]
            if vehid in self.egoids_list and self.dummy:
                continue
            if vehid in self.egoids_list and self.time_step-self.config['delta_step'] in self.ego_vehicles[vehid]._planned_trajectories:
                planned_states = self.ego_vehicles[vehid]._planned_trajectories[self.time_step-self.config['delta_step']][4:]
                self.obstacles[self.time_step][vehid]._prediction = TrajectoryPrediction(
                        Trajectory(self.time_step, planned_states),
                        self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                    )
                self.obstacles_preds[self.time_step][vehid] = [
                            (TrajectoryPrediction(
                                Trajectory(self.time_step, planned_states),
                                self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                            ), 1.0)
                        ]

                continue
            if vehid not in self.obstacles[self.time_step-self.config['delta_step']]:
                continue
            scores = predictions["scores"][aidx]
            smidx = np.argmax(scores)
            self.obstacles_preds[self.time_step][vehid] = []
            for midx in range(len(predictions["trajs"])):
                traj_fre = predictions["trajs"][midx][aidx]
                target_curve = make_curve(predictions["curves"][midx][aidx])
                start_pos = VecSE2(last_pos[aidx, 0], last_pos[aidx, 1], 0.0)
                if len(target_curve) > 10:
                    bound_ind = len(target_curve)//3
                else:
                    bound_ind = len(target_curve)
                closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])
                while closest_ind >= bound_ind-1:
                    bound_ind +=1
                    closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])
                start_proj = start_pos.proj_on_curve(target_curve[:bound_ind], clamped=False)
                start_s = lerp_curve_with_ind(target_curve, start_proj.ind)
                start_ind = start_proj.ind

                if isinstance(self.obstacles[self.time_step][vehid], StaticObstacle):
                    traj = [self.obstacles[self.time_step][vehid].initial_state]
                    for tidx in range(self.config["num_preds"]):
                        time_step = self.time_step+(tidx+1)*self.config["delta_step"]
                        for tt in range(self.config["delta_step"]):
                            cstate = copy(self.obstacles[self.time_step][vehid].initial_state)
                            cstate.time_step = time_step-self.config["delta_step"]+tt+1
                            traj.append(cstate)
                    #try:
                    self.obstacles_preds[self.time_step][vehid].append(
                                (TrajectoryPrediction(
                                    Trajectory(self.time_step, traj),
                                    self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                                ), scores[midx])
                            )
                    #except:
                    #    self.obstacles_preds[self.time_step][vehid].append(
                    #            (TrajectoryPrediction(
                    #                Trajectory(self.time_step, traj),
                    #                self.vehicleinfos[self.traj_idx][1][11]["shape"]
                    #            ), scores[midx])
                    #        )
                    continue
                traj = []
                for tidx in range(self.config["num_preds"]):
                    next_ind, next_pos = move_along_curve(start_ind, target_curve,
                                                          traj_fre[tidx, 0],
                                                          traj_fre[tidx, 1])

                    x = c*next_pos.x - s*next_pos.y + ego_posG.x
                    y = s*next_pos.x + c*next_pos.y + ego_posG.y
                    lx = c*start_pos.x - s*start_pos.y + ego_posG.x
                    ly = s*start_pos.x + c*start_pos.y + ego_posG.y
                    velocity = np.sqrt((x-lx)**2+(y-ly)**2)/dT
                    heading = np.arctan2(y-ly, x-lx)

                    if tidx == self.config["num_preds"]-1:
                        sa = (traj_fre[tidx, 0] - traj_fre[tidx-1, 0])/dT**2
                        da = (traj_fre[tidx, 1]-2*traj_fre[tidx-1, 1]+traj_fre[tidx-2, 1])/dT**2
                        phi = np.arctan2(traj_fre[tidx, 1]-traj_fre[tidx-1, 1], traj_fre[tidx, 0])
                    elif tidx == 0:
                        sa = (traj_fre[tidx+1, 0] - traj_fre[tidx, 0])/dT**2
                        da = (traj_fre[tidx+1, 1]-2*traj_fre[tidx, 1]+start_proj.d)/dT**2
                        phi = np.arctan2(traj_fre[tidx, 1]-start_proj.d, traj_fre[tidx, 0])
                    else:
                        sa = (traj_fre[tidx+1, 0] - traj_fre[tidx, 0])/dT**2
                        da = (traj_fre[tidx+1, 1]-2*traj_fre[tidx, 1]+traj_fre[tidx-1, 1])/dT**2
                        phi = np.arctan2(traj_fre[tidx, 1]-traj_fre[tidx-1, 1], traj_fre[tidx, 0])

                    if tidx == 0:
                        initial_state = copy(self.obstacles[self.time_step][vehid].initial_state)
                        initial_state.acceleration = sa
                        initial_state.acceleration_y= da
                        initial_state.yaw_rate = 0.0
                        initial_state.slip_angle = 0.0
                        traj.append(initial_state)

                    if velocity < 0.5:
                        x = traj[-1].position[0]
                        y = traj[-1].position[1]
                        heading = traj[-1].orientation

                    state = FrenetState(position = np.array([x, y]),
                                               orientation=heading,
                                               velocity=velocity,
                                               acceleration=sa,
                                               acceleration_y=da,
                                               time_step=self.time_step+(tidx+1)*self.config["delta_step"],
                                               yaw_rate=0.0,
                                               slip_angle=0.0)
                    if next_ind.i < len(predictions["lane_ids"][midx][aidx]):
                        lanelet_list = [predictions["lane_ids"][midx][aidx][next_ind.i]]
                    else:
                        lanelet_list = [predictions["lane_ids"][midx][aidx][-1]]
                    lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_list[0])
                    if lanelet.adj_left:
                        lanelet_list.append(lanelet.adj_left)
                    if lanelet.adj_right:
                        lanelet_list.append(lanelet.adj_right)
                    state.set_posF(self.lanelet_network, lanelet_list)
                    #traj.append(state)
                    for tt in range(self.config["delta_step"]):
                        cstate = copy(state)
                        cstate.time_step = state.time_step-self.config["delta_step"]+tt+1
                        traj.append(cstate)
                    start_ind = next_ind
                    start_pos = next_pos
                if midx == smidx:
                    self.obstacles[self.time_step][vehid]._prediction = TrajectoryPrediction(
                            Trajectory(self.time_step, traj),
                            self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                        )


                if self.obstacles[self.time_step][vehid].initial_state.velocity < 2.0 and np.abs(traj_fre[-1, 1]-traj_fre[0, 1]) > 2.0:
                    continue
                #try:
                if self.config['pred_mode'] == "multi" or (self.config['pred_mode'] == "single" and midx == smidx):
                    self.obstacles_preds[self.time_step][vehid].append(
                                (TrajectoryPrediction(
                                    Trajectory(self.time_step, traj),
                                    self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                                ), scores[midx])
                            )
                #except:
                #    self.obstacles_preds[self.time_step][vehid].append(
                #            (TrajectoryPrediction(
                #                Trajectory(self.time_step, traj),
                #                self.vehicleinfos[self.traj_idx][1][11]["shape"]
                #            ), scores[midx])
                #        )

    def update_simple_motion_prediction(self):
        # constant velocity
        for vehid in self.obstacles[self.time_step].keys():
            if vehid in self.egoids_list:
                continue
            if vehid not in self.obstacles[self.time_step-self.config['delta_step']]:
                continue
            obstacle = self.obstacles[self.time_step][vehid]
            if vehid in self.obstacles_preds[self.time_step]:
                continue
            current_lanelet_id = obstacle.initial_state.posF.ind[1]
            #vs = []
            #for t in range(self.config['horizon_steps']//self.config['delta_step']):
            #    if vehid not in self.obstacles[self.time_step-t*self.config['delta_step']]:
            #        continue
            #    vs.append(self.obstacles[self.time_step-t*self.config['delta_step']][vehid].initial_state.velocity)
            current_velocity = obstacle.initial_state.velocity
            #current_velocity = np.mean(vs)
            max_disp = current_velocity * self.config['prediction_steps']*self.config['dt']+obstacle.initial_state.posF.s
            target_vertices = []
            lanelet_list = []
            current_disp = 0
            while current_disp < max_disp:
                if current_lanelet_id is None:
                    break
                current_lanelet = self.lanelet_network.find_lanelet_by_id(current_lanelet_id)
                current_disp += current_lanelet.center_curve[-2].s
                target_vertices.append(current_lanelet.center_vertices[:-1])
                lanelet_list.append(current_lanelet_id)
                if len(current_lanelet.successor) == 0:
                    current_lanelet_id = None
                elif len(current_lanelet.successor) == 1:
                    current_lanelet_id = current_lanelet.successor[0]
                else:
                    for successor_id in current_lanelet.successor:
                        success_lanelet = self.lanelet_network.find_lanelet_by_id(successor_id)
                        if len(success_lanelet.successor) > 0:
                            current_lanelet_id = successor_id
                            break
                    else:
                        current_lanelet_id = current_lanelet.successor[0]
            if len(target_vertices) == 0:
                continue
            target_vertices = np.concatenate(target_vertices, axis=0)
            target_curve = make_curve(target_vertices)
            start_ind = obstacle.initial_state.posF.ind[0]
            start_pos = obstacle.initial_state.posG
            traj = []
            for tidx in range(self.config["num_preds"]):
                if start_ind.i >= len(target_curve)-1:
                    break
                next_ind, next_pos = move_along_curve(start_ind, target_curve,
                                                      current_velocity*self.config['delta_step']*self.config['dt'],
                                                      obstacle.initial_state.posF.d)
                heading = np.arctan2(next_pos.y - start_pos.y, next_pos.x - start_pos.x)
                if tidx == 0:
                    initial_state = copy(self.obstacles[self.time_step][vehid].initial_state)
                    initial_state.acceleration = 0
                    initial_state.acceleration_y= 0
                    initial_state.yaw_rate = 0.0
                    initial_state.slip_angle = 0.0
                    traj.append(initial_state)

                state = FrenetState(
                    position = np.array([next_pos.x, next_pos.y]),
                    orientation=heading,
                    velocity=current_velocity,
                    acceleration=0,
                    acceleration_y=0,
                    time_step=self.time_step+(tidx+1)*self.config["delta_step"],
                    yaw_rate=0.0,
                    slip_angle=0.0
                )
                start_ind = next_ind
                start_pos = next_pos

                state.set_posF(self.lanelet_network, lanelet_list)
                for tt in range(self.config["delta_step"]):
                    cstate = copy(state)
                    cstate.time_step = state.time_step-self.config["delta_step"]+tt+1
                    traj.append(cstate)
            if len(traj)>0:
                self.obstacles[self.time_step][vehid]._prediction = TrajectoryPrediction(
                    Trajectory(self.time_step, traj),
                    self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                )
                self.obstacles_preds[self.time_step][vehid] = [(TrajectoryPrediction(
                    Trajectory(self.time_step, traj),
                    self.vehicleinfos[self.traj_idx][1][vehid]["shape"]
                ), 1.0)]


    def update_ego_vehicle(self, ego):
        obstacles = dict()
        #for vehid in ego.list_ids_neighbor:
        #    obstacles.append(self.obstacles[self.time_step][vehid])
        if self.motion_predictor is not None:
            obstacles_preds = dict()
        else:
            obstacles_preds = None
        for obs_id, ob in self.obstacles[self.time_step].items():
            if obs_id==ego.id:
                continue
            if obs_id in self.background_vehicles and self.background_vehicles[obs_id].front_vehid == ego.id:
                continue
            if obs_id in self.background_vehicles and self.background_vehicles[obs_id].merge_front_vehid == ego.id:
                continue
            if obs_id in self.ego_vehicles and self.ego_vehicles[obs_id].front_vehid == ego.id:
                continue
            if obs_id in self.ego_vehicles and self.ego_vehicles[obs_id].merge_front_vehid == ego.id:
                continue

            obstacles[obs_id] = ob
            if self.motion_predictor is not None and obs_id in self.obstacles_preds[self.time_step]:
                obstacles_preds[obs_id] = self.obstacles_preds[self.time_step][obs_id]

        if self.connected:
            state = ego.update_connected_state(obstacles, obstacles_preds)
        else:
            state = ego.update_state(obstacles, obstacles_preds)

        if state.posF.ind[1] in self.goal_lanelet_ids and state.posF.s > 25:
            ego.arrived = True
        else:
            shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[self.traj_idx][1][ego.id]["shape"].length,
                                          self.vehicleinfos[self.traj_idx][1][ego.id]["shape"].width,
                                          center=state.position,
                                          orientation=state.orientation))
            self.obstacles[self.time_step+1][ego.id] = DynamicObstacle(obstacle_id=ego.id,
                                                            obstacle_type=self.vehicleinfos[self.traj_idx][1][ego.id]["type"],
                                                            initial_state=state,
                                                            obstacle_shape=self.vehicleinfos[self.traj_idx][1][ego.id]["shape"],
                                                            initial_center_lanelet_ids=set([state.posF.ind[1]]),
                                                            initial_shape_lanelet_ids=set(shape_lanelet_ids))

    def update_vehicle(self, veh, acc):
        v = max(0.0, veh.v + acc*self.config["dt"])
        delta_s = max(0.0, veh.v*self.config["dt"] + 0.5*acc*self.config["dt"]**2)
        state = veh.update_state(delta_s, v, acc, self.time_step)
        if state is not None:
            assert state.time_step == self.time_step+1
            shape_lanelet_ids = self.lanelet_network.find_lanelet_by_shape(Rectangle(self.vehicleinfos[self.traj_idx][1][veh.id]["shape"].length,
                                          self.vehicleinfos[self.traj_idx][1][veh.id]["shape"].width,
                                          center=state.position,
                                          orientation=state.orientation))
            self.obstacles[self.time_step+1][veh.id] = DynamicObstacle(obstacle_id=veh.id,
                                                            obstacle_type=self.vehicleinfos[self.traj_idx][1][veh.id]["type"],
                                                            initial_state=state,
                                                            obstacle_shape=self.vehicleinfos[self.traj_idx][1][veh.id]["shape"],
                                                            initial_center_lanelet_ids=set([state.posF.ind[1]]),
                                                            initial_shape_lanelet_ids=set(shape_lanelet_ids))
    def lanelet_occupied(self, state, length):
        lanelet_id = state.posF.ind[1]
        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
        s0 = lerp_curve_with_ind(lanelet.center_curve, state.posF.ind[0])

        # reset start point
        '''
        if s0 > lanelet.center_curve[5].s:
            position = lanelet.center_vertices[5]
            state.position = position
            state.orientation = lanelet.center_curve[5].pos.th
            state.posG = VecSE2(state.position[0], state.position[1], state.orientation)
            state.set_posF(self.lanelet_network, clues=[lanelet_id])
            s0 = lanelet.center_curve[5].s
        '''
        occupanied = False
        for obs in self.obstacles[self.time_step].values():
            if obs.initial_state.posF.ind[1] == lanelet_id:
                s = lerp_curve_with_ind(lanelet.center_curve, obs.initial_state.posF.ind[0])
                if s0 > s: #Do not insert a vehicle in front
                    occupanied = True
                    return occupanied
                dstar = state.velocity*(obs.initial_state.velocity-state.velocity)/(2*np.sqrt(3.0*2.0))
                if np.abs(s-s0) < 0.5*length+0.5*obs.obstacle_shape.length + 2.5 + 1.5*state.velocity - dstar:
                    occupanied = True
                    return occupanied
        return occupanied

    def is_stopped(self, vehid, ts, ts_, te_):
        vels = [self.trajectories[self.traj_idx][t][vehid].velocity for t in range(ts_, te_+1)]
        stopped = False
        stop_steps = 0
        hist_stop_steps = 0

        if vels[ts-ts_] == 0.0:
            tt = ts+1
            while tt <= te_:
                if vels[tt-ts_]==0.0:
                    stop_steps += 1
                else:
                    break
                tt += 1
            if stop_steps >= 2/self.config["dt"]:
                stopped = True
            if not stopped:
                tt = ts-1
                while tt >= ts_:
                    if vels[tt-ts_]==0.0:
                        hist_stop_steps += 1
                        if stop_steps + hist_stop_steps >= 2/self.config["dt"]:
                            stopped = True
                            break
                    else:
                        break
                    tt -= 1
        return stopped, stop_steps

    def render(self, h=5, w=5, add_centroid=True, fill_lanelet=True, draw_scene=True, show_lane=False, show_annotation=False,
                    show_prediction=False, show_planned_trajectories=False):
        self.h, self.w = h, w
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(self.h,self.w))
        plot_limits = self.config["plot_limits"]
        self.ax.set_xlim(plot_limits[:2])
        self.ax.set_ylim(plot_limits[2:])
        ii = 0

        mains, surroundings = list(self.ego_vehicles.keys()), list(self.background_vehicles.keys())
        if draw_scene:
            scene, ego_vehs, back_vehs = self.expert_cr_scenario.commonroad_scenario_at_time_step(self.time_step, mains, surroundings)

            draw_object(scene, ax=self.ax, draw_params={'scenario':{'dynamic_obstacle':{'show_label': False, 'draw_shape': True,
                                                                                            'shape': {'facecolor':'#aaaaaa','edgecolor': '#aaaaaa'}}
                                                                                            ,
                                                                    'static_obstacle':{'show_label': True, 'draw_shape': True,
                                                                                        'shape': {'facecolor':'#808080','edgecolor': '#808080'}},
                                                                        'lanelet':{'fill_lanelet':fill_lanelet,
                                                                                    'draw_border_vertices':False}
                                                                        }
                                                            },
                             plot_limits = plot_limits)

            draw_object(ego_vehs, ax=self.ax, draw_params=draw_params_ego)
            draw_object(back_vehs, ax=self.ax, draw_params=draw_params_obstacle)

        if show_prediction:
            surr_pred_lines = []
            #for vehid in self.background_vehicles.keys():
            for vehid in self.obstacles[self.time_step]:
                if vehid in self.ego_vehicles:
                    continue
                if self.motion_predictor is not None:
                    if vehid not in self.obstacles_preds[self.time_step]:
                        continue
                    preds = self.obstacles_preds[self.time_step][vehid]
                    for pred, score in preds:
                        center_point_xs, center_point_ys = [], []
                        for t in range(self.config["delta_step"], self.config["prediction_steps"]+self.config["delta_step"], self.config["delta_step"]):
                            state = pred.trajectory.state_at_time_step(t+self.time_step)
                            if state:
                                center_point_xs.append(state.position[0])
                                center_point_ys.append(state.position[1])
                            else:
                                break
                        if len(center_point_xs) > 0:
                            surr_pred_lines.append(list(zip(center_point_xs, center_point_ys)))
                else:
                    if vehid not in self.background_vehicles.keys():
                        continue
                    obst = self.obstacles[self.time_step][vehid]
                    center_point_xs, center_point_ys = [], []

                    for t in range(self.config["delta_step"], self.config["prediction_steps"]+self.config["delta_step"], self.config["delta_step"]):
                        state = obst.state_at_time(t+self.time_step)
                        if state:
                            center_point_xs.append(state.position[0])
                            center_point_ys.append(state.position[1])
                        else:
                            break
                    if len(center_point_xs) > 0:
                        surr_pred_lines.append(list(zip(center_point_xs, center_point_ys)))
            surr_pred_traj = mc.LineCollection(surr_pred_lines, colors="g", linewidths=1.5, linestyle="--", zorder=31)
            self.ax.add_collection(surr_pred_traj)

        if show_planned_trajectories and not self.dummy:
            ego_planned_lines = []
            ego_all_planned_lines = []
            for veh in self.ego_vehicles.values():
                if veh.arrived:
                    continue
                center_point_xs, center_point_ys = [], []
                planned_states = veh._planned_trajectories[veh.current_time_step-self.config['delta_step']]
                for state in planned_states:
                    center_point_xs.append(state.position[0])
                    center_point_ys.append(state.position[1])
                ego_planned_lines.append(list(zip(center_point_xs, center_point_ys)))
                #print(len(veh._all_planned_trajectories[veh.current_time_step-self.config['delta_step']]))
                for states in veh._all_planned_trajectories[veh.current_time_step-self.config['delta_step']]:
                    center_point_xs, center_point_ys = [], []
                    for state in states:
                        center_point_xs.append(state.position[0])
                        center_point_ys.append(state.position[1])
                    ego_all_planned_lines.append(list(zip(center_point_xs, center_point_ys)))
            ego_planned_traj = mc.LineCollection(ego_planned_lines, colors="r", linewidths=1.5, linestyle="--", zorder=35)
            ego_all_planned_traj = mc.LineCollection(ego_all_planned_lines, colors="grey", linewidths=1.0, linestyle="--", zorder=31)
            #print(ego_all_planned_lines)
            self.ax.add_collection(ego_planned_traj)
            self.ax.add_collection(ego_all_planned_traj)

        if show_annotation:
            for vehid, veh in self.background_vehicles.items():
                if veh.arrived:
                    continue
                position = self.obstacles[self.time_step][vehid].initial_state.position
                text = "v:{:.2f}m/s\n".format(veh.v)
                if veh.front_vehid is not None:
                    text += "Fid:{} s:{:.2f}m v:{:.2f}m/s\n".format(veh.front_vehid, veh.front_s, veh.front_v)
                if veh.merge_front_vehid is not None:
                    text += "FMid:{} s:{:.2f}m v:{:.2f}m/s w:{:.2f}\n".format(veh.merge_front_vehid, veh.merge_front_s, veh.merge_front_v, veh.front_w)
                if veh.back_vehid is not None:
                    text += "Bid:{} s:{:.2f}m v:{:.2f}m/s w:{:.2f}\n".format(veh.back_vehid, veh.back_s, veh.back_v, veh.back_w)
                self.ax.annotate(text, xy=position, xytext=(5,-2), textcoords='offset pixels')

        p = 0
        #for vehid, veh in self.ego_vehicles.items():
        for vehid in self.egoids_list:
            veh = self.ego_vehicles[vehid]
            if veh.arrived:
                continue
            #position = self.obstacles[self.time_step][vehid].initial_state.position
            position = [22, -6 + p]
            if self.dummy:
                text = "Ego {}\n Speed:{:.2f}m/s   Acc:{:.2f}m/s\u00b2 \n Maneuver: None".format(vehid, veh.v, veh.a)
                self.ax.annotate(text, xy=position, fontsize=12)
                p -= 13
                continue
            if self.connected:
                if veh.action is None:
                    maneuver = "None"
                else:
                    maneuver = ""
                    dind = np.argmax(veh.action[1])
                    maneuver = "{}: {:.1f}% ".format(veh.action[2][dind], veh.action[1][dind]*100)

                    #for dist, name in zip(veh.action[1], veh.action[2]):
                    #    maneuver += "{}: {:.1f}% ".format(name, dist*100)
            else:
                if veh.action is None:
                    maneuver = "None"
                elif veh.action[0] == 0: # follow
                    maneuver = "Follow"
                elif veh.action[0] == 1:
                    maneuver = "Keep speed at {:.2f}m/s".format(veh.action[3])
                elif veh.action[0] == 2:
                    maneuver = "Stop"
                else:
                    maneuver = "Merge"
            text = "Ego {}\n Speed:{:.2f}m/s   Acc:{:.2f}m/s\u00b2 \n Maneuver: {}".format(vehid, veh.current_sv, veh.current_sa, maneuver)
            self.ax.annotate(text, xy=position, fontsize=12)
            p -= 13


        major_ticks_x = np.arange(20, 141, 10)
        major_ticks_y = np.arange(-90, 11, 10)

        self.ax.set_xticks(major_ticks_x)
        self.ax.set_yticks(major_ticks_y)
        self.ax.grid(which='major', linestyle=":", zorder=28.0)
        self.ax.set_aspect('equal')
        for tickx in self.ax.xaxis.get_major_ticks():
            tickx.tick1line.set_visible(False)
            tickx.tick2line.set_visible(False)
            tickx.label1.set_visible(False)
            tickx.label2.set_visible(False)
        for ticky in self.ax.yaxis.get_major_ticks():
            ticky.tick1line.set_visible(False)
            ticky.tick2line.set_visible(False)
            ticky.label1.set_visible(False)
            ticky.label2.set_visible(False)
