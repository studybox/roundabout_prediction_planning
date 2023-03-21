import numpy as np
import os
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet
from typing import List
from ctypes import c_double, c_float, c_int, c_size_t, POINTER, Structure, CDLL, byref

from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle
cdll = CDLL("./frenet_planning/build/libfrenet_planning.so")
MAX_TRAJ_NUM = 2500
MAX_PATH_LENGTH = 155

_c_double_p = POINTER(c_double)
_c_float_p = POINTER(c_float)
_c_int_p = POINTER(c_int)


class FrenetConditions(Structure):
    _fields_ = [
        ("c_x", c_float),                       # current position in x direction
        ("c_vx", c_float),                      # current speed in x direction
        ("c_ax", c_float),                      # current acceleration in x direction
        ("c_y", c_float),
        ("c_vy", c_float),
        ("c_ay", c_float),
        ("c_heading", c_float),
        ("t", _c_float_p),
        ("t_s", _c_float_p),
        ("t_s_d", _c_float_p),
        ("t_s_dd", _c_float_p),
        ("t_d", _c_float_p),
        ("t_d_d", _c_float_p),
        ("t_d_dd", _c_float_p),
        ("nt", c_int),
    ]

class FrenetParameters(Structure):
    _fields_ = [
        ("max_speed", c_float),             # maximum speed [m/s]
        ("max_accel", c_float),             # maximum acceleration [m/s^2]
        ("max_curvature", c_float),         # maximum curvature [1/m]
        ("keep_velocity", c_int),           # keep velocity targets
        #("max_road_width_l", c_double),      # maximum road width to the left [m]
        #("max_road_width_r", c_double),      # maximum road width to the right [m]
        #("d_road_w", c_double),              # road width sampling discretization [m]
        #("dt", c_double),                    # time sampling discretization [s]
        #("maxt", c_double),                  # max prediction horizon [s]
        #("mint", c_double),                  # min prediction horizon [s]
        ("d_s", _c_float_p),                 # target speed sampling discretization [m/s]
        ("n_sample_s", c_int),            # sampling number of target speed
        ("d_d", _c_float_p),                 # target speed sampling discretization [m/s]
        ("n_sample_d", c_int),            # sampling number of target speed
        ("smin", c_float),
        ("smax", c_float),
        ("dmin", c_float),
        ("dmax", c_float),

        #("obstacle_clearance", c_double),    # obstacle radius [m]
        ("ks", c_float),                    # long deviation cost
        ("kd", c_float),                    # lat deviation cost
        ("kj", c_float),                    # jerk cost
        ("kt", c_float),                    # time deviation cost
        ("ko", c_float),                    # dist to obstacle cost
        ("klat", c_float),                  # lateral cost
        ("klon", c_float),                  # longitudinal cost
        ("length", c_float),                # vehicle length
        ("width", c_float),                 # vehicle width
        ("num_threads", c_int)               # number of threads
    ]

class FrenetContexts(Structure):
    _fields_ = [
        ("wx", _c_float_p),     # waypoints x position
        ("wy", _c_float_p),     # waypoints y position
        ("wlw", _c_float_p),    # waypoints left width
        ("wrw", _c_float_p),    # waypoints right width
        ("nw", c_int),           # number of waypoints
        ("o_p1x", _c_double_p),  # obstacles point1 x
        ("o_p1y", _c_double_p),  # obstacles point1 y
        ("o_p2x", _c_double_p),  # obstacles point2 x
        ("o_p2y", _c_double_p),  # obstacles point2 y
        ("o_p3x", _c_double_p),  # obstacles point3 x
        ("o_p3y", _c_double_p),  # obstacles point3 y
        ("o_p4x", _c_double_p),  # obstacles point4 x
        ("o_p4y", _c_double_p),  # obstacles point4 y
        ("o_t", _c_double_p),    # obstacles time
        ("o_s", _c_double_p),    # obstacles score
        ("o_ind", _c_int_p),     # obstacles index
        ("no", c_int)            # number*step of obstacles
    ]

class FrenetTraj(Structure):
    _fields_ = [
        ("feasible", c_int),                     #whether the trajectory is feasible
        ("path_length", c_size_t),               #length of trajectory

        ("final_t", c_float),
        ("final_s", c_float),
        ("final_sv", c_float),
        ("final_sa", c_float),

        ("final_d", c_float),
        ("final_dv", c_float),
        ("final_da", c_float),

        ("cx", c_double * 6),
        ("cy", c_double * 6),

        ("t", c_float * MAX_PATH_LENGTH),
        ("x", c_float * MAX_PATH_LENGTH),     # x positions of fot, if it exists
        ("y", c_float * MAX_PATH_LENGTH),     # y positions of fot, if it exists
        ("s",  c_float * MAX_PATH_LENGTH),         # longitudinal offset of fot, if it exists
        ("d",  c_float * MAX_PATH_LENGTH),         # lateral offset of fot, if it exists
        ("speed_s",  c_float * MAX_PATH_LENGTH),  # x speeds of fot, if it exists
        ("speed_d", c_float * MAX_PATH_LENGTH),   # y speeds of fot, if it exists
        ("heading",  c_float * MAX_PATH_LENGTH),  # x speeds of fot, if it exists
        ("slipangle", c_float * MAX_PATH_LENGTH),   # y speeds of fot, if it exists
        ("acc_s",  c_float * MAX_PATH_LENGTH),  # x speeds of fot, if it exists
        ("acc_d", c_float * MAX_PATH_LENGTH),   # y speeds of fot, if it exists
        ("jerk", c_float * MAX_PATH_LENGTH),   # jerk  of fot, if it exists
        ("cost", c_float)       # cost of frenet path, if it exists
    ]

class FrenetReturns(Structure):
    _fields_ = [
        ("success", c_int),                         # whether a fot was found or not
        ("num_traj", c_size_t),                     # number of trajectories
        ("trajectories", FrenetTraj*MAX_TRAJ_NUM),           # all trajectories
        ("best_trajectory", FrenetTraj*1)             # best trajectory
        ]
'''
class FrenetReturns(Structure):
    _fields_ = [
        ("success", c_int),                         # whether a fot was found or not
        ("path_length", c_size_t),                  # length of path
        ("t", c_float * MAX_PATH_LENGTH),
        ("x", c_float * MAX_PATH_LENGTH),     # x positions of fot, if it exists
        ("y", c_float * MAX_PATH_LENGTH),     # y positions of fot, if it exists
        ("ix", c_float * MAX_PATH_LENGTH),
        ("iy", c_float * MAX_PATH_LENGTH),
        ("s",  c_float * MAX_PATH_LENGTH),         # longitudinal offset of fot, if it exists
        ("d",  c_float * MAX_PATH_LENGTH),         # lateral offset of fot, if it exists
        ("speed_s",  c_float * MAX_PATH_LENGTH),  # x speeds of fot, if it exists
        ("speed_d", c_float * MAX_PATH_LENGTH),   # y speeds of fot, if it exists
        ("acc_s",  c_float * MAX_PATH_LENGTH),  # x speeds of fot, if it exists
        ("acc_d", c_float * MAX_PATH_LENGTH),   # y speeds of fot, if it exists
        ("heading",  c_float * MAX_PATH_LENGTH),  # x speeds of fot, if it exists
        ("slipangle", c_float * MAX_PATH_LENGTH),   # y speeds of fot, if it exists
        ("params", c_float * MAX_PATH_LENGTH),     # next frenet coordinates, if they exist
        ("costs", c_float * MAX_PATH_LENGTH)       # costs of best frenet path, if it exists

    ]

'''

_run_fot = cdll.run
_run_fot.argtypes = (
    POINTER(FrenetConditions),
    POINTER(FrenetParameters),
    POINTER(FrenetContexts),
    POINTER(FrenetReturns),
)
_run_fot.restype = None



_run_dist = cdll.run_dist
_run_dist.argtypes = (
    c_double, c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double
)
_run_dist.restype = c_double



def run_dist(x,y,h,l,w,p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y):
    return _run_dist(c_double(x), c_double(y), c_double(h), c_double(l), c_double(w), \
                     c_double(p1x), c_double(p1y), c_double(p2x), c_double(p2y), \
                        c_double(p3x), c_double(p3y), c_double(p4x), c_double(p4y))

def opt(state, action, obstacles, route_segment, future_steps = [0]):

    x0, y0, heading0, vs0, as0, vd0, ad0, time_step, veh_length, veh_width = state
    T = action["target_T"].astype(np.float32)
    ts = action["target_s"].astype(np.float32)
    ts_d = action["target_v"].astype(np.float32)
    ts_dd = action["target_a"].astype(np.float32)
    ds = action["offset_s"].astype(np.float32)

    td = action["target_d"].astype(np.float32)
    dd = action["offset_d"].astype(np.float32)
    keep_velocity = action["keep_velocity"]
    td_d = np.zeros(len(T)).astype(np.float32)
    td_dd = np.zeros(len(T)).astype(np.float32)


    fc = FrenetConditions(
            c_float(x0), c_float(vs0), c_float(as0),
            c_float(y0), c_float(vd0), c_float(ad0), c_float(heading0),
            T.ctypes.data_as(_c_float_p),
            ts.ctypes.data_as(_c_float_p), ts_d.ctypes.data_as(_c_float_p), ts_dd.ctypes.data_as(_c_float_p),
            td.ctypes.data_as(_c_float_p), td_d.ctypes.data_as(_c_float_p), td_dd.ctypes.data_as(_c_float_p), len(td_dd)
        )

    #route_segment = route.reference_path_segments[route.principle_reference.index(1)]
    wx = route_segment[:,0].astype(np.float32)
    wy = route_segment[:,1].astype(np.float32)
    wlw = route_segment[:,3].astype(np.float32)
    wrw = route_segment[:,4].astype(np.float32)
    if len(wx) < 10:
        dwx = wx[-1] - wx[-2]
        dwy = wy[-1] - wy[-2]
        lw = wlw[-1]
        rw = wrw[-1]
        wx = np.concatenate([wx, np.arange(1, 11-len(wx))*dwx + wx[-1]]).astype(np.float32)
        wy = np.concatenate([wy, np.arange(1, 11-len(wy))*dwy + wy[-1]]).astype(np.float32)
        wlw = np.concatenate([wlx, np.ones(10-len(wlw))*lw]).astype(np.float32)
        wrw = np.concatenate([wrx, np.ones(10-len(wrw))*rw]).astype(np.float32)
        #print(wx, wy, wlw, wrw)
        #raise
    #print(wx[:10])
    #print(wy[:10])
    #print(wlw[:10])
    #print(wrw[:10])
    oinds, op1x, op2x, op3x, op4x, op1y, op2y, op3y, op4y, ot, osc = [], [], [], [], [], [], [], [], [], [], []
    ind = 0
    #print("t", time_step)
    for obs in obstacles:
        if isinstance(obs, List):
            #print(len(obs))
            for pred, score in obs:
                for t in future_steps:
                    occupancy = pred.occupancy_at_time_step(time_step+t)
                    if occupancy is None:
                        break
                    v = occupancy.shape.vertices
                    oinds.append(ind)
                    op1x.append(v[0,0])
                    op1y.append(v[0,1])
                    op2x.append(v[1,0])
                    op2y.append(v[1,1])
                    op3x.append(v[2,0])
                    op3y.append(v[2,1])
                    op4x.append(v[3,0])
                    op4y.append(v[3,1])
                    ot.append(t*0.04)
                    osc.append(score)
                    #osc.append(1.0)

                ind += 1

        else:
            for t in future_steps:
                occupancy = obs.occupancy_at_time(obs.initial_state.time_step+t)
                if isinstance(obs, DynamicObstacle):
                    assert time_step == obs.initial_state.time_step, obs.obstacle_id
                if occupancy is None:
                    break
                v = occupancy.shape.vertices
                oinds.append(ind)
                op1x.append(v[0,0])
                op1y.append(v[0,1])
                op2x.append(v[1,0])
                op2y.append(v[1,1])
                op3x.append(v[2,0])
                op3y.append(v[2,1])
                op4x.append(v[3,0])
                op4y.append(v[3,1])
                ot.append(t*0.04)
                osc.append(1.0)
            ind += 1
    '''
    for ind, obs in enumerate(obstacles):
        for t in future_steps:
            occupancy = obs.occupancy_at_time(obs.initial_state.time_step+t)
            if occupancy is None:
                break
            v = occupancy.shape.vertices
            oinds.append(ind)
            op1x.append(v[0,0])
            op1y.append(v[0,1])
            op2x.append(v[1,0])
            op2y.append(v[1,1])
            op3x.append(v[2,0])
            op3y.append(v[2,1])
            op4x.append(v[3,0])
            op4y.append(v[3,1])
            ot.append(t*0.04)
    '''
    op1x = np.array(op1x).astype(np.float64)
    op1y = np.array(op1y).astype(np.float64)
    op2x =np.array(op2x).astype(np.float64)
    op2y = np.array(op2y).astype(np.float64)
    op3x = np.array(op3x).astype(np.float64)
    op3y = np.array(op3y).astype(np.float64)
    op4x = np.array(op4x).astype(np.float64)
    op4y = np.array(op4y).astype(np.float64)
    ot = np.array(ot).astype(np.float64)
    osc = np.array(osc).astype(np.float64)
    oinds = np.array(oinds).astype(np.int32)
    #print(op1x, ot)
    fct = FrenetContexts(
            wx.ctypes.data_as(_c_float_p),
            wy.ctypes.data_as(_c_float_p),
            wlw.ctypes.data_as(_c_float_p),
            wrw.ctypes.data_as(_c_float_p),
            len(wx),
            op1x.ctypes.data_as(_c_double_p),
            op1y.ctypes.data_as(_c_double_p),
            op2x.ctypes.data_as(_c_double_p),
            op2y.ctypes.data_as(_c_double_p),
            op3x.ctypes.data_as(_c_double_p),
            op3y.ctypes.data_as(_c_double_p),
            op4x.ctypes.data_as(_c_double_p),
            op4y.ctypes.data_as(_c_double_p),
            ot.ctypes.data_as(_c_double_p),
            osc.ctypes.data_as(_c_double_p),
            oinds.ctypes.data_as(_c_int_p),
            len(oinds)
        )
    fp = FrenetParameters(
            c_float(30),                                         # maximum speed [m/s]
            c_float(4.5),                                        # maximum acceleration [m/s^2]
            c_float(6),                                          # maximum curvature [1/m]
            c_int(keep_velocity),                                # keep velocity targets
            ds.astype(np.float32).ctypes.data_as(_c_float_p),    # target speed sampling discretization [m/s]
            c_int(len(ds)),                                      # sampling number of target speed
            dd.astype(np.float32).ctypes.data_as(_c_float_p),    # target lateral sampling discretization rate [-]
            c_int(len(dd)),                                      # sampling number of target lateral
            c_float(0),                                          # smin
            c_float(80),                                         # smax
            c_float(-5),                                         # dmin
            c_float(5),                                          # dmax
            c_float(2.0), # ks 1.0
            c_float(1.0), # kd
            c_float(0.5), # kj
            c_float(1.0), # kt
            c_float(10.0), # ko
            c_float(2.0), # klat
            c_float(1.0), # klon
            c_float(veh_length), #+2.0
            c_float(veh_width), #+0.5
            1,)
    fr = FrenetReturns(0)
    _run_fot(fc, fp, fct, fr)
    '''
    planned_trajectory = []
    for i in range(1, fr.path_length):
        next_state =FrenetState(position = np.array([fr.x[i], fr.y[i]]),
                           orientation=fr.heading[i],
                           velocity=max(0.0, fr.speed_s[i]),
                           velocity_y=fr.speed_d[i],
                           acceleration=fr.acc_s[i],
                           acceleration_y=fr.acc_d[i],
                           time_step=time_step+i,
                           yaw_rate=0.0,
                           slip_angle=fr.slipangle[i])
        phi = np.arctan2(fr.d[i]-fr.d[i-1], fr.s[i]-fr.s[i-1])
        #next_state.posF = Frenet(tuple_form=(0,0,-1,fr.s[i],fr.d[i],phi))
        planned_trajectory.append(next_state)
    '''
    return fr
