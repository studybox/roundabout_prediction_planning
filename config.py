Config = {"datasets":["rounD", "rounD-plus", "mcity", "CHN", "DEU", "EP", "SR", "FT"],
          "all":{"data_dir":["./data/rounD", "./data/rounD-plus", "./data/mcity", "./data/interaction-CHN",
                              "./data/interaction-DEU", "./data/interaction-USASR", "./data/interaction-USAFT",
                              "./data/interaction-USAEP"],
                  "model_dir":"./model/all",
                  #"loaded_model":"PathGH-train_lr-0.001_decay-0.9999_L2-0.0_bs-32-path/checkpoint_50000.pt",
                  #"loaded_model":"LaneGCN-train_lr-0.001_decay-0.9999_L2-0.0_bs-32-/checkpoint_44000.pt",
                  #"loaded_model":"FrenetLaneGCN-train_lr-1e-05_decay-0.9999_L2-0.0_bs-32-/checkpoint_20000.pt",
                  "loaded_model":"FrenetPathGCN-train_lr-0.001_decay-0.9999_L2-0.0_bs-32-/checkpoint_41000.pt",
                  "input_dims": (10, 3),#717,
                  "output_dims": (15, 2),
                  "output_dim":2,
                  "linear_input_dim":108,
                  "linear_output_dim":2,
                  "num_components": 1,
                  "z_dim":2,

                  "route_class_size":8,
                  "route_category_size":8,
                  "node_class_size":4,
                  "node_category_size":4,
                  "act_class_size":4,
                  "act_category_size":4,
                  "int_class_size":4,
                  "int_category_size":4,

                  "int_continuous_size":2,
                  "act_continuous_size":2,
                  "pat_continuous_size":4,

                  "linear_h_dims": {"hidden_dim1":32, "hidden_dim2":64, "hidden_dim3":128, "hidden_dim4":32},
                  "n_actor":128,#64
                  "n_path":128,
                  "n_map":128,#64
                  "n_head":1,
                  "num_blocks":1,
                  "map_feat_size":4,
                  "actor2map_dist":7.0,
                  "map2actor_dist":6.0,
                  "actor2actor_dist":80,

                  "dT":0.2,
                  "num_paths":5,
                  "num_preds":15,
                  "num_mods":1,
                  "cls_coef":1.0,
                  "reg_coef":1.0,
                  "mgn":0.2,
                  "cls_th":2.0,
                  "cls_ignore":0.2,
                  "sample_mode":"ucb",

                  "skip_steps" : 25,
                  "cross_dist": 6.0,
                  "num_scales": 6,

                  "trajectory_model":"LSTM"
                  },

          "rounD":{"data_dir":"./data/rounD",
                   "model_dir":"./model",
                   "loaded_model":"../motion_prediction/model/all/FrenetPathMultiTargetGCN-train_lr-0.001_decay-0.9999_L2-0.0_bs-32-/checkpoint_52000.pt",

                   "dt":0.04,
                   "delta_step":5,
                   "sim_delta_step":5,

                   "horizon_steps": 50,
                   "prediction_steps": 75,
                   "dropout": 0,

                   "grid_length": 5.0,
                   "max_grid_disp_front" :100.,
                   "max_grid_disp_rear" : 50.,
                   "max_grid_radius" : 100.,
                   "max_veh_disp_front" : 80.,
                   "max_veh_disp_rear" : 20.,
                   "max_veh_disp_side" : 50.,
                   "max_veh_radius" : 80.,
                   "num_scales": 6,
                   "veh_along_lane" : False,

                  "n_actor":128,#64
                  "n_path":128,
                  "n_map":128,#64
                  "n_head":1,
                  "num_blocks":1,
                  "map_feat_size":4,
                  "actor2map_dist":7.0,
                  "map2actor_dist":6.0,
                  "actor2actor_dist":80,

                  #"num_paths":5,
                  "num_preds":15,
                  "num_mods":6,
                  #"cls_coef":1.0,
                  #"reg_coef":1.0,
                  #"mgn":0.2,
                  #"cls_th":2.0,
                  #"cls_ignore":0.2,
                  "sample_mode":"bias", #"ucb",#"bias",
                  'pred_mode' : "multi",

                  "front_s_max": 40,
                  "mini_gap": 2.5,
                  "time_head": 1.2,
                   "plot_limits": [35, 135, -85, -5],
                   "dh" : 50,
                   "dw" : 50,
                   "cross_dist": 6.0,
                   }
}

basic_shape_parameters_nei = {'opacity': 1.0,
                               'facecolor': '#2469ff',
                               'edgecolor': '#2469ff',
                               'linewidth': 0.5,
                               'zorder': 20}

basic_shape_parameters_ego = {'opacity': 1.0,
                               'facecolor': '#c30000',
                               'edgecolor': '#c30000',
                               'linewidth': 0.5,
                               'zorder': 20}

basic_shape_parameters_mlego = {'opacity': 1.0,
                               'facecolor': '#09b60c',
                               'edgecolor': '#09b60c',
                               'linewidth': 0.5,
                               'zorder': 20}


basic_shape_parameters_obs = {'opacity': 1.0,
                               'facecolor': '#aaaaaa',
                               'edgecolor': '#aaaaaa',
                               'zorder': 20}

draw_params_neighbor = {'dynamic_obstacle': {
                            'draw_shape': True,
                            'draw_icon': False,
                            'draw_bounding_box': True,
                            'show_label': True,
                            'trajectory_steps': 25,
                            'zorder': 20,
                            'shape': basic_shape_parameters_nei,
                            'occupancy': {
                                'draw_occupancies': 1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
                                'shape': {
                                    'opacity': 0.25,
                                       'facecolor': '#6f9bcb',
                                       'edgecolor': '#48617b',
                                       'zorder': 18,
                                }
                            }
                        }}


draw_params_obstacle = {'dynamic_obstacle': {
                            'draw_shape': True,
                            'draw_icon': False,
                            'draw_bounding_box': True,
                            'show_label': True,
                            'trajectory_steps': 25,
                            'zorder': 20,
                            'shape': basic_shape_parameters_mlego,
                            'occupancy': {
                                'draw_occupancies': 1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
                                'shape': {
                                    'opacity': 0.25,
                                       'facecolor': '#000000',
                                       'edgecolor': '#000000',
                                       'zorder': 20,
                                }
                            }
                        }}

draw_params_ego = {'dynamic_obstacle': {
                            'draw_shape': True,
                            'draw_icon': False,
                            'draw_bounding_box': True,
                            'show_label': True,
                            'trajectory_steps': 0,
                            'zorder': 20,
                            'shape': basic_shape_parameters_ego,
                            'occupancy': {
                                'draw_occupancies': 1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
                                'shape': {
                                    'opacity': 0.25,
                                    'facecolor': '#b05559',
                                    'edgecolor': '#9e4d4e',
                                    'zorder': 20,
                                }
                            }
                    }}


draw_params_mlego = {'dynamic_obstacle': {
                            'draw_shape': True,
                            'draw_icon': False,
                            'draw_bounding_box': True,
                            'show_label': True,
                            'trajectory_steps': 0,
                            'zorder': 20,
                            'shape': basic_shape_parameters_mlego,
                            'occupancy': {
                                'draw_occupancies': 1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
                                'shape': {
                                    'opacity': 0.25,
                                    'facecolor': 'green',
                                    'edgecolor': 'green',
                                    'zorder': 20,
                                }
                            }
                    }}
