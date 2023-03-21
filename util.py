import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data, Batch

from commonroad.scenario.laneletcomplement import *
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet,  move_along_curve
from commonroad.common.file_reader_complement import LaneletCurveNetworkReader
from commonroad.scenario.laneletcomplement import make_lanelet_curve_network
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LineMarking, LaneletType

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TypeVar, Union, List
from torch.utils.data import DataLoader
import dataclasses
import pickle
from collections import defaultdict
from scipy.sparse import csr_matrix
import scipy
from scipy import sparse
import numpy as np

def mix_dataset(Config):
    train_files, val_files = [], []

    data_root_dir = Config["rounD"]["data_dir"]
    data_files = [osp.join(data_root_dir, "processed2", d) for d in  os.listdir(osp.join(data_root_dir, "processed2")) if 'data' in d]
    tf, vf = train_test_split(data_files, test_size=0.2, random_state=42)
    train_files.extend(tf)
    val_files.extend(vf)
    return train_files, val_files


def load_ngsim_scenarios(lanelet_network_filepaths, trajectory_filepaths):

    lanelet_networks = {}
    # load lanelet_networks
    for fp in lanelet_network_filepaths:
        if "i80" in fp:
            lanelet_networks["i80"] = LaneletCurveNetworkReader(fp).lanelet_network
        elif "u101" in fp:
            u101_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:39.89 for ln in u101_lanelet_network.lanelets}
            lanelet_networks["u101"] = make_lanelet_curve_network(u101_lanelet_network, speed_limits)
        elif "highD2" in fp:
            lanelet_networks["highD2"] = LaneletCurveNetworkReader(fp).lanelet_network
        elif "highD3" in fp:
            lanelet_networks["highD3"] = LaneletCurveNetworkReader(fp).lanelet_network
        elif "rounD-plus" in fp:
            rounD_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in rounD_lanelet_network.lanelets}
            if "00" in fp:
                lanelet_networks["rounD-plus00"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
            else:
                lanelet_networks["rounD-plus01"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
        elif "rounD" in fp:
            rounD_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in rounD_lanelet_network.lanelets}
            lanelet_networks["rounD"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
        elif "CHN" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["CHN"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "DEU" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["DEU"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "SR" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["SR"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "EP" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["EP"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "FT" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["FT"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "mcity" in fp:
            mcity_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in mcity_lanelet_network.lanelets}
            lanelet_networks["mcity"] = make_lanelet_curve_network(mcity_lanelet_network, speed_limits)
        else:
            raise ValueError("Can not identify lanelet_network in {}".format(fp))

    trajectories = []
    vehicleinfo = []
    # load trajectories
    for fp in trajectory_filepaths:
        trajdata = pickle.load(open(fp, "rb"))

        obstacle_infos = {}
        obstacle_states = defaultdict(lambda:dict())
        for d in trajdata['def']:
            carid, length, width, tp, f_lo, f_hi = d
            obstacle_infos[int(carid)] = {"shape":Rectangle(length,width),
                                        "type":tp, "frames":(int(f_lo),int(f_hi))}
        for d in trajdata['state']:
            step, carid, x, y, ori, v, i, t, lid, s, d, phi = d
            state = FrenetState(position=np.array([x,y]), orientation=ori, velocity=v, time_step = int(step))
            posF = Frenet(None, None, (i, t, lid, s, d, phi))
            state.posF = posF
            obstacle_states[int(step)][int(carid)] = state

        trajectories.append(obstacle_states)
        if "i80" in fp:
            lanelane_network_id = "i80"
        elif "u101" in fp:
            lanelane_network_id = "u101"
        elif "highD2" in fp:
            lanelane_network_id = "highD2"
        elif "highD3" in fp:
            lanelane_network_id = "highD3"
        elif "rounD-plus" in fp:
            if "00" in fp:
                lanelane_network_id = "rounD-plus00"
            else:
                lanelane_network_id = "rounD-plus01"
        elif "rounD" in fp:
            lanelane_network_id = "rounD"
        elif "mcity" in fp:
            lanelane_network_id = "mcity"
        elif "CHN" in fp:
            lanelane_network_id = "CHN"
        elif "DEU" in fp:
            lanelane_network_id = "DEU"
        elif "SR" in fp:
            lanelane_network_id = "SR"
        elif "EP" in fp:
            lanelane_network_id = "EP"
        elif "FT" in fp:
            lanelane_network_id = "FT"
        else:
            raise ValueError("Can not identify trajectory in {}".format(fp))

        vehicleinfo.append((lanelane_network_id, obstacle_infos))

    return trajectories, vehicleinfo, lanelet_networks

def relative_to_curve(traj_fre, last_pos, vertices):
    # traj_fre #(S, A, 2)
    last_pos = last_pos.cpu().numpy()

    traj_pred_rel = []
    for idx in range(traj_fre.size(1)):
        target_curve = make_curve(vertices[idx].cpu().numpy())
        start_pos = VecSE2(last_pos[idx, 0], last_pos[idx, 1], 0.0)
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
        traj = []
        for t in range(traj_fre.size(0)):
            next_ind, next_pos = move_along_curve(start_ind, target_curve, traj_fre[t, idx, 0].cpu().item(), traj_fre[t, idx, 1].cpu().item())
            traj.append([next_pos.x-start_pos.x, next_pos.y-start_pos.y])
            start_ind = next_ind
            start_pos = next_pos
        traj_pred_rel.append(traj)

    return torch.tensor(traj_pred_rel, dtype=torch.float32).cuda().permute(1, 0, 2)

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def check_accuracy(dataloader, config, model):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, f_disp_error, miss_rate = [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)]

    total_final_preds, total_preds = [0 for _ in range(3)], [0 for _ in range(3)]

    best_disp_error, best_f_disp_error = [[] for _ in range(3)], [[] for _ in range(3)]


    K_total_preds, K_total_final_preds = [0 for _ in range(3)], [0 for _ in range(3)]
    K_disp_error, K_f_disp_error, K_miss_rate =  [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)]
    diversities = 0.0
    diversities_N = 0
    path_acc, target_errors, slc_acc = [], [], []

    loss_mask_sum = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ret = model(batch)
            if "frenet_pred" in ret:
                frenet_pred = ret["frenet_pred"]
                curves = ret["curves"]
                curves_gt = ret["curves_gt"]
                if "converted_pred" in ret:
                    pred_traj_fake_rel = ret["converted_pred"]
                else:
                    pred_traj_fake_rel = relative_to_curve(frenet_pred, batch.obs_traj[-1], curves)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, batch.obs_traj[-1])
                if "converted_gt" in ret:
                    pred_traj_gt_rel = ret["converted_gt"]
                else:
                    pred_traj_gt_rel = relative_to_curve(batch.fut_traj_fre, batch.obs_traj[-1], curves_gt)
                pred_traj_gt = relative_to_abs(pred_traj_gt_rel, batch.obs_traj[-1])

                g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                    pred_traj_gt, pred_traj_gt_rel, batch.has_preds, pred_traj_fake, pred_traj_fake_rel
                )
                g_l2_losses_abs.append(g_l2_loss_abs.item())
                g_l2_losses_rel.append(g_l2_loss_rel.item())


                for tt in range(3):
                    ade = cal_ade(
                        pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
                        pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    fde = cal_fde(
                        pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
                        pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    mr = cal_mr(
                        pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
                        pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    disp_error[tt].append(ade.item())
                    f_disp_error[tt].append(fde.item())
                    miss_rate[tt].append(mr.item())

                    total_preds[tt] += batch.has_preds[:config["num_preds"]//3*(tt+1)-1].sum()
                    total_final_preds[tt] += batch.has_preds[config["num_preds"]//3*(tt+1)-1].sum()



            if "pred" in ret:
                pred_traj_fake_rel = ret['pred']
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, batch.obs_traj[-1])

                g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                    batch.fut_traj, batch.fut_traj_rel, batch.has_preds, pred_traj_fake, pred_traj_fake_rel
                )
                g_l2_losses_abs.append(g_l2_loss_abs.item())
                g_l2_losses_rel.append(g_l2_loss_rel.item())

                for tt in range(3):
                    ade = cal_ade(
                        batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
                        pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    fde = cal_fde(
                        batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
                        pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    mr = cal_mr(
                        batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
                        pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    disp_error[tt].append(ade.item())
                    f_disp_error[tt].append(fde.item())
                    miss_rate[tt].append(mr.item())
                    total_preds[tt] += batch.has_preds[:config["num_preds"]//3*(tt+1)-1].sum()
                    total_final_preds[tt] += batch.has_preds[config["num_preds"]//3*(tt+1)-1].sum()

            if "best" in ret:
                best_pred_traj_fake_rel = ret['best']
                best_pred_traj_fake = relative_to_abs(best_pred_traj_fake_rel, batch.obs_traj[-1])

                for tt in range(3):
                    best_ade = cal_ade(
                        batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
                        best_pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    best_fde = cal_fde(
                        batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
                        best_pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
                    )
                    best_disp_error[tt].append(best_ade.item())
                    best_f_disp_error[tt].append(best_fde.item())

            if "path" in ret:
                path = ret["path"]
                acc_score = cal_acc(ret["path"][batch.has_preds[-1]], ret["gt_path"][batch.has_preds[-1]])
                path_acc.append(acc_score)

            if "target" in ret:

                #path = ret["path"]
                #print("has ", batch.has_preds[-1].sum())
                #acc_score = cal_min_acc(path[batch.has_preds[-1]], batch.fut_path[batch.has_preds[-1]])
                #path_acc.append(acc_score)

                #target_pred = torch.stack([ret["target"][:, 0]*50,
                #                      ret["target"][:, 1]*13.5-5.5], dim=-1)
                target_pred = torch.stack([ret["target"][:, 0],
                                      ret["target"][:, 1]], dim=-1)


                target_gt = batch.fut_target[:, [0,2]]
                target_error = target_gt[batch.has_preds[-1]] - target_pred[batch.has_preds[-1]]
                target_error = target_error**2
                target_error = torch.sqrt(target_error.sum(dim=-1)).sum()
                target_errors.append(target_error.item())
                #acc_score = cal_min_acc(target[batch.has_preds[-1]], batch.fut_target[batch.has_preds[-1]])
                #target_acc.append(acc_score)

            #if "score" in ret and "target" in ret:
            #    slc = ret['score']
            #    target = ret["target"]
            #    _, slc_idcs = slc.max(1)
            #    row_idcs = torch.arange(len(slc_idcs)).long().to(slc_idcs.device)

            #    slc_score = cal_acc(target[row_idcs, slc_idcs][batch.has_preds[-1]], batch.fut_target[batch.has_preds[-1]])
            #    slc_acc.append(slc_score)

            if "reg" in ret:
                # K = 6
                K_pred_traj_fake_rel = ret['reg']
                ades = [[], [], []]
                fdes = [[], [], []]
                mrs = [[], [], []]
                K_pred_traj_final = []
                if "curves_gt" in ret:
                    curves_gt = ret["curves_gt"]
                    pred_traj_gt_rel = relative_to_curve(batch.fut_traj_fre, batch.obs_traj[-1], curves_gt)
                    pred_traj_gt = relative_to_abs(pred_traj_gt_rel, batch.obs_traj[-1])
                else:
                    pred_traj_gt = batch.fut_traj

                for k in range(config['num_mods']):
                    pred_rel = K_pred_traj_fake_rel[k].permute(1, 0, 2)
                    pred_traj = relative_to_abs(pred_rel, batch.obs_traj[-1])

                    K_pred_traj_final.append(pred_traj[-1])
                    for tt in range(3):
                        ade = cal_ade(
                            pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
                            pred_traj[:config["num_preds"]//3*(tt+1)-1],
                            batch.has_preds[:config["num_preds"]//3*(tt+1)-1], mode='raw'
                        )
                        fde = cal_fde(
                            pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
                            pred_traj[:config["num_preds"]//3*(tt+1)-1],
                            batch.has_preds[:config["num_preds"]//3*(tt+1)-1], mode='raw'
                        )
                        mr = cal_mr(
                            pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
                            pred_traj[:config["num_preds"]//3*(tt+1)-1],
                            batch.has_preds[:config["num_preds"]//3*(tt+1)-1], mode='raw'
                        )
                        ades[tt].append(ade)
                        fdes[tt].append(fde)
                        mrs[tt].append(mr)
                        K_total_preds[tt] += batch.has_preds[:config["num_preds"]//3*(tt+1)-1].sum()
                        K_total_final_preds[tt] += batch.has_preds[config["num_preds"]//3*(tt+1)-1].sum()
                for tt in range(3):
                    K_disp_error[tt].append(torch.stack(ades[tt], 1))
                    K_f_disp_error[tt].append(torch.stack(fdes[tt], 1))
                    K_miss_rate[tt].append(torch.stack(mrs[tt], 1))

                diversities += cal_diversity(torch.stack(K_pred_traj_final, 0))
                diversities_N += batch.has_preds.size(1)

        if "best" in ret:
            metrics['best_ade'] = [sum(best_disp_error[tt]) / total_preds[tt] for tt in range(3)]
            metrics['best_fde'] = [sum(best_f_disp_error[tt]) / total_final_preds[tt] for tt in range(3)]


        if "pred" in ret or "frenet_pred" in ret:
            metrics['g_l2_loss_abs']  = [sum(g_l2_losses_abs) / total_preds[-1]]
            metrics['ade'] = [sum(disp_error[tt]) / total_preds[tt] for tt in range(3)]
            metrics['fde'] = [sum(f_disp_error[tt]) / total_final_preds[tt] for tt in range(3)]
            metrics['mr'] = [sum(miss_rate[tt]) / total_final_preds[tt] for tt in range(3)]

        if "path" in ret:
            metrics['path'] = sum(path_acc) / total_final_preds[-1]

        if "target" in ret:
            metrics['target'] = sum(target_errors) / total_final_preds[-1]
            #metrics['path'] = sum(path_acc) / total_final_preds[-1]
            #metrics['target'] = sum(target_acc) / total_final_preds[-1]
            #print("p", sum(path_acc), path[:5], batch.fut_path[:5])
            #print("t", sum(target_acc), target[:5], batch.fut_target[:5])
            #print("t", total_final_preds[2])
            #print("metric ", metrics['path'])
            #print("logits ", path[batch.has_preds[-1]][:5], path[batch.has_preds[-1]][:5].max(1))
            #print("gt ", batch.fut_path[batch.has_preds[-1]][:5])
        #if "score" in ret and "target" in ret:
        #    metrics['score'] = sum(slc_acc) / total_final_preds[-1]
        #    print("s", sum(slc_acc), slc[:5], slc_idcs[:5])

        if "reg" in ret:

            metrics['kade'] = [torch.cat(K_disp_error[tt], 0).sum().item() / K_total_preds[tt] for tt in range(3)]
            metrics['kfde'] = [torch.cat(K_f_disp_error[tt], 0).sum().item() / K_total_final_preds[tt] for tt in range(3)]

            metrics['minkade'] = [torch.cat(K_disp_error[tt], 0).min(1)[0].sum().item()  / total_preds[tt] for tt in range(3)]
            metrics['minkfde'] = [torch.cat(K_f_disp_error[tt], 0).min(1)[0].sum().item() / total_final_preds[tt] for tt in range(3)]

            metrics['minkmr'] = [torch.cat(K_miss_rate[tt], 0).min(1)[0].sum().item() / total_final_preds[tt] for tt in range(3)]
            metrics['diversity'] = diversities/diversities_N
        model.train()
        return metrics

def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, has_preds, pred_traj_fake, pred_traj_fake_rel):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, has_preds, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, has_preds, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel

def cal_acc(pred, gt):
    score = 0
    #print(pred, pred.size())

    #print(gt, gt.size())
    #_, pred_indexs = pred.max(1)
    #print(pred_indexs.size(), gt.size())
    for pred_idx, gt_idx in zip(pred, gt):
        if gt_idx == pred_idx:
            score += 1
    #print(score)
    return score

def cal_min_acc(pred, gt):
    score = 0
    for pred_idx, gt_idx in zip(pred, gt):
        if gt_idx in pred_idx:
            score += 1
    return score


def cal_ade(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, has_preds, mode=mode)
    return ade
def cal_fde(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], has_preds[-1], mode=mode)
    return fde
def cal_ede(pred_traj_gt, pred_traj_fake, has_preds):
    seq_len = pred_traj_gt.size(0)
    ede = []
    for t in range(seq_len):
        de = final_displacement_error(pred_traj_fake[t], pred_traj_gt[t], has_preds[t])
        ede.append(de)
    return ede

def cal_mr(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
    loss = pred_traj_gt[-1][has_preds[-1]] - pred_traj_fake[-1][has_preds[-1]]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=-1))
    if mode == 'raw':
        return loss > 2.0
    else:
        return torch.sum(loss > 2.0)

def cal_diversity(K_pred_traj_final):
    k, n, _  = K_pred_traj_final.size()
    diversity = K_pred_traj_final.view(k, 1, n, 2) - K_pred_traj_final.view(1, k, n, 2)
    diversity = diversity**2
    diversity = torch.sqrt(diversity.sum(dim=-1))
    i = torch.arange(k).view(1,k).repeat(k,1) > torch.arange(k).view(k,1).repeat(1,k)
    return torch.sum(diversity[i].mean(0)).item()

def l2_loss(pred_traj, pred_traj_gt, has_preds, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    num_preds, batch_size, _ = pred_traj.size()
    gt_preds = pred_traj_gt.permute(1, 0, 2)
    has_preds = has_preds.permute(1, 0)
    preds = pred_traj.permute(1, 0, 2)
    last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
        has_preds.device
    ) / float(num_preds)
    max_last, last_idcs = last.max(1)
    mask = max_last > 1.0
    preds = preds[mask]
    gt_preds = gt_preds[mask]
    has_preds = has_preds[mask]

    loss =  (gt_preds[has_preds] - preds[has_preds])**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / has_preds.sum()
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def displacement_error(pred_traj, pred_traj_gt, has_preds, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    #seq_len, _, _ = pred_traj.size()
    #loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = pred_traj[has_preds] - pred_traj_gt[has_preds]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=-1))
    #loss = loss.sum(dim=2).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss
def final_displacement_error(pred_pos, pred_pos_gt, has_preds, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt[has_preds] - pred_pos[has_preds]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=-1))
    #loss = loss.sum(dim=1)
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
        
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def make_paths(grids, node2lane, lane2node, ego_route, alternative_routes):
    path_feat = []
    all_path_lanelets = []
    path_num_nodes = []
    path_node_node = []
    path_node_sender, path_node_receiver = [], []

    ego_path_lanelets = set()
    for ids_lanelets in ego_route.reference_lanelets:
        for lanelet_id in ids_lanelets:
            if lanelet_id in lane2node:
                ego_path_lanelets.add(lanelet_id)

    all_path_lanelets.append(ego_path_lanelets)
    path_node_node_sender, path_node_node_receiver = [], []
    num_path_nodes = 0
    target_curve = ego_route.curves[ego_route.principle_reference.index(1)]

    for path_segment, code in zip(ego_route.reference_lanelets, ego_route.principle_reference):
        num_segment_nodes = 0
        for llid in path_segment:
            for nid in lane2node[llid]:
                pos = grids[nid].pos_list[-1]
                proj = pos.proj_on_curve(target_curve)
                s = lerp_curve_with_ind(target_curve, proj.ind)
                path_node_sender.append(len(path_num_nodes))
                path_node_receiver.append(nid)
                path_feat.append([s, proj.d])
                #path_feat.append([code, code])
                num_segment_nodes += 1
        for pn in range(num_path_nodes, num_path_nodes+num_segment_nodes):
            for ppn in range(pn, num_path_nodes+num_segment_nodes):
                path_node_node_sender.append(pn)
                path_node_node_receiver.append(ppn)
        num_path_nodes += num_segment_nodes
    path_num_nodes.append(num_path_nodes)
    path_node_node.append([path_node_node_sender, path_node_node_receiver])

    # add routes that are different according to grids
    for route in alternative_routes:
        path_lanelets = set()
        for ids_lanelets in route.reference_lanelets:
            for lanelet_id in ids_lanelets:
                if lanelet_id in lane2node:
                    path_lanelets.add(lanelet_id)
        if path_lanelets in all_path_lanelets:
            continue
        # a path
        all_path_lanelets.append(path_lanelets)

        path_node_node_sender, path_node_node_receiver = [], []
        num_path_nodes = 0
        target_curve = route.curves[route.principle_reference.index(1)]

        for path_segment, code in zip(route.reference_lanelets, route.principle_reference):
            num_segment_nodes = 0
            for llid in path_segment:
                for nid in lane2node[llid]:
                    pos = grids[nid].pos_list[-1]
                    proj = pos.proj_on_curve(target_curve)
                    s = lerp_curve_with_ind(target_curve, proj.ind)
                    path_node_sender.append(len(path_num_nodes))
                    path_node_receiver.append(nid)
                    path_feat.append([s, proj.d])
                    #path_feat.append([code, code])
                    num_segment_nodes += 1
            for pn in range(num_path_nodes, num_path_nodes+num_segment_nodes):
                for ppn in range(pn, num_path_nodes+num_segment_nodes):
                    path_node_node_sender.append(pn)
                    path_node_node_receiver.append(ppn)
            num_path_nodes += num_segment_nodes
        path_num_nodes.append(num_path_nodes)
        path_node_node.append([path_node_node_sender,
                               path_node_node_receiver])
    path_lane_edge_index = [path_node_sender, path_node_receiver]

    return path_feat, \
           path_lane_edge_index, path_node_node, \
           path_num_nodes

class Observation(Data):
    def __init__(self, veh_hist = None, #[xs, ys, mask]
                       veh_state = None, #[d, phi, vel]
                       veh_fut = None, #[delta_s, next_d]
                       veh_shapes = None,  #[length, width]
                       veh_code = None, #{1,0}
                       veh_edge_index = None,
                       candidates = None,
                       path = None, #[reference_code]
                       path_code = None,
                        path_lane_edge_index = None,
                        path_node_node_edge_index = None,
                        path_num_nodes = None,
                        lane_ctrs = None,
                        lane_vecs = None,
                        lane_pris = None,
                        lane_widths = None,
                        lane_suc_edge_index = None,
                        lane_pre_edge_index = None,
                        lane_left_edge_index = None,
                        lane_right_edge_index = None):
        super(Observation, self).__init__()
        self.veh_hist = veh_hist
        self.veh_state = veh_state
        self.veh_fut = veh_fut
        self.veh_shapes = veh_shapes
        self.veh_code = veh_code
        self.candidates = candidates
        self.veh_edge_index = veh_edge_index

        self.path = path
        self.path_code = path_code
        self.path_lane_edge_index = path_lane_edge_index
        self.path_node_node_edge_index = path_node_node_edge_index
        self.path_num_nodes = path_num_nodes

        self.lane_ctrs = lane_ctrs
        self.lane_vecs = lane_vecs
        self.lane_pris = lane_pris
        self.lane_widths = lane_widths
        self.lane_suc_edge_index = lane_suc_edge_index
        self.lane_pre_edge_index = lane_pre_edge_index
        self.lane_left_edge_index = lane_left_edge_index
        self.lane_right_edge_index = lane_right_edge_index

    def __inc__(self, key, value):
        if "index" in key or "face" in key:
            if "veh" in key:
                return self.veh_shape.size(0)
            elif "lane" in key:
                return self.lane_ctrs.size(0)
            elif "path" in key:
                return self.path_num_nodes.size(0)
            else:
                return super(Observation, self).__inc__(key, value)
        else:
            return 0

class Trajectory(Data):
    def __init__(self, obs=None, acts=None, dones=None, infos=None, rews=None):
        super(Trajectory, self).__init__()
        self.obs = obs
        self.acts = acts
        self.dones = dones
        self.infos = infos
        self.rews= rews

def dataclass_quick_asdict(obj) -> Dict[str, Any]:
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d

@dataclasses.dataclass(frozen=True)
class Transitions(Dataset):
    obs : List[Observation]
    next_obs: List[Observation]
    acts: torch.Tensor
    dones: torch.Tensor
    infos: List[Any]

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if isinstance(key, slice):
            # Return type is the same as this dataclass. Replace field value with
            # slices.
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            # Return type is a dictionary. Array values have no batch dimension.
            #
            # Dictionary of np.ndarray values is a convenient
            # torch.util.data.Dataset return type, as a torch.util.data.DataLoader
            # taking in this `Dataset` as its first argument knows how to
            # automatically concatenate several dictionaries together to make
            # a single dictionary batch with `torch.Tensor` values.
            return d_item

class ScenarioData(Data):
    def __init__(self, veh_id=None,
                 veh_t=None, veh_x=None, veh_xseq=None, veh_shape=None, veh_has_preds=None,
                 veh_yseq=None, veh_yfre= None, veh_edge_index=None, veh_edge_attr=None,
                 lane_ctrs=None, lane_vecs=None, lane_pris=None, lane_suc_edge_index=None,
                 lane_pre_edge_index=None, lane_left_edge_index=None, lane_right_edge_index=None,
                 veh_path=None, veh_target=None, lane_start=None, lane_path=None, lane_id = None,
                 veh_full_path=None, veh_path_edge_index=None, path_lane_edge_index=None,
                 lane_widths=None, path_num_nodes=None, path_node_node_edge_index=None):
        super(ScenarioData, self).__init__()
        self.veh_id = veh_id
        self.veh_t = veh_t
        self.veh_x = veh_x
        self.veh_xseq = veh_xseq
        self.veh_shape = veh_shape
        self.veh_yseq = veh_yseq
        self.veh_yfre = veh_yfre
        self.veh_path = veh_path
        self.veh_target = veh_target
        self.veh_has_preds = veh_has_preds
        self.veh_edge_index = veh_edge_index
        self.veh_edge_attr = veh_edge_attr

        self.lane_id = lane_id
        self.lane_ctrs = lane_ctrs
        self.lane_vecs = lane_vecs
        self.lane_pris = lane_pris
        self.lane_suc_edge_index = lane_suc_edge_index
        self.lane_pre_edge_index = lane_pre_edge_index
        self.lane_left_edge_index = lane_left_edge_index
        self.lane_right_edge_index = lane_right_edge_index
        self.lane_start = lane_start
        self.lane_path = lane_path

        self.veh_full_path = veh_full_path
        self.veh_path_edge_index = veh_path_edge_index
        self.path_lane_edge_index = path_lane_edge_index
        self.lane_widths=lane_widths
        self.path_num_nodes = path_num_nodes
        self.path_node_node_edge_index = path_node_node_edge_index

    def __inc__(self, key, value):
        if "index" in key or "face" in key:
            if "veh" in key:
                return self.veh_shape.size(0)
            elif "lane" in key:
                return self.lane_ctrs.size(0)
            elif "path" in key:
                return self.path_num_nodes.size(0)
            else:
                return super(ScenarioData, self).__inc__(key, value)
        else:
            return 0

def transitions_collate_fn(batch):
    """Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.
    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument.
    Args:
        batch: The batch to collate.
    Returns:
        A collated batch. Uses Torch's default collate function for everything
        except the "infos" key. For "infos", we join all the info dicts into a
        list of dicts. (The default behavior would recursively collate every
        info dict into a single dict, which is incorrect.)
    """
    result = {}
    graphs = []
    vehs = {"hists":[], "state":[], "shapes":[], "codes":[]}
    ctrs, actor_idcs = [], []
    count = 0

    for i in range(len(batch)):
        idcs = torch.arange(count, count + len(batch[i]["obs"].veh_hist))
        actor_idcs.append(idcs.cuda())
        count += len(batch[i]["obs"].veh_hist)
        ctrs.append(batch[i]["obs"].veh_hist[:,-1,:2].cuda())

        # add the path edge_indexs
        graph = {"ctrs":batch[i]["obs"].lane_ctrs.cuda(),
                 "feats":torch.cat([batch[i]["obs"].lane_vecs, batch[i]["obs"].lane_widths],-1).cuda(),
                 "pris":batch[i]["obs"].lane_pris.cuda(),
                 "num_nodes":batch[i]["obs"].lane_ctrs.size(0),
                 "num_paths":len(batch[i]["obs"].path_num_nodes),
                 "pre":{k:v.cuda() for k, v in batch[i]["obs"].lane_pre_edge_index.items()},
                 "suc":{k:v.cuda() for k, v in batch[i]["obs"].lane_suc_edge_index.items()},
                 "left":batch[i]["obs"].lane_left_edge_index.cuda(),
                 "right":batch[i]["obs"].lane_right_edge_index.cuda(),
                 "path":batch[i]["obs"].path.cuda(),
                 "path->node":batch[i]["obs"].path_lane_edge_index.cuda(),
                 "path-node->path-node":[e.cuda() for e in batch[i]["obs"].path_node_node_edge_index],
                 "path_num_nodes":batch[i]["obs"].path_num_nodes
                 }
        graphs.append(graph)
    vehs["actor_idcs"] = actor_idcs
    vehs["actor_ctrs"] = ctrs
    vehs["hists"] = torch.cat([b["obs"].veh_hist for b in batch],0).cuda()
    vehs["state"] = torch.cat([b["obs"].veh_state for b in batch],0).cuda()
    vehs["shapes"] = torch.cat([b["obs"].veh_shapes for b in batch],0).cuda()
    vehs["codes"] = torch.cat([b["obs"].veh_code for b in batch],0).cuda() # ego veh
    vehs["paths"] = torch.cat([b["obs"].path_code for b in batch],0).cuda() # ego path
    vehs["graphs"] = graphs

    result["obs"] = vehs
    if "acts" in batch[0]:
        result["acts"] = torch.stack([b["acts"] for b in batch],0).cuda()
    if "acts" in batch[0]:
        result["dones"] = torch.cat([b["dones"] for b in batch],0).cuda()
    if "infos" in batch[0]:
        result["infos"] = [sample["infos"] for sample in batch]
    return result

def traj_collate(batch, include_gt=False):
    batch_size = len(batch)
    graphs = []

    ctrs = []
    actor_idcs = []
    count = 0

    actor_paths = []
    for i in range(batch_size):
        idcs = torch.arange(count, count + len(batch[i].veh_xseq))
        actor_idcs.append(idcs.cuda())
        count += len(batch[i].veh_xseq)
        ctrs.append(batch[i].veh_xseq[:,-1,:2].cuda())

        # first get the last steps that has_preds

        # add the path edge_indexs
        graph = {"ctrs":batch[i].lane_ctrs.cuda(),
                 "feats":torch.cat([batch[i].lane_vecs, batch[i].lane_widths],-1).cuda(),
                 "pris":batch[i].lane_pris.cuda(),
                 "ids":batch[i].lane_id.cuda(),
                 "num_nodes":batch[i].lane_ctrs.size(0),
                 "num_paths":batch[i].path_num_nodes.size(0),
                 "pre":{k:v.cuda() for k, v in batch[i].lane_pre_edge_index.items()},
                 "suc":{k:v.cuda() for k, v in batch[i].lane_suc_edge_index.items()},
                 "left":batch[i].lane_left_edge_index.cuda(),
                 "right":batch[i].lane_right_edge_index.cuda(),
                 "start":batch[i].lane_start.cuda(),

                 "path->node":batch[i].path_lane_edge_index.cuda(),
                 "path-node->path-node":[e.cuda() for e in batch[i].path_node_node_edge_index],
                 "veh->path":batch[i].veh_path_edge_index.cuda(),
                 }
        graphs.append(graph)
        batch[i].lane_suc_edge_index = None
        batch[i].lane_pre_edge_index = None
        batch[i].lane_left_edge_index = None
        batch[i].lane_left_edge_index = None
        batch[i].lane_right_edge_index = None

        batch[i].path_node_node_edge_index = None
        batch[i].lane_start = None
        batch[i].lane_vecs = None
        batch[i].lane_pris = None
        batch[i].lane_id = None
    batch = Batch.from_data_list(batch)

    batch['obs_traj'] = batch.veh_xseq[:,:,:2].permute(1, 0, 2).cuda()
    batch['obs_traj_rel'] = batch.veh_xseq[:,:,2:4].permute(1, 0, 2).cuda()
    batch['obs_info'] = batch.veh_xseq[:,:,4:].permute(1, 0, 2).cuda()
    batch['obs_shape'] = batch.veh_shape.cuda()
    batch['edge_index'] = batch.veh_edge_index.cuda()
    batch['veh_batch'] = actor_idcs
    batch['graphs'] = graphs
    batch['veh_ctrs'] = ctrs
    if include_gt:
        batch['fut_traj'] = batch.veh_yseq[:,:,:2].permute(1, 0, 2).cuda()
        batch['fut_traj_rel'] = batch.veh_yseq[:,:,2:].permute(1, 0, 2).cuda()
        batch['fut_traj_fre'] = batch.veh_yfre.permute(1, 0, 2).type('torch.cuda.FloatTensor')
        #batch['fut_path'] = batch.veh_path.squeeze().cuda()
        batch.veh_full_path = batch.veh_full_path.cuda()
        batch['fut_target'] = batch.veh_target.squeeze().cuda()
        batch['has_preds'] = batch.veh_has_preds.permute(1, 0).cuda()
    return batch

def make_data_loader(demonstrations, batch_size):
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts = {key: [] for key in keys}
    for demo_file in demonstrations:
        traj = torch.load(demo_file)
        parts["acts"].extend(traj.acts)

        obs = traj.obs
        parts["obs"].extend(obs[:-1])
        parts["next_obs"].extend(obs[1:])

        dones = traj.dones
        parts["dones"].append(dones)

        infos = []
        for i in range(len(traj.acts)):
            infos.append(traj.infos+[i])
        parts["infos"].extend(infos)
    parts["acts"] = torch.stack(parts["acts"], dim=0)
    parts["dones"] = torch.cat(parts["dones"], dim=0).unsqueeze(1)
    transitions = Transitions(**parts)
    extra_kwargs = dict(shuffle=True, drop_last=True)
    return DataLoader(
            transitions,
            batch_size=batch_size,
            collate_fn=transitions_collate_fn,
            **extra_kwargs,
        )

def dilated_nbrs(sender, receiver, num_nodes, num_scales):
    data = np.ones(len(sender), np.bool)
    csr = sparse.csr_matrix((data, (sender, receiver)), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        new_sender = coo.row.astype(np.int64)
        new_receiver = coo.col.astype(np.int64)
        nbrs.append([new_sender,new_receiver])
    return nbrs

def make_grids(scenario, ego_id, startframe, grid_length, max_disp_front=55, max_disp_rear=30, max_radius=55, cross_dist=6.0, num_scales=6):
    ego_posG = scenario.obstacles[startframe][ego_id].initial_state.get_posG()
    ego_lanelet_id = scenario.obstacles[startframe][ego_id].initial_state.posF.ind[1]
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

    selected_lanelets = []
    for lanelet in scenario.lanelet_network.lanelets:
        if lanelet.lanelet_id == ego_lanelet_id:
            selected_lanelets.append(lanelet)
            continue
        if len(lanelet.successor)==0 and len(lanelet.predecessor)==0:
            # this is case for isolated lanelets
            continue
        s_pos_x, s_pos_y = lanelet.center_curve[0].pos.x, lanelet.center_curve[0].pos.y
        if (s_pos_x - ego_posG.x)**2+(s_pos_y - ego_posG.y)**2 <= max_radius**2:
            selected_lanelets.append(lanelet)
            continue
        e_pos_x, e_pos_y = lanelet.center_curve[-1].pos.x, lanelet.center_curve[-1].pos.y
        if (e_pos_x - ego_posG.x)**2+(e_pos_y - ego_posG.y)**2 <= max_radius**2:
            selected_lanelets.append(lanelet)
            continue
        for curvept in lanelet.center_curve[1:-1]:
            if (curvept.pos.x - ego_posG.x)**2+(curvept.pos.y - ego_posG.y)**2 <= max_radius**2:
                selected_lanelets.append(lanelet)
                break

    ctrs = []
    vecs = []
    pris = []
    lrdists = []



    suc_edges, suc_edges = {}, {}
    pre_edges, pre_edges = {}, {}
    lane_ids = [lanelet.lanelet_id for lanelet in selected_lanelets]
    grids = []
    for lanelet in selected_lanelets:
        start_curvePt = lanelet.center_curve[0]
        start_curveInd = CurveIndex(0, 0.0)
        nodes = []

        lanelet_leftmost = lanelet
        while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
            lanelet_leftmost = scenario.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
        lanelet_rightmost = lanelet
        while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
            lanelet_rightmost = scenario.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)


        if start_curvePt.s + grid_length*1.5 > lanelet.center_curve[-1].s:
            # at least make two nodes from a single lanelet
            ds = (lanelet.center_curve[-1].s - start_curvePt.s) * 0.5
            center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
            center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
            end_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds)
            end_curvePt = lanelet.get_curvePt_by_curveid(end_curveInd)
            nodes.append([(start_curvePt,start_curveInd),
                      (center_curvePt,center_curveInd),
                      (end_curvePt, end_curveInd)])

            start_curvePt = end_curvePt
            start_curveInd = end_curveInd
            center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
            center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
            end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
            end_curvePt = lanelet.center_curve[-1]
            nodes.append([(start_curvePt,start_curveInd),
                      (center_curvePt,center_curveInd),
                      (end_curvePt, end_curveInd)])

            #continue
        else:
            while start_curvePt.s + grid_length <= lanelet.center_curve[-1].s:
                center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, grid_length*0.5)
                center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
                end_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, grid_length)
                end_curvePt = lanelet.get_curvePt_by_curveid(end_curveInd)
                nodes.append([(start_curvePt,start_curveInd),
                              (center_curvePt,center_curveInd),
                              (end_curvePt, end_curveInd)])

                start_curvePt = end_curvePt
                start_curveInd = end_curveInd
                if lanelet.center_curve[-1].s - start_curvePt.s < 0.5*grid_length:
                    start_curvePt, start_curveInd = nodes[-1][0]
                    ds = lanelet.center_curve[-1].s - start_curvePt.s
                    center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
                    center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
                    end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
                    end_curvePt = lanelet.center_curve[-1]
                    nodes[-1] = [(start_curvePt,start_curveInd),
                              (center_curvePt,center_curveInd),
                              (end_curvePt, end_curveInd)]
                    break
                elif lanelet.center_curve[-1].s - start_curvePt.s >= 0.5*grid_length and \
                      lanelet.center_curve[-1].s - start_curvePt.s < grid_length:
                    ds = lanelet.center_curve[-1].s - start_curvePt.s
                    center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
                    center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
                    end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
                    end_curvePt = lanelet.center_curve[-1]
                    nodes.append([(start_curvePt,start_curveInd),
                              (center_curvePt,center_curveInd),
                              (end_curvePt, end_curveInd)])
                    break

        for n in nodes:
            grids.append(Grid(len(grids)))
            grids[-1].add_pos(n[1][0].pos)
            grids[-1].add_ind((n[1][1], lanelet.lanelet_id))

        ctr = []
        vec = []
        origin_ctr = []
        for n in nodes:
            sta_proj = n[0][0].pos.inertial2body(ego_posG)
            cen_proj = n[1][0].pos.inertial2body(ego_posG)
            las_proj = n[2][0].pos.inertial2body(ego_posG)
            ctr.append([cen_proj.x, cen_proj.y])
            vec.append([las_proj.x-sta_proj.x, las_proj.y-sta_proj.y])
            origin_ctr.append([n[1][0].pos.x, n[1][0].pos.y])

        ctrs.append(np.array(ctr))

        vecs.append(np.array(vec))

        lrdists.append(np.concatenate([lanelet_leftmost.distance_line2line(np.array(origin_ctr), line="left"),
                                 lanelet_rightmost.distance_line2line(np.array(origin_ctr), line="right")]
                                 ,-1))
        #ctrs.append(np.array([[n[1][0].pos.x, n[1][0].pos.y] for n in nodes]))
        #vecs.append(np.array([[n[2][0].pos.x-n[0][0].pos.x, n[2][0].pos.y-n[0][0].pos.y] for n in nodes]))
        if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
            pris.append(np.array([[1,0] for _ in range(len(nodes))]))
        elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
            pris.append(np.array([[0,1] for _ in range(len(nodes))]))
        else:
            pris.append(np.array([[1,1] for _ in range(len(nodes))]))


    node_idcs = []
    count = 0
    node2lane = {}
    lane2node = {lanelet.lanelet_id:[] for lanelet in selected_lanelets}
    for lanelet, ctr in zip(selected_lanelets, ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        lane2node[lanelet.lanelet_id] = range(count, count + len(ctr))
        for idx in node_idcs[-1]:
            node2lane[idx] = lanelet.lanelet_id
        count += len(ctr)
    num_nodes = count

    pre_sender, pre_receiver, suc_sender, suc_receiver = [], [], [], []
    for i, lane in enumerate(selected_lanelets):
        idcs = node_idcs[i]
        pre_sender += idcs[1:]
        pre_receiver += idcs[:-1]
        if len(lane.predecessor) > 0:
            for nbr_id in lane.predecessor:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_sender.append(idcs[0])
                    pre_receiver.append(node_idcs[j][-1])


        suc_sender += idcs[:-1]
        suc_receiver += idcs[1:]

        if len(lane.successor) > 0:
            for nbr_id in lane.successor:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_sender.append(idcs[-1])
                    suc_receiver.append(node_idcs[j][0])

    suc_edges[0] = torch.tensor([suc_sender,suc_receiver], dtype=torch.long)
    pre_edges[0] = torch.tensor([pre_sender,pre_receiver], dtype=torch.long)

    i = 1
    for edges in dilated_nbrs(suc_sender, suc_receiver, num_nodes, num_scales):
        suc_edges[i] = torch.tensor(edges, dtype=torch.long)
        i += 1
    i = 1
    for edges in dilated_nbrs(pre_sender, pre_receiver, num_nodes, num_scales):
        pre_edges[i] = torch.tensor(edges, dtype=torch.long)
        i += 1

    ctrs = torch.tensor(np.concatenate(ctrs, 0), dtype=torch.float)
    vecs = torch.tensor(np.concatenate(vecs, 0), dtype=torch.float)
    pris = torch.tensor(np.concatenate(pris, 0), dtype=torch.float)

    lrdists = torch.tensor(np.concatenate(lrdists, 0), dtype=torch.float)

    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane in enumerate(selected_lanelets):
        nbr_ids = lane.predecessor
        for nbr_id in nbr_ids:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                pre_pairs.append([i, j])

        nbr_ids = lane.successor
        for nbr_id in nbr_ids:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                suc_pairs.append([i, j])

        nbr_id = lane.adj_left
        if nbr_id is not None and lane.adj_left_same_direction:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.adj_right
        if nbr_id is not None and lane.adj_right_same_direction:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])

    pre_pairs = torch.tensor(pre_pairs, dtype=torch.long)
    suc_pairs = torch.tensor(suc_pairs, dtype=torch.long)
    left_pairs = torch.tensor(left_pairs, dtype=torch.long)
    right_pairs = torch.tensor(right_pairs, dtype=torch.long)

    num_lanes = len(selected_lanelets)
    dist = ctrs.unsqueeze(1) - ctrs.unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    pre = pre_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
    pre[pre_pairs[:, 0], pre_pairs[:, 1]] = 1
    suc = suc_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
    suc[suc_pairs[:, 0], suc_pairs[:, 1]] = 1

    pairs = left_pairs
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = vecs[ui]
        f2 = vecs[vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left_edges = torch.stack([ui, vi])
    else:
        left_edges = torch.tensor([np.zeros(0, np.int16), np.zeros(0, np.int16)], dtype=torch.long)

    pairs = right_pairs
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = vecs[ui]
        f2 = vecs[vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right_edges = torch.stack([ui, vi])
    else:
        right_edges = torch.tensor([np.zeros(0, np.int16), np.zeros(0, np.int16)], dtype=torch.long)

    return grids, ctrs,  vecs, pris, lrdists, suc_edges, pre_edges, left_edges, right_edges, node2lane, lane2node
