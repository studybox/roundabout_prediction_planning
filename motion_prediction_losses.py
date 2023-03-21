import torch
from torch import Tensor, nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch_scatter import scatter_mean, scatter_max, scatter_add

class GraphCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GraphCrossEntropyLoss, self).__init__()

    def forward(self, logit: Tensor, gt_target: Tensor, idcs: Tensor, dim_size: int, mask: Tensor):
        #print(logit.size(), gt_target.size(), idcs.size())
        #gt_target = gt_target.unsqueeze(1)
        maxl,_ = scatter_max(logit, idcs, dim=0, dim_size=dim_size)
        maxl = maxl[idcs]
        logit -= maxl
        exp_logit = torch.exp(logit)
        #print(exp_logit.size())
        reduced_exp_logit = scatter_add(exp_logit, idcs, dim=0, dim_size=dim_size)
        #print(reduced_exp_logit.size())
        #print((-logit*gt_target).size())
        loss_logit = scatter_add(-logit*gt_target, idcs, dim=0, dim_size=dim_size)
        #print(loss_logit.size())
        loss = loss_logit + torch.log(reduced_exp_logit)
        #print(torch.mean(loss), has_pred.size())

        return torch.mean(loss[mask])


class ScoreLoss(nn.Module):
    def __init__(self, config):
        super(ScoreLoss, self).__init__()
        self.config = config
        self.softmax = nn.Softmax(dim=1)
        #self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: Tensor, has_preds: Tensor) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        #print("cls", cls.size(), "reg", reg.size())
        gt_preds = gt_preds.permute(1, 0, 2)
        has_preds = has_preds.permute(1, 0)
        #print("gt_preds", gt_preds.size(), "has_preds", has_preds.size())

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        #loss_out["reg_loss"] = zero.clone()
        #loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        #print("has_preds", has_preds.size())


        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)

        # compute the predicted score
        cls_sum_exp = torch.logsumexp(cls, 1)

        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )

        dist = torch.cat([x.unsqueeze(1) for x in dist], 1) # [a, m]

        #print("dist", dist[:5])
        gt_score = self.softmax(-dist/self.config["mgn"])

        #print("cls", cls.size(), gt_score.size(), cls_sum_exp.size())
        loss = -(cls*gt_score).sum(1) + cls_sum_exp
        #print("loss", loss.size())
        loss = loss.sum()
        loss_out["cls_loss"] += loss
        loss_out["num_cls"] += mask.sum().item()
        #raise
        '''

        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        print("mgn", mgn.size(), mgn)
        mgn = mgn[mask0 * mask1]
        print("mask01", (mask0 * mask1)[:5])
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]

        print("mask", mask[:5], mask.size(), "sum", mask.sum().item())

        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()
        '''
        #reg = reg[row_idcs, min_idcs]
        #print("reg", reg.size())
        #print("gt_preds", gt_preds.size())

        #print("reg_masked", reg[has_preds].size())
        #print("pred_masked", gt_preds[has_preds].size() )
        #coef = self.config["reg_coef"]
        #print("preds", reg[has_preds].size(), has_preds.sum().item())
        #print(reg[10], gt_preds[10])
        #loss_out["reg_loss"] += coef * self.reg_loss(
        #    reg[has_preds], gt_preds[has_preds]
        #)
        #loss_out["num_reg"] += has_preds.sum().item()
        return loss_out

class LaneGCNLoss(nn.Module):
    def __init__(self, config):
        super(LaneGCNLoss, self).__init__()
        self.config = config

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: Tensor, has_preds: Tensor) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)

        gt_preds = gt_preds.permute(1, 0, 2)
        has_preds = has_preds.permute(1, 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        # compute the predicted score
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )

        dist = torch.cat([x.unsqueeze(1) for x in dist], 1) # [a, m]

        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]

        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()
        return loss_out


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.
    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

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
