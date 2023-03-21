import os
import os.path as osp
import torch
import warnings
import torch.nn as nn
import math
from torch.nn.functional import binary_cross_entropy_with_logits

import numpy as np
import random
from torch_geometric.nn.conv import MessagePassing
from torch.nn import functional as F
from torch.autograd import Variable
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_softmax

from torch.utils.tensorboard import SummaryWriter
from motion_prediction_layers import *
from motion_prediction_losses import GraphCrossEntropyLoss

from util import relative_to_abs, relative_to_curve
from math import gcd
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
#from model import ActorNet, MapNet, A2M, M2M, M2A, A2A, ScoreNet

from torch.nn.parameter import Parameter
from torch_geometric.nn import inits


def get_dtypes(use_gpu=1):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def is_uninitialized_parameter(x: Any) -> bool:
    if not hasattr(nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, nn.parameter.UninitializedParameter)


class Linear(torch.nn.Module):
    r"""Applies a linear tranformation to the incoming data

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    similar to :class:`torch.nn.Linear`.
    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
            If set to :obj:`None`, will match default weight initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
            If set to :obj:`None`, will match default bias initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)

    Shapes:
        - **input:** features :math:`(*, F_{in})`
        - **output:** features :math:`(*, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._load_hook = self._register_load_state_dict_pre_hook(
            self._lazy_load_hook)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(self.in_channels, self.out_channels, self.bias
                     is not None, self.weight_initializer,
                     self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if self.in_channels <= 0:
            pass
        elif self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'uniform':
            bound = 1.0 / math.sqrt(self.weight.size(-1))
            torch.nn.init.uniform_(self.weight.data, -bound, bound)
        elif self.weight_initializer == 'kaiming_uniform':
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        elif self.weight_initializer is None:
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        else:
            raise RuntimeError(f"Linear layer weight initializer "
                               f"'{self.weight_initializer}' is not supported")

        if self.bias is None or self.in_channels <= 0:
            pass
        elif self.bias_initializer == 'zeros':
            inits.zeros(self.bias)
        elif self.bias_initializer is None:
            inits.uniform(self.in_channels, self.bias)
        else:
            raise RuntimeError(f"Linear layer bias initializer "
                               f"'{self.bias_initializer}' is not supported")


    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): The features.
        """
        return F.linear(x, self.weight, self.bias)


    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if is_uninitialized_parameter(self.weight):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias.detach()

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):

        weight = state_dict[prefix + 'weight']
        if is_uninitialized_parameter(weight):
            self.in_channels = -1
            self.weight = nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif is_uninitialized_parameter(self.weight):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')

class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        #kwargs.setdefault('flow', 'target_to_source')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        assert in_channels[1] == out_channels
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            #self.lin_skip = nn.Linear(in_channels[1], heads * out_channels,
            #                       bias=bias)
            self.lin_skip = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            #self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            self.lin_skip = nn.Linear(out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        if self.root_weight:
            self.feed_forward = nn.Sequential(
                nn.Linear(out_channels, 2*out_channels),
                nn.ReLU(),
                nn.Linear(2*out_channels, out_channels)
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=(x[0].size(0), x[1].size(0)))

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = self.lin_skip(out)
        out = F.dropout(self.norm1(out + x[1]), p=self.dropout, training=self.training)

        if self.root_weight:
            forward = self.feed_forward(out)
            out = F.dropout(self.norm2(forward + x[1]), p=self.dropout, training=self.training)
            #x_r = self.lin_skip(x[1])
            #if self.lin_beta is not None:
            #    beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
            #    beta = beta.sigmoid()
            #    out = beta * x_r + (1 - beta) * out
            #else:
            #    out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        #alpha = softmax(alpha, index, ptr, size_i)
        alpha = scatter_softmax(alpha, index, dim=0)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

def graph_gather(graphs, veh_batch):
    batch_size = len(graphs)
    graph = dict()
    veh_count = 0
    veh_counts = []
    for i in range(batch_size):
        veh_counts.append(veh_count)
        veh_count = veh_count + veh_batch[i].size(0)

    graph["num_vehs"] = veh_count
    node_idcs = []
    node_count = 0
    node_counts = []
    for i in range(batch_size):
        node_counts.append(node_count)
        idcs = torch.arange(node_count, node_count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        node_count = node_count + graphs[i]["num_nodes"]

    graph["num_nodes"] = node_count
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]
    for key in ["feats", "pris", "start", "ids"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)
    # graph["obs_idcs"] = [x["start"] for x in graphs]

    path_idcs = []
    path_count = 0
    path_counts = []
    for i in range(batch_size):
        path_counts.append(path_count)
        idcs = torch.arange(path_count, path_count + graphs[i]["num_paths"]).to(
            graphs[i]["feats"].device
        )
        path_idcs.append(idcs)
        path_count = path_count + graphs[i]["num_paths"]

    graph["num_paths"] = path_count
    graph["path_idcs"] = path_idcs # number of batch size with path indices
    # edges between path and nodes
    graph["path->node"] = dict()
    graph["path->node"]["u"] = torch.cat(
            [graphs[j]["path->node"][0] + path_counts[j] for j in range(batch_size)], 0
        )
    graph["path->node"]["v"] = torch.cat(
        [graphs[j]["path->node"][1] + node_counts[j] for j in range(batch_size)], 0
    )

    graph["path-node->path-node"] = []
    for g in graphs:
        graph["path-node->path-node"].extend(g["path-node->path-node"])

    graph["veh->path"] = dict()
    graph["veh->path"]["u"] = torch.cat(
        [graphs[j]["veh->path"][0] + veh_counts[j] for j in range(batch_size)], 0
    )
    graph["veh->path"]["v"] = torch.cat(
        [graphs[j]["veh->path"][1] + path_counts[j] for j in range(batch_size)], 0
    )
    graph["veh_path_idcs"] = []
    for i in range(graph["num_vehs"]):
        graph["veh_path_idcs"].append(graph["veh->path"]["v"][graph["veh->path"]["u"]==i])

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for ki, k2 in enumerate(["u", "v"]):
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][ki] + node_counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for ki, k2 in enumerate(["u", "v"]):
            temp = [graphs[i][k1][ki] + node_counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph


class FrenetPathMultiTargetGCN(nn.Module):
    def __init__(self, config):
        super(FrenetPathMultiTargetGCN, self).__init__()
        self.config = config
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        self.path_net = PathPredNet(config, latent=False)
        self.tar_traj_net = TargetsPredAttNet(config)

        self.path_loss = GraphCrossEntropyLoss()
        self.target_cls_loss = nn.CrossEntropyLoss(reduction="mean")
        self.target_offset_loss = nn.SmoothL1Loss(reduction="mean")
        self.traj_loss = nn.SmoothL1Loss(reduction="mean")

        self.device = torch.device('cuda')
        if "save_path" in config:
            self.writer = SummaryWriter(config["save_path"])
        else:
            self.writer = None

    def select_paths(self, veh_path_idcs, path_num_nodes, nodes, graph):
        hi = []
        wi = []
        p_hi = []
        p_wi = []
        count = 0

        div_term = torch.exp(torch.arange(0, self.config["n_map"], 2) * (-math.log(10000.0) / self.config["n_map"]))
        pe = torch.zeros(40, self.config["n_map"]).cuda()
        position = torch.arange(40).unsqueeze(1)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)

        PE = []

        traj_edge_index = []
        node_traj_edge_index = []
        actor_traj_index = []

        for i in range(veh_path_idcs.size(0)):
            w = graph["path->node"]["v"][graph["path->node"]["u"]==veh_path_idcs[i,0]]
            PE.append(pe[:w.size(0)])

            wi.append(w)
            hi.append(torch.ones_like(w)*i)
            try:
                node_node_edge_index = graph["path-node->path-node"][veh_path_idcs[i,0]]
            except:
                print(len(graph["path-node->path-node"]), i, veh_path_idcs[i,0], veh_path_idcs)
                raise
            p_hi.append(node_node_edge_index[0]+count)
            p_wi.append(node_node_edge_index[1]+count)


            traj_sender, traj_receiver = [], []
            for pn in range(self.config["num_preds"]):
                for ppn in range(pn, self.config["num_preds"]):
                    traj_sender.append(pn + i*self.config["num_preds"])
                    traj_receiver.append(ppn + i*self.config["num_preds"])
            traj_edge_index.append(torch.tensor([traj_sender,
                                                 traj_receiver],
                                                 dtype=torch.long).cuda()
                                    )
            node_traj_sender, node_traj_receiver = [], []
            for pn in range(w.size(0)):
                for ppn in range(self.config["num_preds"]):
                    node_traj_sender.append(pn + count)
                    node_traj_receiver.append(ppn + i*self.config["num_preds"])
            node_traj_edge_index.append(torch.tensor([node_traj_sender,
                                                 node_traj_receiver],
                                                 dtype=torch.long).cuda()
                                    )
            actor_traj_index.append(i*torch.ones(self.config["num_preds"],dtype=torch.long).cuda())

            count += path_num_nodes[veh_path_idcs[i,0]]

        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)
        p_hi = torch.cat(p_hi, 0)
        p_wi = torch.cat(p_wi, 0)

        PE = torch.cat(PE, dim=0)
        target_paths = nodes[wi] + PE

        traj_edge_index = torch.cat(traj_edge_index, dim=1)
        node_traj_edge_index = torch.cat(node_traj_edge_index, dim=1)
        actor_traj_index = torch.cat(actor_traj_index)

        initial_poses = graph["start"][veh_path_idcs].squeeze(1)
        return target_paths, \
               initial_poses, \
               torch.stack([p_hi, p_wi]), \
               torch.stack([torch.arange(0, len(wi)).to(nodes.device), hi]), \
               traj_edge_index,\
               node_traj_edge_index,\
               actor_traj_index

    def update_summary(self, name, value, step):
        if isinstance(value, int) or isinstance(value, float):
            self.writer.add_scalar(name, value, step)
        else:
            self.writer.add_scalars(name, value, step)

    def forward(self, batch):
        # obs_traj_rel:[S, N8, 2] S:10
        # obs_info: [S, NB, 1] S:10
        actor_idcs = batch.veh_batch
        actor_ctrs = batch.veh_ctrs
        batch_size = len(actor_idcs)

        actors_x =  torch.cat([batch.obs_traj_rel, batch.obs_info], -1).permute(1, 2, 0)
        actors_x = self.actor_net(actors_x)

        # contruct traffic features
        graph = graph_gather(batch.graphs, batch.veh_batch)
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        lane_ctrs = torch.cat(node_ctrs, 0)

        nodes = self.a2m(nodes, graph, actors_x, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)


        actors_x = self.m2a(actors_x, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors_x = self.a2a(actors_x, actor_idcs, actor_ctrs)
        #actor_edge_index = self.actor_edges(actor_idcs, actor_ctrs)

        gt_preds = batch.fut_traj_fre[:,:,:2].permute(1, 0, 2)
        gt_targets = torch.stack([batch.fut_target[:,0],
                                  batch.fut_target[:,2]], dim=-1)

        S = torch.arange(0,60,3.)
        D = torch.arange(-3,4,3.)
        tar_candidate = torch.stack([S.unsqueeze(0).repeat(3,1),
                                     D.unsqueeze(1).repeat(1,20)], dim=-1).to(gt_targets.device)
        tar_candidate = tar_candidate.reshape(1, -1, 2).repeat(gt_targets.size(0), 1, 1)
        offsets = gt_targets.unsqueeze(1) - tar_candidate
        _, gt_target_idcs = (offsets**2).sum(-1).min(-1)
        row_idcs = torch.arange(gt_target_idcs.size(0)).to(gt_targets.device)
        gt_target_offsets = offsets[row_idcs, gt_target_idcs]

        has_preds = batch.has_preds.permute(1, 0)
        last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
            has_preds.device
        ) / float(self.config['num_preds'])
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        if self.config["num_mods"] == 1:
            if self.config["sample_mode"] == "ground_truth":
                veh_path_idcs = veh_path_idcs_gt = torch.arange(0, graph["num_paths"]).unsqueeze(1)[batch.veh_full_path == 1].unsqueeze(1).cuda()

            else:
                paths, _ = scatter_max(nodes[graph["path->node"]["v"]], graph["path->node"]["u"], dim=0, dim_size=graph["num_paths"])
                path_logits = self.path_net(actors_x, paths, None, None, graph)
                path_probs = scatter_softmax(path_logits, graph['veh->path']['u'], dim=0).squeeze()
                path_loss = self.path_loss(path_logits, batch.veh_full_path, graph["veh->path"]["u"], has_preds.size(0), mask)
                _, veh_path_idcs_pred = scatter_max(path_logits, graph["veh->path"]["u"], dim=0, dim_size=graph["num_vehs"])
                veh_path_idcs_gt = torch.arange(0, graph["num_paths"]).unsqueeze(1)[batch.veh_full_path == 1].unsqueeze(1).cuda()

            #if self.training:
            veh_path_idcs = veh_path_idcs_gt

            curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs]
            path_nodes, initial_poses,\
            node_edge_index, \
            node_target_edge_index, \
            traj_edge_index, \
            node_traj_edge_index,\
            actor_traj_index = self.select_paths(veh_path_idcs, batch.path_num_nodes, nodes, graph)
            traj_pred, target_pred, target_prob, target_logits, target_offset, _ = self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                      gt_targets,
                                      tar_candidate,
                                      node_edge_index,
                                      node_target_edge_index,
                                      traj_edge_index,
                                      node_traj_edge_index,
                                      actor_traj_index )

        else:
            # Generation of K paths
            frenet_preds = []
            frenet_curves = []
            preds = []
            target_preds = []
            all_target_probs = []

            paths, _ = scatter_max(nodes[graph["path->node"]["v"]], graph["path->node"]["u"], dim=0, dim_size=graph["num_paths"])
            path_logits = self.path_net(actors_x, paths, None, None, graph)
            path_probs = scatter_softmax(path_logits, graph['veh->path']['u'], dim=0).squeeze()

            if self.config["sample_mode"] == "ucb" or self.config["sample_mode"] == "bias":
                if self.config["sample_mode"] == "ucb":
                    veh_path_idcs, ranks = ucb_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
                    #print(veh_path_idcs.size(), ranks.size())
                else:
                    veh_path_idcs, ranks = bias_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
                    #print(veh_path_idcs.size(), ranks.size())
                path_loss = self.path_loss(path_logits, batch.veh_full_path, graph["veh->path"]["u"], has_preds.size(0), mask)

            else:
                veh_path_idcs, ranks = uniform_sample_path(graph, num_samples=self.config['num_mods'])

            scores = []
            for m in range(self.config['num_mods']):
                # the first version doesn't consider the joint path distribution
                path_nodes, initial_poses,\
                node_edge_index, \
                node_target_edge_index, \
                traj_edge_index, \
                node_traj_edge_index,\
                actor_traj_index = self.select_paths(veh_path_idcs[m].unsqueeze(1), batch.path_num_nodes, nodes, graph)
                traj_pred, target_pred, target_prob, _, _, target_probs = self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                          gt_targets,
                                          tar_candidate,
                                          node_edge_index,
                                          node_target_edge_index,
                                          traj_edge_index,
                                          node_traj_edge_index,
                                          actor_traj_index, ranks[m])
                scores.append(path_probs[veh_path_idcs[m]]*target_prob)
                all_target_probs.append(target_probs)
                frenet_preds.append(traj_pred)

                curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs[m]]
                frenet_curves.append(curves)
                preds.append(relative_to_curve(frenet_preds[-1].permute(1, 0, 2), batch.obs_traj[-1], curves).permute(1, 0, 2))
                target_preds.append(target_pred)

            scores = torch.stack(scores, 1)
            frenet_preds = torch.stack(frenet_preds, 1)
            #score_out = self.score_net(actors_x, actor_idcs, actor_ctrs, preds)
            #score_loss = self.score_loss(score_out, batch.fut_traj, batch.has_preds)
            #score_loss = score_loss['cls_loss']/(score_loss['num_cls']+1e-10)
            #frenet_preds_sorted = frenet_preds[score_out["row_idcs"], score_out["sort_idcs"]].view(score_out["cls_sorted"].size(0),
            #                                                                                       score_out["cls_sorted"].size(1), -1, 2)

            #if self.training:
            veh_path_idcs_gt = torch.arange(0, graph["num_paths"]).unsqueeze(1)[batch.veh_full_path == 1].unsqueeze(1).cuda()
            path_nodes, initial_poses,\
            node_edge_index, \
            node_target_edge_index, \
            traj_edge_index, \
            node_traj_edge_index,\
            actor_traj_index = self.select_paths(veh_path_idcs_gt, batch.path_num_nodes, nodes, graph)
            traj_pred, target_pred, _, target_logits, target_offset, _= self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                          gt_targets,
                                          tar_candidate,
                                          node_edge_index,
                                          node_target_edge_index,
                                          traj_edge_index,
                                          node_traj_edge_index,
                                          actor_traj_index )
            curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs_gt]
            converted_pred = relative_to_curve(traj_pred.permute(1, 0, 2), batch.obs_traj[-1], curves)
            #else:
            #    traj_pred = frenet_preds_sorted[:, 0]
            #    converted_pred = score_out['pred'].permute(1, 0, 2)

        traj_loss = self.traj_loss(traj_pred[mask][has_preds[mask]], gt_preds[mask][has_preds[mask]])
        target_cls_loss = self.target_cls_loss(target_logits[mask], gt_target_idcs[mask])
        target_offset_loss = self.target_offset_loss(target_offset[row_idcs, gt_target_idcs][mask], gt_target_offsets[mask])

        tar = batch.veh_full_path==1
        path_idx = graph['veh->path']["v"][tar.squeeze()]
        curves_gt = []
        for idx in range(path_idx.size(0)):
            curves_gt.append(lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==path_idx[idx]]])

        if self.config["num_mods"] == 1:
            ret = {'frenet_pred': traj_pred.permute(1, 0, 2),
                   'curves':curves,
                   'curves_gt':curves_gt,
                   'converted_gt':batch.fut_traj_fre[:,:,2:],
                   'loss':traj_loss + 0.5*path_loss + 0.1*(target_cls_loss+target_offset_loss),
                   'tloss':traj_loss,
                   'ploss':path_loss,
                   'target':target_pred,
                   'tar_cls_loss':target_cls_loss,
                   'tar_offset_loss':target_offset_loss,
                   'path': veh_path_idcs_pred,
                   'gt_path': veh_path_idcs_gt}

        else:
            ret = {'frenet_pred': traj_pred.permute(1, 0, 2),
                   'converted_pred': converted_pred,
                   'curves':frenet_curves,
                   'curves_gt':curves_gt,
                   'converted_gt':batch.fut_traj_fre[:,:,2:],
                   'loss':traj_loss + 0.5*path_loss + 0.1*(target_cls_loss+target_offset_loss),
                   'tloss':traj_loss,
                   'ploss':path_loss,
                   'tar_cls_loss':target_cls_loss,
                   'tar_offset_loss':target_offset_loss,
                   'paths': veh_path_idcs,
                   'path_probs':path_probs,
                   'targets': target_preds,
                   'target_probs':all_target_probs,
                   'scores': scores,
                   'reg':preds}

        return ret

    def predict(self, batch):
        # obs_traj_rel:[S, N8, 2] S:10
        # obs_info: [S, NB, 1] S:10
        actor_idcs = batch.veh_batch
        actor_ctrs = batch.veh_ctrs
        batch_size = len(actor_idcs)

        actors_x =  torch.cat([batch.obs_traj_rel, batch.obs_info], -1).permute(1, 2, 0)
        actors_x = self.actor_net(actors_x)

        # contruct traffic features
        graph = graph_gather(batch.graphs, batch.veh_batch)
        lane_ids = graph["ids"]
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        lane_ctrs = torch.cat(node_ctrs, 0)

        nodes = self.a2m(nodes, graph, actors_x, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)

        actors_x = self.m2a(actors_x, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors_x = self.a2a(actors_x, actor_idcs, actor_ctrs)
        #actor_edge_index = self.actor_edges(actor_idcs, actor_ctrs)

        #gt_preds = batch.fut_traj_fre[:,:,:2].permute(1, 0, 2)
        #gt_targets = torch.stack([batch.fut_target[:,0],
        #                          batch.fut_target[:,2]], dim=-1)

        S = torch.arange(0,60,3.)
        D = torch.arange(-3,4,3.)
        tar_candidate = torch.stack([S.unsqueeze(0).repeat(3,1),
                                     D.unsqueeze(1).repeat(1,20)], dim=-1).to(actors_x.device)
        tar_candidate = tar_candidate.reshape(1, -1, 2).repeat(actors_x.size(0), 1, 1)
        #offsets = gt_targets.unsqueeze(1) - tar_candidate
        #_, gt_target_idcs = (offsets**2).sum(-1).min(-1)
        #row_idcs = torch.arange(gt_target_idcs.size(0)).to(gt_targets.device)
        #gt_target_offsets = offsets[row_idcs, gt_target_idcs]

        #has_preds = batch.has_preds.permute(1, 0)
        #last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
        #    has_preds.device
        #) / float(self.config['num_preds'])
        #max_last, last_idcs = last.max(1)
        #mask = max_last > 1.0


        # Generation of K paths
        frenet_preds = []
        frenet_curves = []
        preds = []
        target_preds = []
        all_target_probs = []

        paths, _ = scatter_max(nodes[graph["path->node"]["v"]], graph["path->node"]["u"], dim=0, dim_size=graph["num_paths"])
        path_logits = self.path_net(actors_x, paths, None, None, graph)
        path_probs = scatter_softmax(path_logits, graph['veh->path']['u'], dim=0).squeeze()

        if self.config["sample_mode"] == "ucb" or self.config["sample_mode"] == "bias":
            if self.config["sample_mode"] == "ucb":
                veh_path_idcs, ranks = ucb_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
            else:
                veh_path_idcs, ranks = bias_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
        else:
            veh_path_idcs, ranks = uniform_sample_path(graph, num_samples=self.config['num_mods'])

        scores = []
        path_lane_ids = []
        #predictions = dict()
        #veh_ids = batch.veh_id.numpy().astype(int)
        for m in range(self.config['num_mods']):
            path_nodes, initial_poses,\
            node_edge_index, \
            node_target_edge_index, \
            traj_edge_index, \
            node_traj_edge_index,\
            actor_traj_index = self.select_paths(veh_path_idcs[m].unsqueeze(1), batch.path_num_nodes, nodes, graph)
            traj_pred, target_pred, target_prob, _, _, target_probs = self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                      None,
                                      tar_candidate,
                                      node_edge_index,
                                      node_target_edge_index,
                                      traj_edge_index,
                                      node_traj_edge_index,
                                      actor_traj_index, ranks[m])
            scores.append((path_probs[veh_path_idcs[m]]*target_prob))
            all_target_probs.append(target_probs.cpu().numpy())
            frenet_preds.append(traj_pred.cpu().numpy())
            target_preds.append(target_pred.cpu().numpy())
            path_lane_ids.append(([(lane_ids[graph['path->node']["v"][graph['path->node']["u"]==pidx]]).cpu().numpy().astype(int) for pidx in veh_path_idcs[m]]))
            frenet_curves.append(([(lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]]).cpu().numpy() for pidx in veh_path_idcs[m]]))
            #pred_traj_rel = relative_to_curve(frenet_preds[-1].permute(1, 0, 2), batch.obs_traj[-1], curves)
            #pred_traj = relative_to_abs(pred_traj_rel, batch.obs_traj[-1])

            # x, y, h, phi, s, d, sa, da

        scores = torch.stack(scores, 1).cpu().numpy()
        #frenet_preds = torch.stack(frenet_preds, 1)

        ret = {'curves':frenet_curves,
               'paths': veh_path_idcs,
               "lane_ids": path_lane_ids,
               'path_probs':path_probs,
               'targets': target_preds,
               'target_probs':all_target_probs,
               'scores': scores,
               'trajs':frenet_preds,
               'last_pos':batch.obs_traj[-1].cpu().numpy(),
               'veh_ids':batch.veh_id.cpu().numpy().astype(int)}

        return ret




def ucb_sample_path(graph, logits, num_samples=6, c=0.2):
    logits = logits.squeeze()
    idcs = []
    n = torch.zeros_like(logits)
    exp_logits = scatter_softmax(logits, graph["veh->path"]["u"], dim=0)
    _, idc = scatter_max(exp_logits, graph["veh->path"]["u"], dim=0)
    idcs.append(idc.view(1,-1))
    ranks = []
    ranks.append(n[idc])
    n[idc] += 1

    for s in range(num_samples-1):
        sn = scatter_add(n, graph["veh->path"]["u"], dim=0) + 0.01
        sn = sn[graph["veh->path"]["u"]]
        _, idc = scatter_max(exp_logits + c*torch.sqrt(torch.log(sn)/n), graph["veh->path"]["u"], dim=0)
        idcs.append(idc.view(1,-1))
        ranks.append(n[idc])
        n[idc] += 1
    return torch.cat(idcs, dim=0), torch.stack(ranks, dim=0).long()

def bias_sample_path(graph, logits, num_samples=6):
    idcs = []
    ranks = []
    n = torch.zeros_like(logits)
    for path_idcs in graph["veh_path_idcs"]:
        m = torch.distributions.categorical.Categorical(logits=logits.squeeze()[path_idcs])
        idcs.append(path_idcs[m.sample((num_samples,))])

    idcs = torch.stack(idcs, dim=1)
    for s in range(num_samples):
        idc = idcs[s]
        ranks.append(n[idc])
        n[idc] += 1
    return idcs.to(logits.device), torch.stack(ranks, dim=0).squeeze().long()

def uniform_sample_path(graph, num_samples=6):
    idcs = []
    ranks = []
    n = torch.zeros_like(logits)
    for path_idcs in graph["veh_path_idcs"]:
        idcs.append(torch.randint(path_idcs[0], path_idcs[-1]+1, (num_samples,1)))
    idcs = torch.cat(idcs, dim=1)
    for s in range(num_samples):
        idc = idcs[s]
        ranks.append(n[idc])
        n[idc] += 1
    return idcs.to(logits.device), torch.stack(ranks, dim=0)


class PathPredNet(nn.Module):
    def __init__(self, config, latent=True):
        super(PathPredNet, self).__init__()
        self.config = config
        self.latent = latent
        if latent:
            size = config["n_actor"]+config["n_map"]+config["act_continuous_size"]+config["pat_continuous_size"]
        else:
            size = config["n_actor"]+config["n_map"]

        self.logit = nn.Sequential(
            LinRes(size, size//2, norm="GN", ng=1),
            nn.Linear(size//2, 1),
        )
    def forward(self, actors, paths, Z_act, Z_pat, graph):
        if self.latent:
            paths = torch.cat([paths, Z_pat, actors[graph["veh->path"]["u"]], Z_act[graph["veh->path"]["u"]]], -1)
        else:
            paths = torch.cat([paths, actors[graph["veh->path"]["u"]]], -1)
        logits = self.logit(paths)
        return logits


class TargetsPredAttNet(nn.Module):
    def __init__(self, config):
        super(TargetsPredAttNet, self).__init__()
        self.config = config
        self.output_dim = 2
        self.num_preds = config["num_preds"]
        dropout = config["dropout"]
        self.num_blocks = config["num_blocks"]

        self.node_blocks = nn.ModuleList(
            [
                TransformerConv(config["n_map"],
                                config["n_map"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.target_blocks = nn.ModuleList(
            [
                TransformerConv((config["n_map"], config["n_actor"]),
                                config["n_actor"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )

        self.hidden2target = nn.Sequential(
            LinRes(config["n_actor"]+2, config["n_actor"], norm="GN", ng=1),
            nn.Linear(config["n_actor"], 1)
        )

        self.hidden2offset = nn.Sequential(
            LinRes(config["n_actor"]+2, config["n_actor"], norm="GN", ng=1),
            nn.Linear(config["n_actor"], 2)
        )

        self.path_blocks = nn.ModuleList(
                [
                   TransformerConv(config["n_path"],
                                   config["n_path"],
                                   heads=config["n_head"],
                                   dropout=dropout)
                   for _ in range(config["num_blocks"])
                ]
        )
        self.traj_blocks = nn.ModuleList(
            [
                TransformerConv((config["n_path"], config["n_actor"]),
                                config["n_actor"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.spatial_embedding = nn.Linear(3, config["n_path"])

        self.hidden2pos = nn.Sequential(
            LinRes(config["n_actor"], config["n_actor"], norm="GN", ng=1),
            nn.Linear(config["n_actor"], 2),
        )

    def forward(self, actors, nodes, initial_poses, target_poses_gt, candidates,
                node_edge_index, node_target_edge_index,
                traj_edge_index, node_traj_edge_index, actor_traj_index, ranks=None):
        batch = actors.size(0)
        # give nodes a sequential order info
        for block in self.node_blocks:
            nodes, ia = block(nodes, node_edge_index, return_attention_weights=True)

        targets = actors
        for block in self.target_blocks:
            targets = block((nodes, targets), node_target_edge_index)

        n = candidates.size(1)
        target_cands = torch.cat([targets.unsqueeze(1).repeat(1, n, 1), candidates], dim=2)
        target_logits = self.hidden2target(target_cands.reshape(batch*n, -1))
        target_logits = target_logits.reshape(batch, n)
        target_offsets = self.hidden2offset(target_cands.reshape(batch*n, -1))
        target_offsets = target_offsets.reshape(batch, n, 2)
        # make K-th prediction
        row_idcs = torch.arange(batch).cuda()
        _, top_idcs = target_logits.topk(self.config['num_mods'], dim=1)
        if ranks is None:
            col_idcs = top_idcs[:, 0]
        else:
            col_idcs = top_idcs[row_idcs, ranks]
        target_poses = candidates[row_idcs, col_idcs] + target_offsets[row_idcs, col_idcs]
        target_probs = F.softmax(target_logits, dim=-1)
        target_prob = target_probs[row_idcs, col_idcs]
        # give paths a sequential order info
        if self.training:
            paths = self.generate_frenet_path(initial_poses, target_poses_gt)
        else:
            paths = self.generate_frenet_path(initial_poses, target_poses)
        paths = self.spatial_embedding(paths)
        for block in self.path_blocks:
            paths = block(paths, traj_edge_index)

        trajs = actors[actor_traj_index] + paths
        for block in self.traj_blocks:
            trajs = block((nodes, trajs), node_traj_edge_index)
        trajs = torch.reshape(self.hidden2pos(trajs), [-1, self.config["num_preds"], 2])
        return trajs, target_poses, target_prob, target_logits, target_offsets, target_probs

    def generate_frenet_path(self, initial_poses, target_poses):
        T = self.config["dt"] * self.config["delta_step"] * self.config["num_preds"]
        s = target_poses[:,0]
        vs0 = initial_poses[:,0]
        sa = 2*(s - vs0*T)/T**2

        d = target_poses[:,1]
        d0 = initial_poses[:, 1]
        vd0 = initial_poses[:, 2]

        da = 2*(d-d0 - vd0*T)/T**2

        pe = torch.zeros(initial_poses.size(0), self.config["num_preds"], 3).cuda()
        t = torch.arange(1, self.config["num_preds"]+1).cuda() * self.config["dt"] * self.config["delta_step"]
        pe[:, :, 0] = t.view(1,-1)*torch.ones_like(s).view(-1,1)
        pe[:, :, 1] = t.view(1,-1)*vs0.view(-1,1) + (t**2).view(1,-1)*0.5*sa.view(-1,1)
        pe[:, :, 2] = t.view(1,-1)*vd0.view(-1,1) + (t**2).view(1,-1)*0.5*da.view(-1,1) + d0.view(-1,1)
        return torch.reshape(pe, [-1, 3])

class TrajPredAttNet(nn.Module):
    def __init__(self, config, latent=True):
        super(TrajPredAttNet, self).__init__()
        self.config = config
        self.decoder_h_dim = config["n_actor"]
        self.embedding_dim = config["n_actor"]
        self.latent = latent
        if latent:
            self.z_act_dim = config['act_continuous_size']
            self.z_int_dim = config['int_continuous_size']
        else:
            self.z_act_dim = 0
            self.z_int_dim = 0
        self.output_dim = 2
        self.num_preds = config["num_preds"]
        dropout = config["dropout"]
        self.num_blocks = config["num_blocks"]

        self.num_blocks = config["num_blocks"]
        self.path_blocks = nn.ModuleList(
            [
                TransformerConv(config["n_path"],
                                config["n_path"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.int_blocks = nn.ModuleList(
            [
                TransformerConv(config["n_actor"],
                                config["n_actor"],
                                heads=config["n_head"],
                                #edge_dim=config['int_continuous_size'],
                                edge_dim=2+self.z_int_dim,
                                dropout=dropout,
                                root_weight=False)
                for _ in range(config["num_blocks"])
            ]
        )
        self.traj_blocks = nn.ModuleList(
            [
                TransformerConv((config["n_path"], config["n_actor"]),
                                config["n_actor"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.embedding = nn.Linear(self.embedding_dim + self.z_act_dim, self.decoder_h_dim)

        self.hidden2pos = nn.Sequential(
            LinRes(self.decoder_h_dim, self.decoder_h_dim, norm="GN", ng=1),
            nn.Linear(self.decoder_h_dim, 2*config["num_preds"]),
        )

    def forward(self,
                actors,
                nodes, last_pos, last_pos_rel,
                Z_act, Z_int,
                actor_edge_index, node_edge_index, node_actor_edge_index,
                fut_pos_rel=None):
        batch = actors.size(0)
        if self.latent:
            actor_xz = torch.cat([actors, Z_act], -1)
        else:
            actor_xz = actors

        output = self.embedding(actor_xz)
        for block in self.path_blocks:
            nodes = block(nodes, node_edge_index)

        dists = last_pos[actor_edge_index[0]] - last_pos[actor_edge_index[1]]
        if self.latent:
            dists = torch.cat([Z_int, dists], -1)
        for i in range(len(self.int_blocks)):
            output = self.int_blocks[i](output, actor_edge_index, dists)
            output = self.traj_blocks[i]((nodes, output), node_actor_edge_index)
        output = torch.reshape(self.hidden2pos(output), [-1, self.config["num_preds"], 2])
        return output

class TrajPredNet(nn.Module):
    def __init__(self, config, latent=True):
        super(TrajPredNet, self).__init__()
        self.config = config
        self.decoder_h_dim = config["n_actor"]
        self.embedding_dim = config["n_actor"]

        self.latent = latent
        if latent:
            self.z_act_dim = config['act_continuous_size']
            self.z_int_dim = config['int_continuous_size']
        else:
            self.z_act_dim = 0
            self.z_int_dim = 0

        self.output_dim = 2
        self.num_preds = config["num_preds"]
        dropout = config["args"].dropout

        self.decoder = nn.LSTM(
            2*self.embedding_dim + self.z_act_dim,
            self.decoder_h_dim,
            1
            #dropout=dropout
        )
        self.num_blocks = config["num_blocks"]
        self.path_blocks = nn.ModuleList(
            [
                TransformerConv(config["n_path"],
                                config["n_path"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.int_blocks = nn.ModuleList(
            [
                TransformerConv(config["n_actor"],
                                config["n_actor"],
                                heads=config["n_head"],
                                #edge_dim=config['int_continuous_size'],
                                edge_dim=2+self.z_int_dim,
                                dropout=dropout,
                                root_weight=False)
                for _ in range(config["num_blocks"])
            ]
        )
        self.traj_blocks = nn.ModuleList(
            [
                TransformerConv((config["n_path"], config["n_actor"]),
                                config["n_actor"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )

        self.initial_h = nn.Linear(self.embedding_dim+self.z_act_dim, self.decoder_h_dim)
        self.hidden2pos = nn.Linear(self.embedding_dim, self.output_dim)
        self.spatial_embedding = nn.Linear(self.output_dim, self.embedding_dim)

    def forward(self,
                actors,
                nodes, last_pos, last_pos_rel,
                Z_act, Z_int,
                actor_edge_index, node_edge_index, node_actor_edge_index,
                fut_pos_rel=None):
        batch = actors.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)
        if self.latent:
            actor_xz = torch.cat([actors, Z_act], -1)
        else:
            actor_xz = actors
        initial_h = torch.unsqueeze(self.initial_h(actor_xz), 0)
        initial_c = torch.zeros(
            1, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (initial_h, initial_c)
        decoder_input = torch.cat([decoder_input, actor_xz], dim=1)
        decoder_input = decoder_input.view(1, batch, 2*self.embedding_dim + self.z_act_dim)

        for block in self.path_blocks:
            nodes = block(nodes, node_edge_index)

        pred_traj = []

        #assert(fut_pos_rel.size(0) == self.num_preds)
        for t in range(self.num_preds):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            output = output.view(-1, self.decoder_h_dim)
            dists = last_pos[actor_edge_index[0]] - last_pos[actor_edge_index[1]]
            if self.latent:
                dists = torch.cat([Z_int, dists], -1)
            for i in range(len(self.int_blocks)):
                output = self.int_blocks[i](output, actor_edge_index, dists)
                output = self.traj_blocks[i]((nodes, output), node_actor_edge_index)

            rel_pos = self.hidden2pos(output)
            pred_traj.append(rel_pos)

            #if fut_pos is None:
            curr_pos = rel_pos + last_pos
            curr_pos_rel = rel_pos
            #else:
            #    curr_pos = fut_pos_rel[i] + last_pos
            #    curr_pos_rel = fut_pos_rel[i]

            decoder_input = self.spatial_embedding(curr_pos_rel)
            decoder_input = torch.cat([decoder_input, actor_xz], dim=1)
            decoder_input = decoder_input.view(1, batch, 2*self.embedding_dim  + self.z_act_dim)
            last_pos = curr_pos

        return torch.stack(pred_traj, dim=1)
