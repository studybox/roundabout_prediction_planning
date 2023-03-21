import gym
from functools import partial
import torch
import torch.nn as nn
from typing import Callable, Any, Dict, Mapping, Optional, Sequence, Tuple, TypeVar, Union, List, Type
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from layers import ActorNet, MapNet, A2M, M2M, M2A, A2A, P2A, graph_gather, LinRes


class GraphFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space:gym.spaces.Dict, config):
        super(GraphFeatureExtractor, self).__init__(observation_space, features_dim=1)
        self.observation_space = observation_space
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        self.p2a = P2A(config)

        self._features_dim = config["n_actor"]

    def forward(self, obs:Dict[str, Any]):
        actor_idcs = obs["actor_idcs"]
        actor_ctrs = obs["actor_ctrs"]
        batch_size = len(actor_idcs)

        actors =  obs["hists"].permute(0, 2, 1)
        #print(actors.size())
        actors = self.actor_net(actors)
        # contruct traffic features
        graph = graph_gather(obs["graphs"], actor_idcs)
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        lane_ctrs = torch.cat(node_ctrs, 0)
        #print(lane_ctrs.size(), actors.size(), actor_idcs)
        #raise
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)

        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs, obs["shapes"])
        actors = self.a2a(actors, actor_idcs, actor_ctrs, obs["state"], obs["codes"])
        # only ego vehicles
        graph["veh_path_idcs"] = torch.arange(0, graph["num_paths"]).unsqueeze(1)[obs["paths"]].unsqueeze(1).cuda()
        # batch["ego_path"] is [1,0,0,1,0,1,0]
        features = self.p2a(actors, nodes, graph)
        return features

class TargetGraphNet(nn.Module):
    def __init__(self, features_dim,
                       net_arch,
                       activation_fn,
                       device):
        super(TargetGraphNet, self).__init__()
        self.policy_net = LinRes(features_dim, features_dim, norm="GN", ng=1)
        self.value_net = LinRes(features_dim, features_dim, norm="GN", ng=1)
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

class ActorCriticGraphPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        config,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh
    ):
        super(ActorCriticGraphPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init = True, #TODO
            features_extractor_class=GraphFeatureExtractor,
            features_extractor_kwargs={"config":config}
        )
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TargetGraphNet(self.features_dim,
                                           net_arch=self.net_arch,
                                           activation_fn=self.activation_fn,
                                           device=self.device)

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def extract_features(self, obs:Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: (th.Tensor)
        :return: (th.Tensor)
        """
        assert self.features_extractor is not None, "No feature extractor was set"
        return self.features_extractor(obs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
