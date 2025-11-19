import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th

class Embedding(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Cell Oracle environment.

    This extractor implements the "Shared Embedding" stage described in the thesis:
    1. It takes the dictionary observation space with 'current_expression' and
       'target_expression'.
    2. It processes each 3000-dim vector through a separate but identical MLP
       (3000 -> 512 -> 128) to create a 128-dim embedding.
    3. It concatenates these two embeddings into a single 256-dim feature vector.
    4. This final 256-dim vector is then passed to the actor and critic heads.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        print(observation_space.spaces)

        n_genes_ = observation_space["current_expression"].shape[0]

        def create_embedding(n_genes=3000):
            return nn.Sequential(
                nn.Linear(n_genes, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU()
            )

        self.extractors = nn.ModuleDict({
            "current_expression": create_embedding(n_genes_),
            "target_expression": create_embedding(n_genes_)
        })
        self._last_mask = None


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._last_mask = observations.get("action_mask", None)
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        # Discrete action space: build logits and apply mask
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(latent_pi)  # [B, A]
            mask = self.features_extractor._last_mask.bool()
            large_neg = th.finfo(logits.dtype).min
            masked_logits = th.where(mask, logits, th.full_like(logits, large_neg))
            return self.action_dist.proba_distribution(action_logits=masked_logits)
        return super()._get_action_dist_from_latent(latent_pi, latent_sde)