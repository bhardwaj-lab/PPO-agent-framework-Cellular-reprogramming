import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# This class is your "embedding network"
class GeneEmbeddingExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_dim: int = 128):
        # The 'features_dim' for the super().__init__ will be the total size
        # after we process and concatenate all our inputs.
        # Here, it will be embedding_dim * number_of_inputs.
        super().__init__(observation_space, features_dim=embedding_dim * len(observation_space.spaces))

        extractors = {}

        # For each key in your observation dictionary ("current_expression", "target_expression", etc.)
        for key, subspace in observation_space.spaces.items():
            # Create a simple neural network (an MLP) to embed this specific input
            # It takes the input (e.g., 3000 genes) and outputs a dense vector of size `embedding_dim`
            extractors[key] = nn.Sequential(
                nn.Linear(subspace.shape[0], 256),  # First hidden layer
                nn.ReLU(),  # Activation function
                nn.Linear(256, embedding_dim)  # Output embedding layer
            )

        # Store these individual networks in a ModuleDict
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # This method is called by the PPO agent.
        # It processes each part of the observation dictionary with its corresponding network.

        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            # Get the embedding for each input
            encoded_tensor_list.append(extractor(observations[key]))

        # Concatenate all the embeddings into a single vector
        return th.cat(encoded_tensor_list, dim=1)









    policy_kwargs = dict(
        # Tell SB3 to use your custom class instead of the default one
        features_extractor_class=GeneEmbeddingExtractor,

        # You can pass arguments to your custom class's __init__ method here
        features_extractor_kwargs=dict(embedding_dim=128),

        # This net_arch is now applied AFTER the custom extraction and concatenation
        net_arch=dict(pi=[64], vf=[64]),

        activation_fn=nn.Tanh
    )

    # Create the model
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        # ... other PPO parameters
    )