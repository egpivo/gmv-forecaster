from typing import List

import torch

from forecaster.training.model.layer import (
    CompressedInteractionNetwork,
    FeaturesEmbedding,
    FeaturesLinear,
    MultiLayerPerceptron,
)


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    Extreme Deep Factorization Machine Model for recommendation systems.

    References
    ----------
    - J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int,
        mlp_dims: List[int],
        dropout: float,
        cross_layer_sizes: List[int],
        split_half: bool = True,
    ) -> None:
        """
        Initialize the Extreme Deep Factorization Machine Model.

        Parameters
        ----------
        field_dims : List[int]
            The dimensions of the input fields.
        embed_dim : int
            The dimension of the embedding.
        mlp_dims :  List[int]
            The dimensions of the multi-layer perceptron.
        dropout : float
            The dropout probability.
        cross_layer_sizes : List[int]
            The sizes of cross layers.
        split_half : bool, optional
            Whether to split half of the cross layers, by default True
        """
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(
            len(field_dims), cross_layer_sizes, split_half
        )
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of size ``(batch_size, num_fields)``

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        embed_x = self.embedding(x)
        x = (
            self.linear(x)
            + self.cin(embed_x)
            + self.mlp(embed_x.view(-1, self.embed_output_dim))
        )
        return torch.sigmoid(x.squeeze(1))
