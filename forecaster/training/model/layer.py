from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class CompressedInteractionNetwork(torch.nn.Module):
    """
    Compressed Interaction Network module.

    This module implements the Compressed Interaction Network (CIN) layer used in xDeepFM models.

    References
    ----------
    - Jianxun Lian, et al. "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems", 2018.

    Parameters
    ----------
    - input_dim (int): Dimensionality of input features.
    - cross_layer_sizes (List[int]): Sizes of cross layers.
    - split_half (bool, optional): Whether to split the cross layer size in half. Defaults to True.
    """

    def __init__(
        self, input_dim: int, cross_layer_sizes: List[int], split_half: bool = True
    ) -> None:
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i, cross_layer_size in enumerate(cross_layer_sizes):
            self.conv_layers.append(
                torch.nn.Conv1d(
                    input_dim * prev_dim,
                    cross_layer_size,
                    1,
                    stride=1,
                    dilation=1,
                    bias=True,
                )
            )
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CIN layer.

        Parameters
        ----------
        - x (torch.Tensor): Input tensor of size ``(batch_size, num_fields, embed_dim)``.

        Returns
        --------
        - torch.Tensor: Output tensor after passing through the CIN layer.
        """
        xs = []
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class FeaturesEmbedding(torch.nn.Module):
    """
    Features Embedding module.

    This module performs feature embedding for input features.

    Parameters
    ----------
    - field_dims (List[int]): Dimensions of input fields.
    - embed_dim (int): Dimensionality of embedding.
    """

    def __init__(self, field_dims: List[int], embed_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeaturesEmbedding layer.

        Parameters
        ----------
        - x (torch.Tensor): Input tensor of size ``(batch_size, num_fields)``.

        Returns
        --------
        - torch.Tensor: Output tensor after passing through the embedding layer.
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = x.long()
        return self.embedding(x)


class FeaturesLinear(torch.nn.Module):
    """
    Features Linear module.

    This module performs linear transformation for input features.

    Parameters
    ----------
    - field_dims (List[int]): Dimensions of input fields.
    - output_dim (int, optional): Dimensionality of output. Defaults to 1.
    """

    def __init__(self, field_dims: List[int], output_dim: int = 1) -> None:
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeaturesLinear layer.

        Parameters
        ----------
        - x (torch.Tensor): Input tensor of size ``(batch_size, num_fields)``.

        Returns
        --------
        - torch.Tensor: Output tensor after passing through the linear layer.
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = x.long()
        return torch.sum(self.fc(x), dim=1) + self.bias


class MultiLayerPerceptron(torch.nn.Module):
    """
    Multi-Layer Perceptron module.

    This module implements a multi-layer perceptron (MLP) with configurable hidden layers.

    Parameters
    ----------
    - input_dim (int): Dimensionality of input features.
    - embed_dims (List[int]): Dimensions of hidden layers.
    - dropout (float): Dropout rate.
    - output_layer (bool, optional): Whether to include an output layer. Defaults to True.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dims: List[int],
        dropout: float,
        output_layer: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.extend(
                [
                    torch.nn.Linear(input_dim, embed_dim),
                    torch.nn.BatchNorm1d(embed_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=dropout),
                ]
            )
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiLayerPerceptron.

        Parameters
        ----------
        - x (torch.Tensor): Input tensor.

        Returns
        --------
        - torch.Tensor: Output tensor after passing through the MLP.
        """
        return self.mlp(x)
