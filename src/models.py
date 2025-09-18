import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

activation_dict = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 3, activation: str = "silu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_dim + 1, hidden_dim)
        self.activation = activation_dict[activation]()
        layers = []
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_dict[activation]())
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, t], dim=-1)
        x = self.activation(self.input_layer(x))
        x = self.layers(x)
        x = self.output_layer(x)
        return x


def sinusoidal_time_embed(t: torch.Tensor, dim: int = 128, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    ang = t * freqs[None, :]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class SimpleModelSinTime(SimpleModel):
    def __init__(self, t_dim: int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.t_dim = t_dim
        self.time_embed = nn.Sequential(
            nn.Linear(t_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = sinusoidal_time_embed(t, self.t_dim)
        t = self.time_embed(t)
        x = self.activation(self.input_layer(x) + t)
        x = self.layers(x)
        x = self.output_layer(x)
        return x


class SimpleModelLabelled(SimpleModel):
    def __init__(self, num_classes: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.input_layer = nn.Linear(self.input_dim + 1 + num_classes, self.hidden_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, t, y], dim=-1)
        x = self.activation(self.input_layer(x))
        x = self.layers(x)
        x = self.output_layer(x)
        return x