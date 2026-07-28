
"""PyTorch model architecture used by the legacy behavior-cloning policy."""

from typing import Tuple

import math

import torch
import torch.nn as nn
from einops import rearrange


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class SimpleCNN(nn.ModuleList):
    def __init__(self, in_channels: int, input_shape: Tuple[int, int], out_channels: int):
        super().__init__()

        self.extend(
            [
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, stride=1),
                Flatten(),
            ]
        )

        with torch.no_grad():
            test_input = torch.zeros(1, in_channels, *input_shape)
            flattened_dim = self.forward(test_input).size(-1)

        self.extend(
            [
                nn.Linear(flattened_dim, out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x)
        return x

    def reset_parameters(self) -> None:
        for module in self:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The buffer is already on the model's device. Cast only its dtype so
        # adding it does not force AMP activations back to float32.
        positional = self.pe[:, : x.size(1)].to(dtype=x.dtype)
        return x + positional


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * (-math.log(10_000.0) / half_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = timesteps.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        heads = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, "b n (h d) -> b h n d", h=heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=heads)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = attention_scores.softmax(dim=-1)

        output = torch.matmul(attention, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        heads: int = 8,
        dim_head: int = 128,
    ):
        super().__init__()
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dim_feedforward=256,
        )
        self.cross_attention = CrossAttention(
            query_dim=dim,
            context_dim=cond_dim,
            heads=heads,
            dim_head=dim_head,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(x)
        return self.norm(x + self.cross_attention(x, condition))


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        sensor_dim: int = 7,
        depth_features_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        context_length: int = 5,
        action_horizon: int = 20,
    ):
        super().__init__()

        self.action_input_proj = nn.Linear(action_dim, hidden_dim)
        self.depth_projection = nn.Linear(depth_features_dim, hidden_dim)
        self.non_visual_obs_projection = nn.Linear(sensor_dim, hidden_dim)

        self.depth_encoder = SimpleCNN(
            in_channels=1,
            input_shape=(110, 210),
            out_channels=depth_features_dim,
        )

        self.time_embedding = SinusoidalPosEmb(hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, action_dim)

        self.decoder_position_embedding = SinusoidalPositionalEncoding(
            hidden_dim,
            max_len=action_horizon + 1,
        )
        self.encoder_position_embedding = SinusoidalPositionalEncoding(
            hidden_dim,
            max_len=(context_length * 2) + 1,
        )

    def forward(
        self,
        depth_images: torch.Tensor,
        non_visual_obs: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, context_length = non_visual_obs.shape[:2]

        noisy_action = self.action_input_proj(noisy_action)

        depth_images = depth_images.reshape(
            batch_size * context_length,
            depth_images.shape[-3],
            depth_images.shape[-2],
            depth_images.shape[-1],
        )
        depth_features = self.depth_encoder(depth_images)
        depth_features = depth_features.reshape(batch_size, context_length, -1)
        depth_features = self.depth_projection(depth_features)

        non_visual_features = self.non_visual_obs_projection(non_visual_obs)

        # Compute the sinusoidal time embedding in FP32, then cast it to the
        # activation dtype selected by autocast.
        time_token = self.time_embedding(t).to(dtype=noisy_action.dtype).unsqueeze(1)

        # Interleave the visual and non-visual context tokens without first
        # allocating a large zero tensor:
        # depth_0, nonvisual_0, depth_1, nonvisual_1, ..., time.
        encoder_input = torch.stack(
            (depth_features, non_visual_features),
            dim=2,
        ).flatten(1, 2)
        encoder_input = torch.cat((encoder_input, time_token), dim=1)
        encoder_input = self.encoder_position_embedding(encoder_input)

        decoder_input = torch.cat((time_token, noisy_action), dim=1)
        decoder_input = self.decoder_position_embedding(decoder_input)

        for block in self.transformer_blocks:
            decoder_input = block(decoder_input, encoder_input)

        output = self.output_proj(decoder_input)
        return output[:, 1:, :]
