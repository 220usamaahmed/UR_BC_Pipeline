"""PyTorch model architecture used by the legacy behavior-cloning policy."""

import math

import torch
import torch.nn as nn
from einops import rearrange


class Flatten(nn.Module):
    """Flatten a contiguous range of tensor dimensions."""

    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'


class SimpleCNN(nn.ModuleList):
    """CNN encoder used by the trained policy checkpoint."""

    def __init__(self, in_channels, input_shape, out_channels) -> None:
        super().__init__()
        self.extend([
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=1),
            Flatten(),
        ])

        with torch.no_grad():
            sample = torch.zeros(1, in_channels, *input_shape)
            flattened_dim = self.forward(sample).size(-1)
        self.extend([
            nn.Linear(flattened_dim, out_channels),
            nn.ReLU(True),
        ])
        self.reset_parameters()

    def forward(self, input_tensor):
        for module in self:
            input_tensor = module(input_tensor)
        return input_tensor

    def reset_parameters(self):
        for module in self:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    nn.init.calculate_gain('relu'),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position encodings to a sequence."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', encoding.unsqueeze(0))

    def forward(self, input_tensor):
        return (
            input_tensor
            + self.pe[:, :input_tensor.size(1), :].to(input_tensor.device)
        )


class SinusoidalPosEmb(nn.Module):
    """Encode diffusion timesteps with sinusoidal embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embedding = torch.exp(
            torch.arange(half_dim, device=device)
            * -(torch.log(torch.tensor(10000.0)) / half_dim)
        )
        embedding = timesteps[:, None] * embedding[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class CrossAttention(nn.Module):
    """Multi-head cross-attention between action and observation tokens."""

    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, input_tensor, context):
        heads = self.heads
        query = rearrange(
            self.to_q(input_tensor),
            'b n_decoder (h d) -> b h n_decoder d',
            h=heads,
        )
        key = rearrange(
            self.to_k(context),
            'b n_context (h d) -> b h n_context d',
            h=heads,
        )
        value = rearrange(
            self.to_v(context),
            'b n_context (h d) -> b h n_context d',
            h=heads,
        )
        attention = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale
        ).softmax(dim=-1)
        output = torch.matmul(attention, value)
        output = rearrange(
            output,
            'b h n_decoder d -> b n_decoder (h d)',
        )
        return self.to_out(output)


class DiffusionTransformerBlock(nn.Module):
    """Self-attention followed by observation cross-attention."""

    def __init__(self, dim, cond_dim, heads=8, dim_head=128):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dim_feedforward=256,
        )
        self.cross_attn = CrossAttention(dim, cond_dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_tensor, condition):
        input_tensor = self.attn(input_tensor)
        return self.norm(
            input_tensor + self.cross_attn(input_tensor, condition)
        )


class ConditionalDiffusionModel(nn.Module):
    """Conditional flow-matching policy loaded by the ROS node."""

    def __init__(
        self,
        action_dim=7,
        output_dim=10,
        sensor_dim=7,
        depth_features_dim=512,
        hidden_dim=256,
        num_layers=2,
    ):
        super().__init__()
        # output_dim is retained for checkpoint/API compatibility.
        del output_dim
        self.action_input_proj = nn.Linear(action_dim, hidden_dim)
        self.depth_projection = nn.Linear(depth_features_dim, hidden_dim)
        self.non_visual_obs_projection = nn.Linear(sensor_dim, hidden_dim)
        self.non_visual_dropout = nn.Dropout(0.2)
        self.depth_encoder = SimpleCNN(
            1,
            (110, 210),
            depth_features_dim,
        )
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(hidden_dim))
        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        self.decoder_position_embedding = SinusoidalPositionalEncoding(
            hidden_dim,
            max_len=21,
        )
        self.encoder_position_embedding = SinusoidalPositionalEncoding(
            hidden_dim,
            max_len=16,
        )
        self.inference_step = 0

    def forward(self, depth_images, non_visual_obs, noisy_action, timestep):
        batch_size = non_visual_obs.shape[0]
        context_length = non_visual_obs.shape[1]
        noisy_action = self.action_input_proj(
            noisy_action.to(
                device=depth_images.device,
                dtype=torch.float32,
            )
        )
        depth_images = depth_images.reshape(
            batch_size * context_length,
            depth_images.shape[-3],
            depth_images.shape[-2],
            depth_images.shape[-1],
        )
        depth_features = self.depth_encoder(depth_images.to(torch.float32))
        depth_features = depth_features.reshape(
            batch_size,
            context_length,
            -1,
        )
        depth_features = self.depth_projection(
            depth_features.to(torch.float32)
        )
        non_visual_obs = self.non_visual_obs_projection(
            non_visual_obs.to(torch.float32)
        )
        non_visual_obs = self.non_visual_dropout(non_visual_obs)
        timestep = self.time_mlp(timestep.to(torch.float32))
        timestep = timestep.unsqueeze(1).repeat(
            non_visual_obs.shape[0],
            1,
            1,
        )

        encoder_input = torch.zeros(
            (
                batch_size,
                (context_length * 2) + 1,
                non_visual_obs.shape[-1],
            ),
            dtype=torch.float32,
            device=non_visual_obs.device,
        )
        encoder_input[:, 0:-2:2, :] = depth_features
        encoder_input[:, 1:-1:2, :] = non_visual_obs
        encoder_input[:, -1, :] = timestep.squeeze(1)
        encoder_input = self.encoder_position_embedding(encoder_input)

        decoder_input = torch.cat((timestep, noisy_action), dim=1)
        decoder_input = self.decoder_position_embedding(decoder_input)
        for block in self.transformer_blocks:
            decoder_input = block(decoder_input, encoder_input)

        output = self.output_proj(decoder_input)[:, 1:, :]
        self.inference_step += 1
        return output
