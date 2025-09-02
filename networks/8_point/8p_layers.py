import torch
from torch import nn
import math


class CrossAttention(nn.Module):
    """
    ================================================================================
    Code adapted from:
        Repository: https://github.com/crockwell/rel_pose
        Paper: "The 8-Point Algorithm as an Inductive Bias for Relative Pose
               Prediction by ViTs"
        Authors: Christopher Rockwell, Justin Johnson, David F. Fouhey
    ================================================================================
    """

    def __init__(
        self,
        dim=768,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        cross_features=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # --> head_dim = 768 // 8 = 96
        self.scale = head_dim**-0.5  # --> scale = 1 / sqrt(96)

        self.qkv = nn.Linear(
            dim, dim * 3, bias=qkv_bias
        )  # Linear layer for query, key, value (dim -> 3*dim)
        self.attn_drop = nn.Dropout(attn_drop)  # Attention dropout
        self.proj_fundamental = nn.Linear(
            dim + int(6 * self.num_heads), dim
        )  # Output projection layer
        self.proj_drop = nn.Dropout(proj_drop)  # Output dropout

        self.cross_features = cross_features

    def forward(self, x1, x2, intrinsics=None):
        B, N, C = x1.shape  # B=batch_size, N=576, C=768
        # x1: [B, 576, 768]
        # x2: [B, 576, 768]

        # Compute query, key, value for x1
        qkv1 = (
            self.qkv(x1)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # qkv1: [3, B, 8, 576, 96]  --> 3 refers to (query, key, value), 8=num_heads, 96=head_dim
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        # q1, k1, v1: [B, 8, 576, 96]

        # Compute query, key, value for x2
        qkv2 = (
            self.qkv(x2)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # qkv2: [3, B, 8, 576, 96]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        # q2, k2, v2: [B, 8, 576, 96]

        # Cross-attention: attention of q2 over k1 and q1 over k2
        attn_1 = (q2 @ k1.transpose(-2, -1)) * self.scale
        # attn_1: [B, 8, 576, 576] --> attention weights for q2 on k1

        attn_2 = (q1 @ k2.transpose(-2, -1)) * self.scale
        # attn_2: [B, 8, 576, 576] --> attention weights for q1 on k2

        # Softmax over attention weights, resulting in refined attention maps
        attn_fundamental_1 = attn_1.softmax(dim=-1) * attn_1.softmax(dim=-2)
        # attn_fundamental_1: [B, 8, 576, 576]

        attn_fundamental_2 = attn_2.softmax(dim=-1) * attn_2.softmax(dim=-2)
        # attn_fundamental_2: [B, 8, 576, 576]

        # Positional encoding added to value vectors
        positional = PositionalEncoding(B, N, intrinsics=intrinsics).to(x1.device)
        # positional: [B, 576, 6] --> positional encoding with 6 additional dimensions
        v1 = torch.cat(
            [v1, positional.unsqueeze(1).repeat(1, self.num_heads, 1, 1)], dim=3
        )
        # v1: [B, 8, 576, 102] --> concatenated with positional encodings (96+6=102)

        v2 = torch.cat(
            [v2, positional.unsqueeze(1).repeat(1, self.num_heads, 1, 1)], dim=3
        )
        # v2: [B, 8, 576, 102] --> concatenated with positional encodings (96+6=102)

        if self.cross_features:
            # Cross attention feature merging
            fundamental_1 = (v2.transpose(-2, -1) @ attn_fundamental_1) @ v1
            # fundamental_1: [B, 102, 576] @ [B, 8, 576, 576] @ [B, 576, 102] --> [B, 8, 102, 102]

            fundamental_2 = (v1.transpose(-2, -1) @ attn_fundamental_2) @ v2
            # fundamental_2: [B, 102, 576] @ [B, 8, 576, 576] @ [B, 576, 102] --> [B, 8, 102, 102]
        else:
            # Self-attention feature merging
            fundamental_1 = (v1.transpose(-2, -1) @ attn_fundamental_1) @ v1
            # fundamental_1: [B, 102, 576] @ [B, 8, 576, 576] @ [B, 576, 102] --> [B, 8, 102, 102]

            fundamental_2 = (v2.transpose(-2, -1) @ attn_fundamental_2) @ v2
            # fundamental_2: [B, 102, 576] @ [B, 8, 576, 576] @ [B, 576, 102] --> [B, 8, 102, 102]

        # Reshaping and projection
        fundamental_1 = fundamental_1.reshape(
            B,
            int(C + 6 * self.num_heads),
            int((C + 6 * self.num_heads) / self.num_heads),
        ).transpose(-2, -1)
        # fundamental_1: [B, 102, 96] --> [B, 102 + 6 * 8, 96] reshaped to [B, 1024, 96]

        fundamental_2 = fundamental_2.reshape(
            B,
            int(C + 6 * self.num_heads),
            int((C + 6 * self.num_heads) / self.num_heads),
        ).transpose(-2, -1)
        # fundamental_2: [B, 102, 96] --> [B, 1024, 96] reshaped to [B, 1024, 96]

        # Projection back to the original dimension (768)
        fundamental_2 = self.proj_fundamental(fundamental_2)
        # fundamental_2: [B, 576, 768]

        fundamental_1 = self.proj_fundamental(fundamental_1)
        # fundamental_1: [B, 576, 768]

        return fundamental_2, fundamental_1


class FundamentalCombiner(nn.Module):
    """
    ================================================================================
    Code adapted from:
        Repository: https://github.com/crockwell/rel_pose
        Paper: "The 8-Point Algorithm as an Inductive Bias for Relative Pose
               Prediction by ViTs"
        Authors: Christopher Rockwell, Justin Johnson, David F. Fouhey
    ================================================================================
    """

    def __init__(
        self,
        embed_dim=768,  # Embedding dimension (default: 768)
        dropout_prob=0.0,  # Dropout probability (default: 0)
        num_heads=8,  # Number of attention heads (default: 8)
    ):
        super().__init__()
        # Apply dropout if the probability is greater than 0, otherwise use Identity (no-op)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        self.lnorm = nn.LayerNorm(embed_dim)  # LayerNorm over the embedding dimension
        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim, embed_dim * 2
            ),  # First MLP layer expands to 4x the embedding dimension
            nn.GELU(),  # Activation function
            nn.Linear(
                embed_dim * 2, embed_dim
            ),  # Second MLP layer reduces back to the embedding dimension
            nn.Dropout(dropout_prob),  # Apply dropout after the MLP
        )

    def forward(self, fundamental1, fundamental2):
        b_s = fundamental1.shape[0]  # Batch size (B)
        nf = fundamental1.shape[
            -1
        ]  # Number of features (nf), which is the embedding dimension (768)
        # fundamental1: [B, 576, 768]
        # fundamental2: [B, 576, 768]

        # Concatenate along a new dimension (dim=1), resulting in a tensor with two elements in the new axis
        fundamental_inter = torch.cat(
            [fundamental1.unsqueeze(1), fundamental2.unsqueeze(1)], dim=1
        )
        # fundamental_inter: [B, 2, 576, 768] --> 2 elements for fundamental1 and fundamental2 along the new axis
        # Reshape by flattening the second and third dimensions (concatenating 2*576 along the sequence length axis)
        fundamental = fundamental_inter.reshape(b_s, -1, nf)
        # fundamental: [B, 1152, 768] --> (2*576, 768), merging both fundamental tensors into a single sequence

        # Apply layer normalization, MLP, and residual connection with dropout
        fundamental = fundamental + self.dropout(self.mlp(self.lnorm(fundamental)))
        # fundamental: [B, 1152, 768] --> final output shape remains the same after transformation

        return fundamental


class PoseRegressor(nn.Module):
    """
    ================================================================================
    Code adapted from:
        Repository: https://github.com/crockwell/rel_pose
        Paper: "The 8-Point Algorithm as an Inductive Bias for Relative Pose
               Prediction by ViTs"
        Authors: Christopher Rockwell, Justin Johnson, David F. Fouhey
    ================================================================================


    Pose regressor module that takes a flattened fundamental matrix input and predicts a 6-dimensional pose vector.

    Args:
        embed_dim (int): Embedding dimension of the input, default is 768.
    """

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        # Fully connected layers for pose regression
        self.fc = nn.Linear(
            embed_dim * (embed_dim // num_heads + 6) * 2, 64
        )  # First layer reduces the flattened input to 64 units
        self.act = nn.GELU()  # GELU activation function
        self.fc2 = nn.Linear(64, 32)  # Second layer reduces to 32 units
        self.fc3 = nn.Linear(32, 6)  # Final layer outputs 6-dimensional pose vector

    def forward(self, fundamental: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pose regression.

        Args:
            fundamental (torch.Tensor): Input tensor of shape [B, 204, embed_dim],
                                        representing a batch of fundamental matrices.

        Returns:
            torch.Tensor: Output pose tensor of shape [B, 6], representing the predicted 6D pose.
        """
        # Flatten the input tensor across dimensions 1 and 2 (spatial dimensions)
        pose = self.fc(fundamental.flatten(1))  # [B, 204*embed_dim] -> [B, 64]

        # Apply GELU activation
        pose = self.act(pose)  # [B, 64]

        # Pass through second fully connected layer
        pose = self.fc2(pose)  # [B, 64] -> [B, 32]

        # Apply activation again
        pose = self.act(pose)  # [B, 32]

        # Pass through final fully connected layer to get 6D pose
        pose = self.fc3(pose)  # [B, 32] -> [B, 6]

        return pose  # [B, 6]


class PositionalEncoding(nn.Module):
    def __init__(self, height: int, width: int, d_model: int):
        """
        Initialize positional encoding for 2D features.

        Args:
            height: Height of the feature map
            width: Width of the feature map
            d_model: Number of dimensions for the positional encoding
        """
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        pe = torch.zeros(d_model, height, width)
        d_model = d_model // 2

        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)

        pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        self.register_buffer("pe", pe)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Positional encoding of shape [d_model, height, width]
        """
        return self.pe
