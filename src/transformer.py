# Written by: Inal A. Mashukov
# MIT License.


from builtins import hasattr

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import copy 
from typing import Tuple, Optional 


def _get_activation_fn(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Activation should be 'relu' or 'gelu', not {activation}. \n")
    

def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def compute_num_patches(lookback: int, patch_size: int, stride: int, pad_end: bool) -> int:
    if patch_size > lookback: 
        raise ValueError("Patch size cannot be greater than lookback length! \n")
    
    if pad_end:
        pad_len = (stride - (lookback - patch_size) % stride) % stride
        lookback += pad_len

    return (lookback - patch_size) // stride + 1 


# ---------------------------------------------------------
# BatchNorm Wrapper
# ---------------------------------------------------------
class TemporalBatchNorm(nn.Module):
    """
    Wraps BatchNorm1d to replace LayerNorm.
    
    Input shape: (Batch, Sequence, d_model)
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose to (Batch, d_model, Sequence)
        x = x.transpose(1, 2).contiguous()
        x = self.bn(x)
        # Transpose back to (Batch, Sequence, d_model)
        return x.transpose(1, 2).contiguous()


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------
class Patching(nn.Module):
    def __init__(self, patch_size: int, stride: int, pad_end: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride 
        self.pad_end = pad_end 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if x.ndim != 3:
            raise ValueError(f"Expected input X of shape (B, L, C), got {x.shape}. \n")
        
        x = x.permute(0, 2, 1)

        if self.pad_end:
            L = x.shape[-1]
            pad_len = (self.stride - (L - self.patch_size) % self.stride) % self.stride
            if pad_len > 0:
                x = F.pad(x, (0, pad_len), mode = 'replicate') 

        patches = x.unfold(dimension = -1, size = self.patch_size, step = self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()

        return patches
    

class TemporalEmbedding(nn.Module):
    def __init__(self, lookback: int, patch_size: int, stride: int, d_model: int, pad_end: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        self.lookback = lookback
        self.patch_size = patch_size 
        self.stride = stride 
        self.d_model = d_model 
        self.pad_end = pad_end 
    
        self.patcher = Patching(patch_size = patch_size, stride = stride, pad_end = pad_end)
        self.num_patches = compute_num_patches(lookback = lookback, patch_size = patch_size, stride = stride, pad_end = pad_end)
        
        self.W_p = nn.Linear(patch_size, d_model)
        self.W_pos = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        patches = self.patcher(x) 
        tokens = self.W_p(patches.permute(0, 1, 3, 2)) 
        tokens = tokens + self.W_pos
        tokens = tokens.reshape(B*C, self.num_patches, self.d_model)
        tokens = self.dropout(tokens)
        return tokens, patches 


class FeatureEmbedding(nn.Module):
    def __init__(self, lookback: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model 
        self.W_f = nn.Linear(lookback, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inverted = x.permute(0, 2, 1) 
        features = self.W_f(x_inverted) 
        features = self.dropout(features)
        return features


# ---------------------------------------------------------
# Attention
# ---------------------------------------------------------
class SDPA_MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, nheads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        self.d_model = d_model
        self.nheads = nheads
        self.head_dim = d_model // nheads
        self.dropout = dropout 

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape 

        q = self.q_proj(query).view(B, L_q, self.nheads, self.head_dim).transpose(1,2)
        k = self.k_proj(key).view(B, L_k, self.nheads, self.head_dim).transpose(1,2)
        v = self.v_proj(value).view(B, L_k, self.nheads, self.head_dim).transpose(1,2)

        dropout_p = self.dropout if self.training else 0.0 
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(out) 



# ---------------------------------------------------------
# Decoupled Specialists
# ---------------------------------------------------------
class TimeSpecialist(nn.Module):
    def __init__(self, d_model: int, nheads: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu") -> None:
        super().__init__()
        self.norm_t1 = TemporalBatchNorm(d_model)
        self.t_self_attn = SDPA_MultiheadAttention(d_model, nheads, dropout=dropout)
        
        self.norm_t2 = TemporalBatchNorm(d_model)
        self.t_cross_attn = SDPA_MultiheadAttention(d_model, nheads, dropout=dropout)
        
        self.norm_t3 = TemporalBatchNorm(d_model)
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def pre_cross(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_t = z_t + self.dropout(self.t_self_attn(self.norm_t1(z_t), self.norm_t1(z_t), self.norm_t1(z_t)))
        z_t_norm = self.norm_t2(z_t)
        return z_t, z_t_norm

    def post_cross(self, z_t: torch.Tensor, z_t_norm: torch.Tensor, z_f_norm: torch.Tensor, B: int, C: int, N: int) -> torch.Tensor:
        z_t_flat = z_t_norm.reshape(B, C * N, -1)
        t_cross = self.t_cross_attn(query=z_t_flat, key=z_f_norm, value=z_f_norm)
        z_t = z_t + self.dropout(t_cross.reshape(B * C, N, -1))
        z_t = z_t + self.dropout(self.ffn_t(self.norm_t3(z_t)))
        return z_t




class FeatureSpecialist(nn.Module):
    def __init__(self, d_model: int, nheads: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu") -> None:
        super().__init__()
        self.norm_f1 = TemporalBatchNorm(d_model)
        self.f_self_attn = SDPA_MultiheadAttention(d_model, nheads, dropout=dropout)
        
        self.norm_f2 = TemporalBatchNorm(d_model)
        self.f_cross_attn = SDPA_MultiheadAttention(d_model, nheads, dropout=dropout)
        
        self.norm_f3 = TemporalBatchNorm(d_model)
        self.ffn_f = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def pre_cross(self, z_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_f = z_f + self.dropout(self.f_self_attn(self.norm_f1(z_f), self.norm_f1(z_f), self.norm_f1(z_f)))
        z_f_norm = self.norm_f2(z_f)
        return z_f, z_f_norm

    def post_cross(self, z_f: torch.Tensor, z_f_norm: torch.Tensor, z_t_norm: torch.Tensor, B: int, C: int, N: int) -> torch.Tensor:
        z_t_flat = z_t_norm.reshape(B, C * N, -1)
        f_cross = self.f_cross_attn(query=z_f_norm, key=z_t_flat, value=z_t_flat)
        z_f = z_f + self.dropout(f_cross)
        z_f = z_f + self.dropout(self.ffn_f(self.norm_f3(z_f)))
        return z_f








class ModularOracleCoAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nheads: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu") -> None:
        super().__init__()
        self.time_specialist = TimeSpecialist(d_model, nheads, dim_feedforward, dropout, activation)
        self.feature_specialist = FeatureSpecialist(d_model, nheads, dim_feedforward, dropout, activation)

    def forward(self, z_t: torch.Tensor, z_f: torch.Tensor, B: int, C: int, N: int, 
                use_time: bool = True, use_feature: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if use_time:
            z_t, z_t_norm = self.time_specialist.pre_cross(z_t)
        else:
            z_t_norm = self.time_specialist.norm_t2(z_t)

        if use_feature:
            z_f, z_f_norm = self.feature_specialist.pre_cross(z_f)
        else:
            z_f_norm = self.feature_specialist.norm_f2(z_f)

        if use_time and use_feature:
            z_t_out = self.time_specialist.post_cross(z_t, z_t_norm, z_f_norm, B, C, N)
            z_f_out = self.feature_specialist.post_cross(z_f, z_f_norm, z_t_norm, B, C, N)
        elif use_time:
            z_t_out = self.time_specialist.post_cross(z_t, z_t_norm, z_f_norm, B, C, N)
            z_f_out = z_f
        elif use_feature:
            z_t_out = z_t
            z_f_out = self.feature_specialist.post_cross(z_f, z_f_norm, z_t_norm, B, C, N)
        else:
            z_t_out, z_f_out = z_t, z_f

        return z_t_out, z_f_out






# ---------------------------------------------------------
# ORACLE Architecture
# ---------------------------------------------------------
class ORACLE(nn.Module):
    def __init__(self, lookback: int, patch_size: int, stride: int, d_model: int, nheads: int,
                 num_layers: int, dim_feedforward: Optional[int] = None, dropout: float = 0.1,
                 num_classes: Optional[int] = None, forecast_len: Optional[int] = None,
                 pretrain: bool = False) -> None:
        
        super().__init__()
        self.pretrain = pretrain

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.temporal_embed = TemporalEmbedding(lookback, patch_size, stride, d_model, dropout=dropout)
        self.feature_embed = FeatureEmbedding(lookback, d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            ModularOracleCoAttentionBlock(d_model=d_model, nheads=nheads,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)])

        self.norm_t_final = TemporalBatchNorm(d_model)
        self.norm_f_final = TemporalBatchNorm(d_model)

        if self.pretrain:
            # Reconstruction Head maps hidden vector directly back to raw patch space
            self.head = nn.Linear(d_model, patch_size)
        elif num_classes is not None:
            self.is_classification = True
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.is_classification = False
            if forecast_len is None:
                raise ValueError("Must provide forecast_len for forecasting tasks. \n")
            # Joint Representation Head
            joint_dim = (self.temporal_embed.num_patches * d_model) + d_model
            self.head = nn.Linear(joint_dim, forecast_len) 

    def random_masking(self, z_t: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masks out a random subset of temporal patches across the batch and channels.
        Returns the masked embeddings and the boolean mask.
        """
        BC, N, D = z_t.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(BC, N, device=z_t.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        mask = torch.ones(BC, N, device=z_t.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        z_t = z_t * (1 - mask.unsqueeze(-1))
        return z_t, mask
        
    def forward(self, x: torch.Tensor, use_time: bool = True, use_feature: bool = True, mask_ratio: float = 0.40) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        
        z_t, patches = self.temporal_embed(x) 
        z_f = self.feature_embed(x)           
        N = patches.shape[-1]

        mask = None
        if self.pretrain and self.training:
            # Self-Supervised Masking Context Loop
            z_t, mask = self.random_masking(z_t, mask_ratio)

        for layer in self.layers:
            z_t, z_f = layer(z_t, z_f, B, C, N, use_time, use_feature)

        z_t = self.norm_t_final(z_t)
        z_f = self.norm_f_final(z_f)

        if self.pretrain:
            # Predict the raw values of the patches across every timestep slot
            pred_patches = self.head(z_t)  # (B*C, N, patch_size)
            target_patches = patches.permute(0, 1, 3, 2).reshape(B * C, N, -1)
            return pred_patches, target_patches, mask if mask is not None else torch.ones(B*C, N, device=x.device)

        # Zero out ablated streams to strictly control the joint head
        if not use_time: z_t = torch.zeros_like(z_t)
        if not use_feature: z_f = torch.zeros_like(z_f)

        if hasattr(self, 'is_classification') and self.is_classification:
            return self.head(z_f.mean(dim=1)) 
        else:
            z_t_flat = z_t.reshape(B, C, -1) 
            z_combined = torch.cat([z_t_flat, z_f], dim=-1) 
            return self.head(z_combined)       


