# https://arxiv.org/abs/2411.00755
import math
import torch
import torch.nn.functional as F
from torch import nn

# from kernel.rotary import apply_rotary_emb
# from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from models.rms_norm import RMSNorm


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def nan_to_num(input_tensor, nan=0.0, posinf=None, neginf=None):
    if posinf is not None:
        input_tensor[input_tensor == float('inf')] = posinf
    else:
        input_tensor[input_tensor == float('inf')] = float('1e18')  # Use a large number instead of inf
    if neginf is not None:
        input_tensor[input_tensor == float('-inf')] = neginf
    else:
        input_tensor[input_tensor == float('-inf')] = float('-1e18')  # Use a large negative number instead of -inf
    input_tensor[torch.isnan(input_tensor)] = nan
    return input_tensor

class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        model_parallel_size,
        decoder_kv_attention_heads,
        embed_dim,
        depth,
        num_heads,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads // model_parallel_size
        self.num_kv_heads = decoder_kv_attention_heads // model_parallel_size if decoder_kv_attention_heads is not None else num_heads // model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)    # bsz,7500,256
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim) # bsz, 7500, 2*8, 256/8/2
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)# bsz, 7500, 2*8(or num_kv_heads), 256/8/2
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)# bsz, 7500, 8(or num_kv_heads), 256/8

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2) # bsz, 16, 7500, 16
        k = repeat_kv(k.transpose(1, 2), self.n_rep)# bsz, 16, 7500, 16 no change here when n_heads is same as num_kv_heads
        v = repeat_kv(v.transpose(1, 2), self.n_rep)# bsz, 8, 7500, 32   no change here
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) # bsz, 16, 7500, 7500
        # if attn_mask is None:
        #     attn_mask = torch.triu(   
        #         torch.zeros([tgt_len, src_len])
        #         .float()
        #         .fill_(float("-inf"))
        #         .type_as(attn_weights),
        #         1 + offset,
        #     )  # The upper triangular part of the matrix is defined as the elements on and above the diagonal.
        # attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = nan_to_num(attn_weights)
        # attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1] # [bdz, 8, 7500,7500]
        
        attn = torch.matmul(attn_weights, v) # [bdz, 8, 7500,32]
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim) # [bdz, 7500, 256]

        attn = self.out_proj(attn)
        return attn, attn_weights

# import argparse

# def main():
#     parser = argparse.ArgumentParser(description='Example of model configuration.')
#     parser.add_argument('--decoder_kv_attention_heads', type=int, default=8,
#                         help='Number of attention heads in the decoder key-value mechanism.')
#     parser.add_argument('--model_parallel_size', type=int, default=1,
#                         help='Size of the model parallelism.')

#     args = parser.parse_args()
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     attention = MultiheadDiffAttn(1,8,256,12,8)
#     attention = attention.to(device)
#     x=  torch.randn([2,256,7500]).transpose(1,2).to(device) 

#     max_seq_length = 7500  # maximum length of the sequence
#     embedding_dim = 256   # embedding dimension used in rotary embeddings
#     cos, sin = get_relative_positional_encodings(max_seq_length, embedding_dim // 2// 8) # assuming rotary embedding applies to half the dimension
#     cos = cos.to(device)
#     sin = sin.to(device)
#     y = attention(x,(cos,sin))
#     print(y.shape)

def get_relative_positional_encodings(max_len, dim):
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pos_embedding = position * div_term
    cos = torch.cos(pos_embedding)
    sin = torch.sin(pos_embedding)
    return cos, sin

# if __name__ == '__main__':
#     main()


    