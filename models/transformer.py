# https://arxiv.org/abs/2411.00755
import math
# import timm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from timm.layers import PatchDropout
from models.attn_net_gated import *
# from patch_dropout import *
from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
from models.multihead_diffattn import *

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, class_token = False):   
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 5000,256
        position = torch.arange(0, max_len).unsqueeze(1) # 5000,1
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # 128
        pe[:, 0::2] = torch.sin(position * div_term)  # odd [sin, cos, sin, cos,......][5000,128]
        pe[:, 1::2] = torch.cos(position * div_term)  # even [5000,128]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        # trunc_normal_(self.pos_embed_for_scale, std=0.036)
         
    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class TransformerEncoderLayerWithDiffAttn(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, depth=1, activation="relu"):
        super(TransformerEncoderLayerWithDiffAttn, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        model_parallel_size = 1
        decoder_kv_attention_heads = nhead
        self.self_attn = MultiheadDiffAttn(model_parallel_size, decoder_kv_attention_heads, d_model, depth, nhead)

        # self.linear1 = Linear(d_model, dim_feedforward)
        # self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)

        # self.norm1 = LayerNorm(d_model)
        # self.norm2 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)

        # self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src2 = self.self_attn(src, src, src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2 = src.transpose(0,1) # tgt_len, bsz, embed_dim
        src2,_ = self.self_attn(src2,(None,None))  # bsz, tgt_len, embed_dim
        src2 = src2.transpose(0,1) # tgt_len, bsz, embed_dim
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''
    def __init__(self, d_model, h, d_ff, num_layers, dropout, class_token):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1, class_token=class_token)
        self.token = class_token

        # encode_layer = nn.TransformerEncoderLayer(
        #     d_model=self.d_model, 
        #     nhead=self.h, 
        #     dim_feedforward=self.d_ff, 
        #     dropout=self.dropout)
        # # self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)
        # self.transformer_encoder = nn.ModuleList([encode_layer for _ in range(num_layers)]) # this would result in ignorance in parameters calculation
        # self.transformer_encoder = nn.ModuleList([
        #     nn.TransformerEncoderLayer(
        #         d_model=self.d_model, 
        #         nhead=self.h, 
        #         dim_feedforward=self.d_ff, 
        #         dropout=self.dropout
        #     ) for _ in range(num_layers)
        # ])
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayerWithDiffAttn(
                d_model=self.d_model, 
                nhead=self.h, 
                dim_feedforward=self.d_ff, 
                dropout=self.dropout,
                depth=depth
            ) for depth in range(num_layers)
        ]) # customized transformer encoder class which can return attn weights

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if class_token else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6) # 0.036

    def forward(self, x):
        features = x
        out = features[2].permute(0, 2, 1) #[b,256,410]
        if self.token:
            out = torch.cat((self.cls_token.expand(out.shape[0], -1, -1),out),dim=1) # [B x (L+1) x d_model]
        out = self.pe(out)  # only add positional encoding 
        out = out.permute(1, 0, 2)

        # Pass features[2] through first 3 layers
        for i in range(4):
            out = self.transformer_encoder[i](out)

        cls_token = out[0] # pass the cls_token to following layers
 
        # Pass features[3] through the next 3 layers    
        out = features[3].permute(0, 2, 1) #[b,256,201]
        out = torch.cat((cls_token.unsqueeze(1), out), dim=1)  
        out = self.pe(out)  # re-add positional encoding
        out = out.permute(1, 0, 2)
        for i in range(4,8):
            out = self.transformer_encoder[i](out)
        cls_token = out[0]

        if self.num_layers > 8:
            # Pass features[-1] through the final 2 layers    
            out = features[-1].permute(0, 2, 1) #[b,256,183]
            out = torch.cat((cls_token.unsqueeze(1), out), dim=1)  
            out = self.pe(out)  # re-add positional encoding
            out = out.permute(1, 0, 2)
            for i in range(8,self.num_layers):
                out = self.transformer_encoder[i](out)
            cls_token = out[0]

        return cls_token, out

# 15 second model
class CTN(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes, class_token,\
                if_attn_gated_module):
        super(CTN, self).__init__()
        
        self.encoder = nn.Sequential( # downsampling factor = 20
            nn.Conv1d(12, 132, kernel_size=14, stride=3, padding=2, bias=False, groups=12),  #Layer 1: Output of chanllege: [b, 132, 2497]
            nn.BatchNorm1d(132),
            nn.ReLU(inplace=True),
            nn.Conv1d(132, 264, kernel_size=14, stride=3, padding=0, bias=False, groups=12),#Layer 2: Output of chanllege: [b, 264, 868]
            nn.BatchNorm1d(264),
            nn.ReLU(inplace=True),
            nn.Conv1d(264, d_model, kernel_size=10, stride=2, padding=0, bias=False, groups=12),#Layer 3: Output of chanllege: [b, 264, 410]
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False, groups=12),#Layer 4: Output of chanllege: [b, 264, 201]
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False, groups=12),#Layer 5: Output of chanllege: [b, 264, 192]
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False, groups=12),#Layer 6: Output of chanllege: [b, 264, 183]
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1, class_token=class_token)
        # attn_gated_module from https://github.com/mahmoodlab/HIPT/blob/780fafaed2e5b112bc1ed6e78852af1fe6714342/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py#L33 
        self.if_attn_gated_module = if_attn_gated_module

        if self.if_attn_gated_module:
            self.global_attn_pool = Attn_Net_Gated(L=184, D=184, dropout=0.25, n_classes=len(classes))
            self.global_rho = nn.Sequential(*[nn.Linear(184, 184), nn.ReLU(), nn.Dropout(0.25)])
            #exp 9
            # self.fc3 = nn.Linear(184, deepfeat_sz) #nb_demo?
            #exp 10
            # self.fc3 = nn.Linear(206, deepfeat_sz) #nb_demo?
            # self.dropout = nn.Dropout(p=0.25) 
            # # Define the second linear layer
            # self.fc5 = nn.Linear(deepfeat_sz, 1) 
            #exp 11
            self.wide_project = nn.Sequential(
                nn.Linear(1, 92),  # First linear layer
                nn.ReLU(),                          # ReLU activation
                nn.Linear(92, 184)  # Second linear layer to output
            )
            bag_classifiers = [nn.Linear(184, 1) for i in range(len(classes))] #use an indepdent linear layer to predict each class
            self.n_classes = len(classes)
            self.classifiers = nn.ModuleList(bag_classifiers)

        else:
            self.fc1 = nn.Linear(d_model, deepfeat_sz)
            self.fc2 = nn.Linear(deepfeat_sz+nb_feats+nb_demo, len(classes))
            self.dropout = nn.Dropout(dropout_rate)


        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        #self.apply(_weights_init)
            
    def forward(self, x, wide_feats):
        # z = self.encoder(x)          # encoded sequence is batch_sz x nb_ch x seq_len
        intermediate_embeds = []
        z = x
        for layer in self.encoder:
            z = layer(z)
            if isinstance(layer, nn.Conv1d):
                intermediate_embeds.append(z)
        # out = self.transformer(z)    # transformer output is batch_sz x d_model
        cls_token,out = self.transformer(intermediate_embeds)

        # attn_gated_module
        if self.if_attn_gated_module: ### Global Aggregation
            # exp9:
            # out = out.permute(1,2,0)
            # wide_feats = wide_feats.unsqueeze(2).expand(-1, -1, 184)
            # out = torch.cat([wide_feats, out], dim=1)
            # A_4096, h_4096 = self.global_attn_pool(out)  # N, n_classes ;  N,D,
            # A_4096 = torch.transpose(A_4096, 2, 1)
            # A_4096 = F.softmax(A_4096, dim=2) 
            # h_path = torch.bmm(A_4096, h_4096)
            # out = self.global_rho(h_path)

            # out = self.fc3(out)
            # out = F.relu(out)
            # out = self.dropout(out)
            # out = self.fc5(out).squeeze(2)
            # return out
            # exp10:
            # out = out.permute(1,2,0)
            # A_4096, h_4096 = self.global_attn_pool(out)  # N, n_classes ;  N,D,
            # A_4096 = torch.transpose(A_4096, 2, 1)
            # A_4096 = F.softmax(A_4096, dim=2) 
            # h_path = torch.bmm(A_4096, h_4096)
            # out = self.global_rho(h_path)  # B,n_classes, L(184/?)
            # wide_feats = wide_feats.unsqueeze(1).expand(-1, 27, -1) # B,n_classes(broadcast),22
            # out = torch.cat([wide_feats, out], dim=2) # B,n_classes, L+22=184+22=206
            # out = self.fc3(out)
            # out = F.relu(out)
            # out = self.dropout(out)
            # out = self.fc5(out).squeeze(2)
            # return out
            # exp11:
            out = out.permute(1,2,0)
            wide_feats = self.wide_project(wide_feats.unsqueeze(2)) # B,22->B,22,184
            out = torch.cat([wide_feats, out], dim=1) #B,264+22,184
            A_4096, h_4096 = self.global_attn_pool(out)  # N, n_classes ;  N,D,
            A_4096 = torch.transpose(A_4096, 2, 1)
            A_4096 = F.softmax(A_4096, dim=2) 
            h_path = torch.bmm(A_4096, h_4096) #B,n_classes,184(N_sequence)
            out = self.global_rho(h_path) 
            logits = torch.empty(h_path.shape[0], self.n_classes).float().to(out.device) # empty B,27
            for c in range(self.n_classes):
                logits[:, c] = self.classifiers[c](out[:,c,:]).squeeze(1)
            return logits
        else:
            cls_token = self.dropout(F.relu(self.fc1(cls_token)))
            cls_token = self.fc2(torch.cat([wide_feats, cls_token], dim=1))
            return cls_token                
