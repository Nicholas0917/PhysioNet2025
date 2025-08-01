import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import CTN

# Parameters
debug = False
patience = 10
batch_size = 32 * torch.cuda.device_count() #64
tr_batch_size = 24 * torch.cuda.device_count() # or 32*3gpus
window = 15*500
dropout_rate = 0.2
deepfeat_sz = 64
padding = 'zero' # 'zero', 'qrs', or 'none'
fs = 500
filter_bandwidth = [3, 45]
polarity_check = []
model_name = 'my_model'

# Transformer parameters
d_model = 288   # embedding size 256/276 for depthwise of 12 leads
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 12  # number of encoding layers, originally be 8
class_token = True # adding classification token by Xiaoya
if_attn_gated_module = False # adding attention gated module for channel by Xiaoya

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ch_idx = 1
nb_demo = 2
nb_feats = 20
thrs_per_class = False
class_weights = None
tr_n_wins = 5 # here originally to be 1 
val_n_wins = 10
te_n_wins = 20

classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Step 1: Simulate 12-lead ECG --------
sampling_rate = 500
duration = 10 # seconds
n_samples = sampling_rate * duration

# Generate 12 synthetic ECG signals (independent for demo purposes)
ecg_12lead = np.array([
    nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate)
    for _ in range(12)
])  # Shape: (12, 5000)

# Normalize signals (optional)
ecg_12lead = (ecg_12lead - ecg_12lead.mean(axis=1, keepdims=True)) / ecg_12lead.std(axis=1, keepdims=True)

# -------- Step 2: Format for ResNet (batch_size=1, channels=12, length=5000) --------
input_tensor = torch.tensor(ecg_12lead, dtype=torch.float32).unsqueeze(0)

# inp_t, lbl_t = inp_t.float().to(device), lbl_t.float().to(device) # for multiple window

# Get (normalized) demographic data and append to top (normalized) features
# age_t = torch.FloatTensor((get_age(hdr[13])[None].T - data_df.Age.mean()) / data_df.Age.std()) # age normalized 
# sex_t = torch.FloatTensor([1. if h.find('Female') >= 0. else 0 for h in hdr[14]])[None].T
# wide_feats = torch.cat([age_t, sex_t, feats_t.squeeze(1).float()], dim=1).to(device)

wide_feats = torch.cat([
    torch.tensor([1]).float(),
    torch.tensor([0]).float(),
    torch.zeros(nb_feats).squeeze(0).float()], dim=0).to(device).unsqueeze(0) 
 
model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes, class_token, \
                if_attn_gated_module).to(device)

output = model(input_tensor, wide_feats)
print("Model output shape:", output.shape)
