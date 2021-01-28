

from typing import Optional

import torch

# General
hidden_size: int = 512
dec_hidden_size: Optional[int] = 512
embed_size: int = 512
pointer = True

# Data
max_vocab_size = 30000
embed_file: Optional[str] = None  # use pre-trained embeddings
source = 'samples'    # use value: train or  big_samples 
data_path: str = 'files/{}.txt'.format(source)
val_data_path: Optional[str] = 'files/dev.txt'
test_data_path: Optional[str] = 'files/test.txt'
max_src_len: int = 400  # exclusive of special tokens such as EOS
max_tgt_len: int = 200  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 500
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
losses_path = "loss/"
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 20
batch_size = 32
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1





# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 2000