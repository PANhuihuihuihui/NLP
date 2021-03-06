

from typing import Optional

import torch

# General
hidden_size: int = 512
dec_hidden_size: Optional[int] = 512
embed_size: int = 512
pointer = True

#
model_name ="seq2seq"
encoder_save_name = "encoder_bi_512"
decoder_save_name ="decoder_ni_512"
attention_save_name = "attention_512"
reduce_state_save_name = "reduce"

# Data
max_vocab_size = 20000
embed_file: Optional[str] = None  # use pre-trained embeddings
source = 'train'    # use value: train or  big_samples 
data_path: str = '/userhome/30/hjpan/NLP/merge/data/{}.pt'.format(source)
val_data_path: Optional[str] = '/userhome/30/hjpan/NLP/merge/data/val.pt'
test_data_path: Optional[str] = '/userhome/30/hjpan/NLP/merge/data/test.pt'
max_src_len: int = 700  # exclusive of special tokens such as EOS
max_tgt_len: int = 400  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 500
enc_rnn_dropout: float = 0.4
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
losses_path = "loss/val_losses.pkl"
log_path = "log/"
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.005
lr_decay = 0.001
initial_accumulator_value = 0.1
epochs = 10
batch_size = 8
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