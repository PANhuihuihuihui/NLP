from typing import Optional

import torch

# General
enc_hidden_size: int = 400
dec_hidden_size: int = 400
embed_size: int = 100 # should not change if you load word2vec pretrain
pointer = True

#

model_name ="SkillO"
encoder_save_name = "encoder_bi_{}".format(enc_hidden_size)
decoder_save_name ="decoder_h_{}".format(enc_hidden_size)
attention_save_name = "attention_{}".format(enc_hidden_size)
reduce_state_save_name = "reduce_{}".format(enc_hidden_size)
vocab_save_name = "vocab_{}".format(enc_hidden_size)

# Data
max_vocab_size = 30000
embed_file: Optional[str] = "files/pretrained_w2v"  # use pre-trained embeddings
data_path: str = 'files/train.txt'
val_data_path: Optional[str] = 'files/val.txt'
test_data_path: Optional[str] = 'files/test.txt'
max_skill_len: int = 2  # exclusive of special tokens such as EOS
max_src_len: int = 300
max_tgt_len: int = 400  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 40
max_dec_steps: int = 500
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0.2
dec_rnn_dropout = 0.2
dec_out_dropout = 0.2


# Training
losses_path = "loss/val_losses.pkl"
log_path = "log/"
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.005
lr_decay = 0.001
initial_accumulator_value = 0.1
epochs = 10
batch_size = 4
coverage = True
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