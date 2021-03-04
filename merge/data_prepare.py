import torch
import numpy as np
import random
from processdataall import *
#some config
train = "/userhome/30/hjpan/NLP/SAMA/data/train.pt"
test = "/userhome/30/hjpan/NLP/SAMA/data/test.pt"
vocab_dir = "/userhome/30/hjpan/NLP/SAMA/data/vocab.pt"

# form SAMA .pt data to text file
train_dataset = torch.load(train)
test_dataset = torch.load(test)
gVocab = torch.load(vocab_dir)
# print("train: ",train_dataset)
# print("test: ",test_dataset)
# print("vocb: ",gVocab)
# get basic info
print("train: ",len(train_dataset),"test: ",len(test_dataset))
print("src_vocab: ",len(gVocab.src_vocab),"tar_vocab: ",len(gVocab.tgt_vocab),"skill_vocab",len(gVocab.skilltgt_vocab))
max_src= 0
max_tar= 0
for instance in train_dataset:
    if max_src< len(instance.src):
        max_src = len(instance.src)
    if max_tar < len(instance.tgt):
        max_tar = len(instance.tgt)
    
print(max_src,max_tar)
"""
train:  8610 test:  2047
src_vocab:  14289 tar_vocab:  18712 skill_vocab 8506
704 421
"""
data = train_dataset + test_dataset
train = data[0:8000] 
val = data[8000:9000] 
test = data[9000:-1]

print("train: ",len(train),"val: ",len(val),"test: ",len(test))
torch.save(train, '/userhome/30/hjpan/NLP/merge/data/train.pt')
torch.save(train, '/userhome/30/hjpan/NLP/merge/data/val.pt')
torch.save(train, '/userhome/30/hjpan/NLP/merge/data/test.pt')




