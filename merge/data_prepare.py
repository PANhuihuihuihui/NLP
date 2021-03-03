import torch
import numpy as np
import random
from printEnvVariables import dataPreprocess
#some config
train = "train.pt"
test = "test.pt"
vocab_dir = "vocab.pt"

# form SAMA .pt data to text file
train_dataset = torch.load(train)
test_dataset = torch.load(test)
gVocab = torch.load(vocab_dir)
# print("train: ",train_dataset)
# print("test: ",test_dataset)
# print("vocb: ",gVocab)
for i in range(10):
    instance = random.choice(test)
    print(instance.src)
    print(instance.skilltgt)
    print(instance.skillnet)
    print(instance.tgt)
    print("___________")
    print(instance.srcid)
    print(instance.skilltgtid)
    print(instance.skillnetid)
    print(instance.tgtid)
gold_result = []





