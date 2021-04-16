import torch
import numpy as np
import random
from processdataall import *


def write_samples(samples, file_path, opt='w'):
    """Write the samples into a file.
    """
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')

def combine(data,filename):
    final_list = []
    for instance in data:
        src = ' '.join([str(elem) for elem in instance.src])
        src_skill_pred = " ".join([str(elem) for elem in instance.skillnet])
        tgt = ' '.join([str(elem) for elem in instance.tgt]) 
        final_list.append(src+"<sep>"+src_skill_pred+"<sep>"+tgt)
        if len(final_list)<10:
            print(src+"<sep>"+src_skill_pred+"<sep>"+tgt)
    print(filename,len(final_list))
    write_samples(final_list,filename)


"""
train:  8610 test:  2047
src_vocab:  14289 tar_vocab:  18712 skill_vocab 8506
704 421
"""

# This dataset is comming from SAMA 2020
train = "/userhome/30/hjpan/NLP/SAMA/data/train.pt"
test = "/userhome/30/hjpan/NLP/SAMA/data/test.pt"
vocab_dir = "/userhome/30/hjpan/NLP/merge/data/vocab.pt"

train_dataset = torch.load(train)
test_dataset = torch.load(test)
gVocab = torch.load(vocab_dir)


data = train_dataset + test_dataset
train = data[0:8000] 
val = data[8000:8500] 
test = data[8500:-1]

combine(train,"train.txt")
combine(val,"val.txt")
combine(test,"test.txt")
