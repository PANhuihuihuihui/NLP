import torch
import numpy as np
import random
from processdataall import *
#some config
train = "/userhome/30/hjpan/NLP/SAMA/data/train.pt"
test = "/userhome/30/hjpan/NLP/SAMA/data/test.pt"


# form SAMA .pt data to text file
train_dataset = torch.load(train)
test_dataset = torch.load(test)
# gVocab = torch.load(vocab_dir)
# print("train: ",train_dataset)
# print("test: ",test_dataset)
# print("vocb: ",gVocab)
# get basic info
# print("train: ",len(train_dataset),"test: ",len(test_dataset))
# print("src_vocab: ",len(gVocab.src_vocab),"tar_vocab: ",len(gVocab.tgt_vocab),"skill_vocab",len(gVocab.skilltgt_vocab))
# max_src= 0
# max_tar= 0
# for instance in train_dataset:
#     if max_src< len(instance.src):
#         max_src = len(instance.src)
#     if max_tar < len(instance.tgt):
#         max_tar = len(instance.tgt)
    
# print(max_src,max_tar)
"""
train:  8610 test:  2047
src_vocab:  14289 tar_vocab:  18712 skill_vocab 8506
704 421
"""
def write_samples(samples, file_path, opt='w'):
    """Write the samples into a file.
    """
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')

def combine(data,filename):
    "<eos>|||0,2,0 2,3,1 4,6,6 11,12,7 17,18,7 18,19,8"
    final_list = []
    for instance in data:
        length = len(instance.tgt)
        tgt = ' '.join([str(elem) for elem in instance.tgt])
        tgt.replace("<EOS>","")
        final_list.append(tgt+" <eos>|||")
        if len(final_list)<10:
            print(tgt+" <eos>|||")
    print(filename,len(final_list))
    write_samples(final_list,filename)
def src (data,filename):
    valuelist = ["__start_JD__ "," __end_JD__","__start_skill__ "," __end_skill__ ","__start_rel_skill__ "," __end_rel_skill__"]
    final_list = []
    for instance in data:
        src = ' '.join([str(elem) for elem in instance.src])
        
        src_skill_pred = " ".join([str(elem) for elem in instance.skillnet])
        src_skill = " ".join([str(elem) for elem in instance.skilltgt])
        src_skill.replace("<SEP>","")
        final_list.append(valuelist[0]+src+valuelist[1]+valuelist[2]+src_skill+valuelist[3]+valuelist[4]+src_skill_pred+valuelist[5])
        if len(final_list)<10:
            print(valuelist[0]+src+valuelist[1]+valuelist[2]+src_skill+valuelist[3]+valuelist[4]+src_skill_pred+valuelist[5])
    print(filename,len(final_list))
    write_samples(final_list,filename)

data = train_dataset + test_dataset
train = data[0:8000] 
val = data[8000:8500] 
test = data[8500:-1]

combine(train,"train.txt")
combine(val,"valid.txt")
src(train,"train.txt")
src(val,"valid.txt")
combine(test,"test.txt")
src(test,"test.txt")
# torch.save(train, '/userhome/30/hjpan/NLP/merge/data/train.pt')
# torch.save(train, '/userhome/30/hjpan/NLP/merge/data/val.pt')
# torch.save(train, '/userhome/30/hjpan/NLP/merge/data/test.pt')

"""
src_vocab:  14289 tar_vocab:  18712 skill_vocab 8506
704 421
train:  8000 val:  1000 test:  1656
"""

# check the oov and eos index
# key= list(gVocab.tgt_vocab.id2word.keys())
# print(key[0:100])
# key1 = list(gVocab.src_vocab.id2word.keys())
# print(key1[0:100])

# for i in range(50):
#     print(gVocab.src_vocab.id2word[str(i)] == gVocab.tgt_vocab.id2word[str(i)])