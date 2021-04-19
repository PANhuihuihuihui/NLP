import torch
import numpy as np
import random
from processdataall import *
from utils import skills
import thulac
thu1 = thulac.thulac(seg_only=False, model_path="/home/phj/software/models")  #设置模式为行分词模式

def write_samples(samples, file_path, opt='w'):
    """Write the samples into a file.
    """
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')
def process(data,filename,skill_list):
    thu1 = thulac.thulac(seg_only=False, model_path="/home/phj/software/models")  #设置模式为行分词模式
    final_list = []
    for instance in data:
        src = ' '.join([str(elem) for elem in instance.src])
        tgt = ' '.join([str(elem) for elem in instance.tgt])
        if len(instance.skilltgt) < 5:
            string = tgt.replace(" ",'')
            src_SAMA_extract = skills(string,thu1,skill_list,tgt)
            if len(src_SAMA_extract) > len(instance.skilltgt):
                src_SAMA_extract = " <SEP> ".join(src_SAMA_extract)
                src_SAMA_extract += " <EOS>"
            else:
                src_SAMA_extract= " ".join([str(elem) for elem in instance.skilltgt])
        else:
            src_SAMA_extract= " ".join([str(elem) for elem in instance.skilltgt])
        print(src_SAMA_extract)
        src_skill_pred = " ".join([str(elem) for elem in instance.skillnet])
        final_list.append(src+"<sep>"+src_SAMA_extract+"<sep>"+src_skill_pred+"<sep>"+tgt)
    print(filename,len(final_list))
    write_samples(final_list,filename)


"""
train:  8610 test:  2047
src_vocab:  14289 tar_vocab:  18712 skill_vocab 8506
704 421
"""

# This dataset is comming from SAMA 2020
# train = "/userhome/30/hjpan/NLP/SAMA/data/train.pt"
# test = "/userhome/30/hjpan/NLP/SAMA/data/test.pt"
# vocab_dir = "/userhome/30/hjpan/NLP/merge/data/vocab.pt"

train = "/home/phj/NLP/SAMA/dataset/train.pt"
test = "/home/phj/NLP/SAMA/dataset/test.pt"
vocab_dir = "/home/phj/NLP/SAMA/dataset/vocab.pt"

train_dataset = torch.load(train)
test_dataset = torch.load(test)
gVocab = torch.load(vocab_dir)

src_dic = gVocab.src_vocab
skill_dic = gVocab.skilltgt_vocab
tgt_dic = gVocab.tgt_vocab
skill_list = [str(elem).lower() for elem in skill_dic.word2id.keys()]
# print(skill_list)






data = train_dataset + test_dataset
train = data[0:8000] 
val = data[8000:8500] 
test = data[8500:-1]

process(train,"train.txt",skill_list)
process(val,"val.txt",skill_list)
process(test,"test.txt",skill_list)

