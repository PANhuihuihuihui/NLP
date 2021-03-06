import sys
import os
import pathlib
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from vocab import *
import config



class SampleDataset(Dataset):

    def __init__(self, path): 
        # structuer
        # [src,skilltgt, skillnet ,tgt ,srcid ,skilltgtid, skillnetid,tgtid]
        self.dataPreprocesslist = torch.load(path)
        # Keep track of how many data points.
        self._len = len(self.dataPreprocesslist)

    def __getitem__(self, index):
        return {
            'x':  self.dataPreprocesslist[index].srcid,
            'len_OOV': self.dataPreprocesslist[index].srcid.count(0),
            'y': self.dataPreprocesslist[index].tgtid,
            'x_len': len(self.dataPreprocesslist[index].srcid),
            'y_len': len(self.dataPreprocesslist[index].tgtid)
        }

    def __len__(self):
        return self._len


def collate_fn(batch):

    def padding(indice, max_length, pad_idx=1):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item))
                      for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, len_OOV

