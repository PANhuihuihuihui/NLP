

from collections import Counter
import numpy as np
from untils import load_pretrain_emb,norm2one


class Vocab(object):
    PAD = 0 # padding
    SOS = 1 # start of sentence
    EOS = 2 # end of sentence
    UNK = 3 # unknown word

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None # TODO: add word2vec or BERT+BEP

    def add_words(self, words):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            words (list): The list of tokens to be added.
        """
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def load_embeddings_txt(self, file_path: str, dtype=np.float32) -> int:
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings
    def load_embedding_sama(self,path: str):
        embedding, vec_dim = load_pretrain_emb(path)
        ukn_count = 0
        scale = np.sqrt(3.0/vec_dim)
        vocab_size = len(self)
        emb = np.zeros([vocab_size,vec_dim], dtype='float32')
        for word,id in self.word2index.items():
            if word in embedding:
                emb[int(id), :] = norm2one(embedding[word])
            elif word.lower() in embedding:
                emb[int(id), :] = norm2one(embedding[word.lower()])
            elif word != "<PAD>":
                ukn_count += 1
                random_vec = np.random.uniform(-scale, scale, size=(vec_dim,)).astype('float32')
                emb[int(id), :] = random_vec
        self.embeddings = emb
        return vocab_size-ukn_count, ukn_count

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)
