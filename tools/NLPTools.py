import torch
import numpy as np
import unicodedata
from torch.utils.data import DataLoader, Dataset

import re
PAD_token = 0
SOS_token = 1
EOS_token = 2



class Lang:
    def __init__(self, embedding):

        if embedding is None:
            self.stoi = {"PAD":PAD_token, "SOS":SOS_token, "EOS":EOS_token}
            self.itos = {PAD_token:"PAD", SOS_token:"SOS", EOS_token:"EOS"} # Only used for decoding
            self.n_words = 3
        else:
            self.stoi = embedding.stoi
            self.itos = embedding.itos
            self.n_words = len(embedding.itos)

    def index_words(self, sentence):
        for word in sentence.split(' '):
            #For debug:
            #if word not in self.stoi:
            #    print(sentence)
            self.index_word(word)

    def index_word(self, word):
        if word not in self.stoi:
            #For debug:
            #print('word:'+word+'-')
            self.stoi[word] = self.n_words
            self.itos[self.n_words] = word
            self.n_words += 1


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters
def normalize_string(s):

    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)


    return s

def read_langs(filename,embedding=None):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    input_lang1 = Lang(embedding=embedding)
    input_lang2 = Lang(embedding=embedding)
    output_lang = Lang(embedding=None)

    return input_lang1, input_lang2, output_lang, pairs

MAX_LENGTH = 50


def filter_pair(p):

    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and len(p[2].split(' ')) < MAX_LENGTH

def filter_pair2(p):

    p_splited0=p[0].split(' ')
    p_splited1=p[1].split(' ')
    p_splited2=p[2].split(' ')
    if '' in p_splited0:
        del p_splited0[p_splited0.index('')]
    if '' in p_splited1:
        del p_splited1[p_splited1.index('')]
    if '' in p_splited2:
        del p_splited2[p_splited2.index('')]
    p=[]
    p.append(' '.join(p_splited0))
    p.append(' '.join(p_splited1))
    p.append(' '.join(p_splited2))

    return p


def filter_pairs(pairs):
    pairs = [pair for pair in pairs if filter_pair(pair)]
    pairs = [filter_pair2(pair) for pair in pairs]
    return pairs


def prepare_data(filename, embedding=None):
    input_lang1, input_lang2, output_lang, pairs = read_langs(filename,embedding=embedding)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang1.index_words(pair[0])
        input_lang2.index_words(pair[1])
        output_lang.index_words(pair[2])

    return [input_lang1,input_lang2,output_lang], pairs



# Return a list of indexes, one for each word in the sentence
def indices_from_sentence(lang, sentence):
    return [lang.stoi[word] for word in sentence.split(' ')]

def tensors_from_sentence(lang, sentence, add_EOS=False):
    indices = indices_from_sentence(lang, sentence)
    if add_EOS:
        indices.append(EOS_token)
    indices = torch.LongTensor(indices)
    return indices

def tensors_from_pair(langs, pair):
    input1_tensor = tensors_from_sentence(langs[0], pair[0], add_EOS=False)
    input2_tensor = tensors_from_sentence(langs[1], pair[1], add_EOS=False)
    target_tensor = tensors_from_sentence(langs[2], pair[2], add_EOS=True)
    return [input1_tensor, input2_tensor, target_tensor]

def pad_pairs(langs, pairs):
    pairs = [tensors_from_pair(langs, pair) for pair in pairs]
    pairs_length = [[len(sequence) for sequence in pair] for pair in pairs]
    pairs_max_length = np.array(pairs_length).max(axis=0).tolist()

    pairs_padded = []
    pairs_mask = []
    for pair, pair_length in zip(pairs,pairs_length):
        pair_padded = []
        pair_mask = []
        for seq, seq_len, max_len in zip(pair,pair_length,pairs_max_length):
            if seq_len < max_len :
                seq = torch.cat([seq,torch.zeros(max_len-seq_len).long()],dim=0)
            seq_mask = torch.cat([torch.zeros(seq_len),torch.ones(max_len-seq_len)],dim=0).long()
            pair_padded.append(seq)
            pair_mask.append(seq_mask)
        pairs_padded.append(pair_padded)
        pairs_mask.append(pair_mask)

    data={}
    data['pairs_padded']=pairs_padded
    data['pairs_length']=pairs_length
    data['pairs_mask']=pairs_mask

    return data, pairs_max_length


class order2taskplan_Dataset(Dataset):
    def __init__(self, data):
        self.pairs_padded = data['pairs_padded']
        self.pairs_length = data['pairs_length']
        #self.pairs_max_length = data['pairs_max_length']
        self.pairs_mask = data['pairs_mask']

    def __getitem__(self, index):
        return self.pairs_padded[index], self.pairs_length[index], self.pairs_mask[index]

    def __len__(self):
        return len(self.pairs_length)