import unicodedata
import sys
import os
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf8')
import string
import re
import random
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchtext.vocab as vocab
import math
import numpy as np
import shutil
print('PyTorch Version: ',torch.__version__)

parser = argparse.ArgumentParser(description='order2taskplan-pytorch')
parser.add_argument('--resume','-r',default=False,
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--epochs', default=20000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

args = parser.parse_args()

#torch.backends.cudnn.benchmark = True
glove= vocab.GloVe(name='6B', dim=300)
wv_stoi = glove.stoi
wv_size = 300
wv_itos = glove.itos
wv_vectors = glove.vectors
print('Loaded', len(wv_vectors), 'words')


USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False
print('USE_CUDA:',USE_CUDA)
SOS_token_encoding = 273531
EOS_token_encoding = 306778

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name, embedding):
        self.name = name
        if embedding is None:
            self.stoi = {}
            self.itos = {0:"SOS", 1:"EOS"}
            self.n_words = 2
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

def read_langs(lang1, lang2, lang3):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s-%s.txt' % (lang1, lang2, lang3)).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    input_lang1 = Lang(lang1, embedding=glove)
    input_lang2 = Lang(lang2, embedding=glove)
    output_lang = Lang(lang3, embedding=None)

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


def prepare_data(lang1_name, lang2_name, lang3_name):
    input_lang1, input_lang2, output_lang, pairs = read_langs(lang1_name, lang2_name, lang3_name)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang1.index_words(pair[0])
        input_lang2.index_words(pair[1])
        output_lang.index_words(pair[2])

    return input_lang1, input_lang2, output_lang, pairs


#input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

input_lang1, input_lang2, output_lang, _ = prepare_data('order', 'environment','taskplan-whole')
_, _, _, pairs = prepare_data('order', 'environment','taskplan-train')
_, _, _, pairs_test = prepare_data('order','environment','taskplan-test')
# Print an example pair


class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, n_layers=1,eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.a2 = nn.Parameter(torch.ones(self.n_layers,1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(self.n_layers,1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)

        ln_out = ln_out * self.a2 + self.b2
        return ln_out

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size=400000, input_size=300, hidden_size=300, embedding=None, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, self.input_size)
        if embedding is not None:
            self.embedding.weight.data = embedding.vectors

        self.gru = nn.GRU(hidden_size, hidden_size,n_layers)
        #self.ln_hidden = LayerNormalization(hidden_size,n_layers)
        #self.ln_out = LayerNormalization(hidden_size,1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        x = embedded.clone()
        output, hidden = self.gru(embedded, hidden)
        output = output + x #residual connection

        #output = self.ln_out.forward(output)
        #hidden = self.ln_hidden.forward(hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()

        return hidden

class MergeEncoder(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super(MergeEncoder, self).__init__()
        self.input1_size = input1_size
        self.input2_size = input2_size
        self.combine = nn.Linear(input1_size+input2_size, output_size)

    def forward(self, input1, input2):
        '''
        output = Variable(torch.zeros(self.n_layers,1,self.input2_size+self.input1_size))
        if USE_CUDA:
            output = output.cuda()
        output = torch.cat((input1[0], input2[0]), 1)
        output = self.combine(output).unsqueeze(0)
        '''
        output = Variable(torch.zeros(1, self.input2_size + self.input1_size))
        if USE_CUDA:
            output = output.cuda()
        output = torch.cat((input1, input2), 1)
        output = self.combine(output)

        return output


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Define the layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn1 = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn2 = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        #self.ln_hidden = LayerNormalization(hidden_size, n_layers)
        #self.ln_feedforward = LayerNormalization(hidden_size, 1)
        #self.ln_out = LayerNormalization(output_size, 1)

    def forward(self, input, hidden, encoder1_outputs, encoder2_outputs):
        # Get the embedding of the current input word
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # Combine embedding and hidden state, run through attention layer

        attn_weights1 = F.softmax(self.attn1(torch.cat((embedded[0], hidden[0]), 1)))
        attn_weights2 = F.softmax(self.attn2(torch.cat((embedded[0], hidden[0]), 1)))

        # Multiply attention weights over encoder outputs
        attn_applied1 = torch.bmm(attn_weights1.unsqueeze(0), encoder1_outputs.unsqueeze(0))
        attn_applied2 = torch.bmm(attn_weights2.unsqueeze(0), encoder2_outputs.unsqueeze(0))

        # Combine embedding with attention output, run that through another layer, apply dropout
        output = torch.cat((embedded[0], attn_applied1[0],attn_applied2[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        #print(output)
        # Apply GRU to the output so far
        x = output.clone()
        output, hidden = self.gru(output, hidden)
        output = output + x
        # Apply log softmax to output
        output = F.log_softmax((self.out(output[0])))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights1, attn_weights2

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

# Return a list of indexes, one for each word in the sentence
def indices_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indices_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair):
    input1_variable = variable_from_sentence(input_lang1, pair[0])
    input2_variable = variable_from_sentence(input_lang2, pair[1])
    target_variable = variable_from_sentence(output_lang, pair[2])
    return (input1_variable, input2_variable, target_variable)


teacher_forcing_ratio = 0.5
clip = 5.0


def train(input1_variable,
          input2_variable,
          target_variable,
          encoder1,
          encoder2,
          mergeEncoderList,
          decoder,
          encoder1_optimizer,
          encoder2_optimizer,
          mergeEncoder_optimizerList,
          decoder_optimizer,
          criterion,
          max_length=MAX_LENGTH):
    for param in encoder1.parameters():
        param.requires_grad=True
    # Zero gradients of both optimizers
    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    for mergeEncoder_optimizer in mergeEncoder_optimizerList:
        mergeEncoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input1_length = input1_variable.size()[0]
    input2_length = input2_variable.size()[0]
    target_length = target_variable.size()[0]

    # Prepare input and output variables
    encoder1_hidden = encoder1.init_hidden()
    encoder2_hidden = encoder2.init_hidden()

    encoder1_outputs = Variable(torch.zeros(max_length, encoder1.hidden_size))
    encoder2_outputs = Variable(torch.zeros(max_length, encoder2.hidden_size))

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if USE_CUDA:
        encoder1_outputs = encoder1_outputs.cuda()
        encoder2_outputs = encoder2_outputs.cuda()
        decoder_input = decoder_input.cuda()

    # Run words through encoder
    for ei in range(input1_length):
        encoder1_output, encoder1_hidden = encoder1(input1_variable[ei], encoder1_hidden)
        encoder1_outputs[ei] = encoder1_output[0][0]
    for ei in range(input2_length):
        encoder2_output, encoder2_hidden = encoder2(input2_variable[ei], encoder2_hidden)
        encoder2_outputs[ei] = encoder2_output[0][0]


    # Copy last hidden state from encoder as decoder's first hidden state
    i=0
    decoder_hidden = Variable(torch.zeros(n_layers,1,hidden_size))
    if USE_CUDA:
        decoder_hidden=decoder_hidden.cuda()
    for mergeEncoder in mergeEncoderList:
        decoder_hidden[i] = mergeEncoder.forward(encoder1_hidden[i],encoder2_hidden[i])
        i+=1

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention1, decoder_attention2 = decoder.forward(decoder_input,
                                                                                decoder_hidden,
                                                                                encoder1_outputs,
                                                                                encoder2_outputs)
            loss += criterion(decoder_output[0].unsqueeze(0), target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention1, decoder_attention2 = decoder.forward(decoder_input,
                                                                                decoder_hidden,
                                                                                encoder1_outputs,
                                                                                encoder2_outputs)
            loss += criterion(decoder_output[0].unsqueeze(0), target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder1.parameters(), clip)
    torch.nn.utils.clip_grad_norm(encoder2.parameters(), clip)
    for mergeEncoder in mergeEncoderList:
        torch.nn.utils.clip_grad_norm(mergeEncoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder1_optimizer.step()
    encoder2_optimizer.step()
    for mergeEncoder_optimizer in mergeEncoder_optimizerList:
        mergeEncoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length



def train_hall(input1_variable,
          encoder1,
          encoder_hall,
          encoder_hall_optimizer,
          criterion_hall,
          max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    for param in encoder1.parameters():
        param.requires_grad=False

    encoder_hall_optimizer.zero_grad()
    loss_hall = 0

    # Get size of input and target sentences
    input1_length = input1_variable.size()[0]

    # Prepare input and output variables
    encoder1_hidden = encoder1.init_hidden()
    encoder_hall_hidden = encoder_hall.init_hidden()

    encoder1_outputs = Variable(torch.zeros(max_length, encoder1.hidden_size),requires_grad=False)
    encoder_hall_outputs = Variable(torch.zeros(max_length, encoder2.hidden_size))


    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if USE_CUDA:
        encoder1_outputs = encoder1_outputs.cuda()
        encoder_hall_outputs = encoder_hall_outputs.cuda()

    # Run words through encoder
    for ei in range(input1_length):
        encoder1_output, encoder1_hidden = encoder1(input1_variable[ei], encoder1_hidden)
        encoder1_outputs[ei] = encoder1_output[0][0]

        encoder_hall_output, encoder_hall_hidden = encoder_hall(input1_variable[ei], encoder_hall_hidden)
        encoder_hall_outputs[ei] = encoder_hall_output[0][0]
        #print encoder1_output[0].unsqueeze(0).size(), encoder_hall_output[0].unsqueeze(0).size()
        loss_hall += criterion_hall( input=encoder_hall_output[0].unsqueeze(0),target=encoder1_output[0].unsqueeze(0))

    # Backpropagation
    loss_hall.backward()
    torch.nn.utils.clip_grad_norm(encoder_hall.parameters(), clip)
    encoder_hall_optimizer.step()

    return loss_hall.data[0] / input1_length


import time
import math


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = float(s) / float(percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

# Initialize models (comment out to continue training)
hidden_size = 300
n_layers = 3
dropout_p = 0.05

encoder1 = EncoderRNN(len(wv_stoi), hidden_size, n_layers=n_layers)
encoder2 = EncoderRNN(len(wv_stoi), hidden_size, n_layers=n_layers)
encoder_hall = EncoderRNN(len(wv_stoi), hidden_size, n_layers=n_layers)
list(encoder1.parameters())[0].requires_grad=False
list(encoder2.parameters())[0].requires_grad=False
list(encoder_hall.parameters())[0].requires_grad=False

mergeEncoderList=[]
for i in range(n_layers):
    mergeEncoderList.append(MergeEncoder(hidden_size,hidden_size,hidden_size))
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)
print(mergeEncoderList)
print(decoder)

# Move models to GPU
if USE_CUDA:
    encoder1.cuda()
    encoder2.cuda()
    encoder_hall.cuda()
    for mergeEncoder in mergeEncoderList:
        mergeEncoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder1_optimizer = optim.Adam([{'params':encoder1.gru.parameters()}], lr=learning_rate)
encoder2_optimizer = optim.Adam([{'params':encoder2.gru.parameters()}], lr=learning_rate)
encoder_hall_optimizer = optim.Adam([{'params':encoder_hall.gru.parameters()}], lr=learning_rate)

mergeEncoder_optimizerList=[]
for mergeEncoder in mergeEncoderList:
    mergeEncoder_optimizerList.append(optim.Adam(mergeEncoder.parameters(), lr=learning_rate))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
criterion_hall = nn.MSELoss()


def evaluate(pair, max_length=MAX_LENGTH):
    training_pair = variables_from_pair(pair)
    input1_variable = training_pair[0]
    input2_variable = training_pair[1]
    target_variable = training_pair[2]
    target_length = target_variable.size()[0]
    #sentence1= pair[0]
    #sentence2= pair[1]
    #target_sentence=pair[2]

    #input1_variable = variable_from_sentence(input_lang1, sentence1)
    input1_length = input1_variable.size()[0]
    encoder1_hidden = encoder1.init_hidden()

    #input2_variable = variable_from_sentence(input_lang2, sentence2)
    input2_length = input2_variable.size()[0]
    encoder2_hidden = encoder2.init_hidden()

    #target_variable = variable_from_sentence(output_lang,target_sentence)

    encoder1_outputs = Variable(torch.zeros(max_length, encoder1.hidden_size))
    encoder2_outputs = Variable(torch.zeros(max_length, encoder2.hidden_size))

    if USE_CUDA:
        encoder1_outputs = encoder1_outputs.cuda()
        encoder2_outputs = encoder2_outputs.cuda()

    for ei in range(input1_length):
        encoder1_output, encoder1_hidden = encoder1.forward(input1_variable[ei], encoder1_hidden)
        encoder1_outputs[ei] = encoder1_outputs[ei] + encoder1_output[0][0]

    for ei in range(input2_length):
        encoder2_output, encoder2_hidden = encoder2.forward(input2_variable[ei], encoder2_hidden)
        encoder2_outputs[ei] = encoder2_outputs[ei] + encoder2_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    if USE_CUDA: decoder_input = decoder_input.cuda()
    # decoder_hidden = decoder.init_hidden()
    i=0
    decoder_hidden = Variable(torch.zeros(n_layers,1,hidden_size))
    if USE_CUDA:
        decoder_hidden=decoder_hidden.cuda()
    for mergeEncoder in mergeEncoderList:
        decoder_hidden[i] = mergeEncoder.forward(encoder1_hidden[i],encoder2_hidden[i])
        i+=1

    decoded_words = []
    decoder_attentions1 = torch.zeros(max_length, max_length)
    decoder_attentions2 = torch.zeros(max_length, max_length)

    loss=0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention1, decoder_attention2 = decoder.forward(decoder_input,
                                                                            decoder_hidden,
                                                                            encoder1_outputs,
                                                                            encoder2_outputs)
        decoder_attentions1[di] = decoder_attention1.data
        decoder_attentions2[di] = decoder_attention2.data
        if di < target_length:
            #print decoder_output[0].size()
            #print target_variable[di].size()
            loss += criterion(decoder_output[0].unsqueeze(0), target_variable[di])

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions1[:di + 1],decoder_attentions2[:di + 1], loss.data[0] / target_length

def test_input(sentence1,sentence2, max_length=MAX_LENGTH):

    input1_variable = variable_from_sentence(input_lang1, sentence1)
    input1_length = input1_variable.size()[0]
    encoder1_hidden = encoder1.init_hidden()

    input2_variable = variable_from_sentence(input_lang2, sentence2)
    input2_length = input2_variable.size()[0]
    encoder2_hidden = encoder2.init_hidden()

    encoder1_outputs = Variable(torch.zeros(max_length, encoder1.hidden_size))
    encoder2_outputs = Variable(torch.zeros(max_length, encoder2.hidden_size))

    if USE_CUDA:
        encoder1_outputs = encoder1_outputs.cuda()
        encoder2_outputs = encoder2_outputs.cuda()

    for ei in range(input1_length):
        encoder1_output, encoder1_hidden = encoder1.forward(input1_variable[ei], encoder1_hidden)
        encoder1_outputs[ei] = encoder1_outputs[ei] + encoder1_output[0][0]

    for ei in range(input2_length):
        encoder2_output, encoder2_hidden = encoder2.forward(input2_variable[ei], encoder2_hidden)
        encoder2_outputs[ei] = encoder2_outputs[ei] + encoder2_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    if USE_CUDA: decoder_input = decoder_input.cuda()
    # decoder_hidden = decoder.init_hidden()
    i=0
    decoder_hidden = Variable(torch.zeros(n_layers,1,hidden_size))
    if USE_CUDA:
        decoder_hidden=decoder_hidden.cuda()
    for mergeEncoder in mergeEncoderList:
        decoder_hidden[i] = mergeEncoder.forward(encoder1_hidden[i],encoder2_hidden[i])
        i+=1

    decoded_words = []
    decoder_attentions1 = torch.zeros(max_length, max_length)
    decoder_attentions2 = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention1, decoder_attention2 = decoder.forward(decoder_input,
                                                                            decoder_hidden,
                                                                            encoder1_outputs,
                                                                            encoder2_outputs)
        decoder_attentions1[di] = decoder_attention1.data
        decoder_attentions2[di] = decoder_attention2.data

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions1[:di + 1],decoder_attentions2[:di + 1]





def evaluate_randomly(pairs,n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0],pair[1])
        print('=', pair[2])
        output_words, attention1, attention2, loss = evaluate(pair)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        print('loss = ',loss, '  perplexity = ',math.exp(loss) )
        print('')

def calculate_test_loss(n=100):
    losses=[]
    for i in range(n):
        pair_test = random.choice(pairs_test)
        _,_,_,loss = evaluate(pair_test)
        losses.append(loss)
    loss_avg = sum(losses)/n
    return loss_avg

def save_checkpoint(state, is_best, filename='./checkpoint/checkpoint.pth.tar'):
    print("=> saving checkpoint ..")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoint/best_model/model_best.pth.tar')
    print('=> checkpoint saved.')

# Configuring training

plot_every = 100
print_every = 100
test_every = 100
save_every = 1000

# Keep track of time elapsed and running averages
start = time.time()
best_loss = 0
plot_losses = []
plot_losses_test = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every
print_loss_hall_total = 0 # Reset every print_every
plot_loss_hall_total = 0 # Reset every plot_every

if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint = torch.load('./checkpoint/checkpoint.pth.tar')
    args.start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    encoder1_optimizer.load_state_dict((checkpoint['encoder1_optimizer']))
    encoder2.load_state_dict(checkpoint['encoder2_state_dict'])
    encoder2_optimizer.load_state_dict((checkpoint['encoder2_optimizer']))
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder_optimizer.load_state_dict((checkpoint['decoder_optimizer']))

    for mergeEncoder, i in zip(mergeEncoderList, range(len(mergeEncoderList))):
        mergeEncoder.load_state_dict(checkpoint['mergeEncoder_state_dict_list'][i])
    for mergeEncoder_optimizer, i in zip(mergeEncoder_optimizerList, range(len(mergeEncoder_optimizerList))):
        mergeEncoder_optimizer.load_state_dict(checkpoint['mergeEncoder_optimizer_list'][i])
    del checkpoint
    print("=> loaded checkpoint")
else:
    print("=> Start training from scratch")


# Begin!
if not args.pretrained:
    for epoch in range(args.start_epoch, args.epochs+1):
        # Get training data for this cycle
        training_pair = variables_from_pair(random.choice(pairs))
        input1_variable = training_pair[0]
        input2_variable = training_pair[1]
        target_variable = training_pair[2]

        # Run the train function
        # Run the train function
        loss = train(input1_variable,
                     input2_variable,
                     target_variable,
                     encoder1,
                     encoder2,
                     mergeEncoderList,
                     decoder,
                     encoder1_optimizer,
                     encoder2_optimizer,
                     mergeEncoder_optimizerList,
                     decoder_optimizer,
                     criterion,
                     MAX_LENGTH)
        loss_hall = train_hall(input1_variable,
                               encoder1,
                               encoder_hall,
                               encoder_hall_optimizer,
                               criterion_hall,
                               MAX_LENGTH)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        # Keep track of loss
        print_loss_hall_total += loss_hall
        plot_loss_hall_total += loss_hall

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_hall_avg = print_loss_hall_total / print_every
            print_loss_total = 0
            print_loss_hall_total = 0
            print_loss_test_avg = calculate_test_loss(print_every)
            plot_losses_test.append(print_loss_test_avg)

            print_summary = '%s (%d %d%%) %.4f \t %4f \t %4f' % (
            time_since(start, float(epoch) / args.epochs), epoch, float(epoch) / args.epochs * 100, print_loss_avg,print_loss_test_avg, print_loss_hall_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % save_every ==0:

            is_best = print_loss_test_avg > best_loss
            best_loss = max(print_loss_test_avg, best_loss)
            model_dictionary={'epoch': epoch + 1,
                              'best_loss': best_loss,
                              'encoder1_state_dict':encoder1.state_dict(),
                              'encoder1_optimizer':encoder1_optimizer.state_dict(),
                              'encoder2_state_dict':encoder2.state_dict(),
                              'encoder2_optimizer':encoder2_optimizer.state_dict(),
                              'decoder_state_dict':decoder.state_dict(),
                              'decoder_optimizer':decoder_optimizer.state_dict(),
                              }
            mergeEncoder_state_dict_list = []
            for mergeEncoder in mergeEncoderList:
                mergeEncoder_state_dict_list.append(mergeEncoder.state_dict())
            model_dictionary['mergeEncoder_state_dict_list']=mergeEncoder_state_dict_list
            mergeEncoder_optimizer_state_dict_list = []
            for mergeEncoder_optimizer in mergeEncoder_optimizerList:
                mergeEncoder_optimizer_state_dict_list.append(mergeEncoder_optimizer.state_dict())
            model_dictionary['mergeEncoder_optimizer_list'] = mergeEncoder_optimizer_state_dict_list
            save_checkpoint(model_dictionary, is_best)


if not args.pretrained:

    if args.resume:
        file_save_loss = open('./result/plot_losses.txt', 'a')
        for loss in plot_losses:
            file_save_loss.write(str(loss) + '\n')
        file_save_loss.close()

        file_save_loss_test = open('./result/plot_losses_test.txt', 'a')
        for loss in plot_losses_test:
            file_save_loss_test.write(str(loss) + '\n')
        file_save_loss_test.close()

    else:
        file_save_loss = open('./result/plot_losses.txt','w')
        for loss in plot_losses:
            file_save_loss.write(str(loss)+'\n')
        file_save_loss.close()

        file_save_loss_test = open('./result/plot_losses_test.txt','w')
        for loss in plot_losses_test:
            file_save_loss_test.write(str(loss)+'\n')
        file_save_loss_test.close()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#%matplotlib inline

def show_plot(points1,points2):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.6) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points1,'r',label='train loss')
    ax.plot(points2,'b',label='test loss')
    plt.ylabel('nllloss')
    plt.xlabel('epoch (x100)')
    plt.title(('Model Loss'))
    ax.legend()
    plt.show()


show_plot(plot_losses,plot_losses_test)



evaluate_randomly(pairs)
#plt.matshow(attentions.numpy())

def show_attention(input_sentence1, input_sentence2, output_words, attentions1, attentions2):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(211)
    cax = ax.matshow(attentions1.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence1.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax = fig.add_subplot(212)
    cax = ax.matshow(attentions2.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence2.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluate_and_show_attention(sequence1,sequence2):
    output_words, attentions1, attentions2 = test_input(sequence1,sequence2)
    print('input =', sequence1, '\t', sequence2)
    print('output =', ' '.join(output_words))
    show_attention(sequence1, sequence2, output_words, attentions1, attentions2)

pair_test = random.choice(pairs_test)
evaluate_and_show_attention(pair_test[0],pair_test[1])