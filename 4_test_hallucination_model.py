import importlib
import sys

importlib.reload(sys)
#sys.setdefaultencoding('utf8')
import argparse
import torch
import torchtext.vocab as vocab
from tools import NLPTools
from torch.utils.data import DataLoader
from order2taskplan.model import order2taskplanModel
import time
from shutil import copyfile
import logging
import pathlib
from tools.plot import savePlot
print('PyTorch Version: ',torch.__version__)

parser = argparse.ArgumentParser(description='order2taskplan-pytorch')
parser.add_argument('--resume','-r',default=True,
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--epochs', default=200, type=int, metavar='E',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, metavar='SE',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--learning_rate', default=0.0002, type=float,
                    help='learning rate')
parser.add_argument('--fix_embed', default=True, type=bool,
                    help='fix pre-trained embedding')
parser.add_argument('--embedding_dim', default=300, type=int,
                    help='embedding dim')
parser.add_argument('--hidden_size', default=100, type=int,
                    help='RNN hidden size')
parser.add_argument('--num_layers', default=3, type=int,
                    help='number of RNN layers')
parser.add_argument('--dropout_rnn', default=0.2, type=float,
                    help='dropout rate of RNN')
parser.add_argument('--dropout_rnn_output', default=0.2, type=float,
                    help='dropout rate of RNN output')
parser.add_argument('--dropout_emb', default=0.2, type=float,
                    help='dropout rate of embedding layer')
parser.add_argument('--packing', default=False, type=bool,
                    help='packing padded rnn sequence')
parser.add_argument('--teacher_forcing_ratio', default=0.5, type=float,
                    help='teacher forcing ratio in decoding process')
parser.add_argument('--log_file', default='result/1_result.log',
                    help='log_file name to be saved')
parser.add_argument('--plot_file', default='result/1_scores.pdf',
                    help='plot_file name to be saved')
parser.add_argument('--model_file', default='checkpoint/2_best_model.pt',
                    help='model_file to be saved')
args = parser.parse_args()

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

USE_CUDA = torch.cuda.is_available()
print('USE_CUDA:',USE_CUDA)

# Load pretrained embedding (GloVe)
glove= vocab.GloVe(name='6B', dim=300)
print('Loaded', len(glove.itos), 'words')

pairs = {}
langs, _ = NLPTools.prepare_data('data/order-environment-taskplan-whole.txt', embedding=glove)
_, pairs['train'] = NLPTools.prepare_data('data/order-environment-taskplan-train.txt', embedding=glove)
_, pairs['test'] = NLPTools.prepare_data('data/order-environment-taskplan-test.txt', embedding=glove)

# Print an example pair
dataset={}
pairs_maxlen={}
loader={}
for state in ['train', 'test']:
    dataset[state], pairs_maxlen[state] = NLPTools.pad_pairs(langs=langs, pairs=pairs[state])
    dataset[state] = NLPTools.order2taskplan_Dataset(data = dataset[state])
loader['train'] = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
loader['test'] = DataLoader(dataset['test'], batch_size=1, shuffle=False)


max_seqlen=[]
for i,j in zip(pairs_maxlen['train'],pairs_maxlen['test']):
    max_seqlen.append(max(i,j))

if args.resume:
    log.info('[loading previous model...]')
    checkpoint = torch.load('checkpoint/best_model/2_best_model.pt')
    args = checkpoint['config']
    state_dict = checkpoint['state_dict']
    epoch_0 = checkpoint['epoch'] + 1
    model = order2taskplanModel(args=args, loader=loader,
                                langs=langs, max_seqlen=max_seqlen, embedding=glove, state_dict=state_dict)
    model.cuda()
    model.network.eval()
else:
    epoch_0 =1
    model = order2taskplanModel(args=args, loader=loader,
                                langs=langs, max_seqlen=max_seqlen, embedding=glove, state_dict=None)
    model.cuda()

    best_val_score = 0.0
    ppl_trains = []
    ppl_tests = []
    exact_matches = []
    f1_scores = []

test_input = pairs['test'][1]
print('-------------------------------------------------------')
#print('Order: ', test_input[0])
print('Objects: ', test_input[1])
print('Taskplan(gt): ', test_input[2])

print('-------------------------------------------------------')

dataset, pairs_maxlen = NLPTools.pad_pairs(langs=langs, pairs=[test_input])
x1 = torch.LongTensor(dataset['pairs_padded'][0][0]).cuda()
x2 = torch.LongTensor(dataset['pairs_padded'][0][1]).cuda()
y = torch.LongTensor(dataset['pairs_padded'][0][2][:5]).cuda()
print('human_behavior: ', [langs[2].itos[index.item()] for index in y])
x1_mask = torch.LongTensor(dataset['pairs_mask'][0][0]).cuda()
x2_mask = torch.LongTensor(dataset['pairs_mask'][0][1]).cuda()

x1_padded = torch.zeros(1,9).long().cuda()
x1_padded[0,:len(x1)] = x1
x2_padded = torch.zeros(1,18).long().cuda()
x2_padded[0,:len(x2)] = x2

x1_mask_padded = torch.ones(1,9).long().cuda()
x1_mask_padded[0,:len(x1_mask)] = x1_mask
x2_mask_padded = torch.ones(1,18).long().cuda()
x2_mask_padded[0,:len(x2_mask)] = x2_mask

outputs, output_indices = model.network.forward_for_hall(x1_padded,x2_padded,
                                                x1_mask_padded,x2_mask_padded,
                                                y=y.unsqueeze(0),INPUT1_TYPE='hall')
# HALLUCIANTION
output_txt = [langs[2].itos[index.item()] for index in output_indices[0]]
print(' '.join(output_txt))n