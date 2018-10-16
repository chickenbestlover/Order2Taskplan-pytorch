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
from tools.plot import savePlot_hall, savePlot_hall1, savePlot_hall2
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
parser.add_argument('--log_file', default='result/2_result.log',
                    help='log_file name to be saved')
parser.add_argument('--plot_file', default='result/2_scores.pdf',
                    help='plot_file name to be saved')
parser.add_argument('--model_file', default='checkpoint/2_checkpoint.pt',
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
    checkpoint = torch.load('checkpoint/best_model/1_best_model.pt')
    #args = checkpoint['config']
    state_dict = checkpoint['state_dict']
    epoch_0 = checkpoint['epoch'] + 1
    model = order2taskplanModel(args=args, loader=loader,
                                langs=langs, max_seqlen=max_seqlen, embedding=glove, state_dict=state_dict)
    model.cuda()

else:
    epoch_0 =1
    model = order2taskplanModel(args=args, loader=loader,
                                langs=langs, max_seqlen=max_seqlen, embedding=glove, state_dict=None)
    model.cuda()

try:

    best_val_score = 0.0
    loss_trains = []
    ppl_tests, exact_matches, f1_scores = {},{},{}
    for type in ['normal','normal+behavior','none','none+behavior','hall','hall+behavior']:
        ppl_tests[type] = []
        exact_matches[type] = []
        f1_scores[type] = []

    for epoch in range(epoch_0,epoch_0 + args.epochs):
        start_time = time.time()
        loss = model.train_hall()

        if epoch==epoch_0:
            ppl_test_normal, exact_match_normal, f1_score_normal = model.evaluate(INPUT1_TYPE="normal",recursive_len=0)
            ppl_test_normal_b, exact_match_normal_b, f1_score_normal_b = model.evaluate(INPUT1_TYPE="normal",recursive_len=6)

            ppl_test_none, exact_match_none, f1_score_none = model.evaluate(INPUT1_TYPE="none",recursive_len=0)
            ppl_test_none_b, exact_match_none_b, f1_score_none_b = model.evaluate(INPUT1_TYPE="none",recursive_len=6)


        ppl_test_hall, exact_match_hall, f1_score_hall = model.evaluate(INPUT1_TYPE="hall",recursive_len=0)
        ppl_test_hall_b, exact_match_hall_b, f1_score_hall_b = model.evaluate(INPUT1_TYPE="hall",recursive_len=6)


        elapsed_time = time.time() - start_time
        loss_trains.append(loss)
        ppl_tests['normal'].append(ppl_test_normal)
        ppl_tests['none'].append(ppl_test_none)
        ppl_tests['hall'].append(ppl_test_hall)
        exact_matches['normal'].append(exact_match_normal)
        exact_matches['none'].append(exact_match_none)
        exact_matches['hall'].append(exact_match_hall)
        f1_scores['normal'].append(f1_score_normal)
        f1_scores['none'].append(f1_score_none)
        f1_scores['hall'].append(f1_score_hall)

        ppl_tests['normal+behavior'].append(ppl_test_normal_b)
        ppl_tests['none+behavior'].append(ppl_test_none_b)
        ppl_tests['hall+behavior'].append(ppl_test_hall_b)
        exact_matches['normal+behavior'].append(exact_match_normal_b)
        exact_matches['none+behavior'].append(exact_match_none_b)
        exact_matches['hall+behavior'].append(exact_match_hall_b)
        f1_scores['normal+behavior'].append(f1_score_normal_b)
        f1_scores['none+behavior'].append(f1_score_none_b)
        f1_scores['hall+behavior'].append(f1_score_hall_b)

        log.info('|Epoch {:3d}| train MSE loss {:6.2f} | valid ppl {:6.2f}, F1 {:6.2f}, EM {:6.2f}| elapsed: {:3.0f} |'.format(
            epoch, loss, ppl_test_hall, f1_score_hall, exact_match_hall, elapsed_time))

        model.save(args.model_file,epoch)
        if f1_score_hall > best_val_score:
            best_val_score = f1_score_hall
            copyfile(
                args.model_file,'checkpoint/best_model/2_best_model.pt')
            log.info('[new best model saved.]')

        savePlot_hall(args,loss_trains, ppl_tests, exact_matches, f1_scores)
        savePlot_hall1(args,loss_trains, ppl_tests, exact_matches, f1_scores)
        savePlot_hall2(args,loss_trains, ppl_tests, exact_matches, f1_scores)



except KeyboardInterrupt:
    print('Exiting from training early')
