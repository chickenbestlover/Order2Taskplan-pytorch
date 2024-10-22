import logging
import math
from collections import Counter

import torch
import torch.optim as optim
from order2taskplan.criterion import masked_cross_entropy
from order2taskplan.seqs2seq import seqs2seq
from order2taskplan.seq2seq import seq2seq
from order2taskplan.selfAttnDiscriminator import selfAttnDiscriminator
from order2taskplan.layers import AttentionRNNDiscriminator
from torch.autograd import Variable
import pathlib
logger = logging.getLogger(__name__)


class order2taskplanModel(object):
    def __init__(self, args, loader, langs,max_seqlen, embedding=None, state_dict=None):
        self.args = args
        self.updates = state_dict['updates'] if state_dict else 0
        self.langs = langs
        self.loader=loader
        # build network
        self.network = seqs2seq(args,
                                input_embedding=embedding,
                                output_lang=langs[2],
                                max_seqlen=max_seqlen)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'],strict=False)

        # build optimizer for seq2seq
        hall_params = list(map(id, self.network.hall_net.parameters()))
        #hall_params=[]
        base_params = filter(lambda p: id(p) not in hall_params,
                             self.network.parameters())
        base_params = [p for p in base_params if p.requires_grad]
        self.optimizer = optim.Adamax(base_params, args.learning_rate, weight_decay=0)
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        # build optimizer for hall
        hall_params = [p for p in self.network.hall_net.parameters() if p.requires_grad]
        self.optimizer_hall = optim.Adamax(hall_params, args.learning_rate, weight_decay=0)
        #if state_dict:
        #    self.optimizer_hall.load_state_dict(state_dict['optimizer_hall'])



        self.criterion = torch.nn.MSELoss()



        # calculate total number of parameters in the network
        num_params = sum(p.data.numel() for p in self.network.parameters()
                         if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr())
        print ("{} parameters".format(num_params))

    def cuda(self):
        self.network.cuda()

    def train(self):
        self.network.train()
        losses = []
        for i,(pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader['train']):
            # Prepare input & target
            x1 = Variable(pairs_padded[0]).cuda()
            x2 = Variable(pairs_padded[1]).cuda()
            y = Variable(pairs_padded[2]).cuda()
            x1_mask = Variable(pairs_mask[0]).cuda()
            x2_mask = Variable(pairs_mask[1]).cuda()
            y_mask = Variable(pairs_mask[2]).cuda()
            y_length = Variable(pairs_len[2]).cuda()
            # Forward propagation
            outputs, outputs_indices = self.network.forward(x1=x1, x2=x2, x1_mask=x1_mask, x2_mask=x2_mask, y=y)
            # Calculate cross-entropy loss
            # outputs: [ batch x seq_len x #vocab_output ]
            # y: [ Batch x seq_len ]
            loss = masked_cross_entropy(logits=outputs,
                                        target=y,
                                        target_mask=y_mask,
                                        target_length=y_length)
            # loss = self.criterion.forward(input=outputs.view(-1,outputs.size(2)),target=y.view(-1))

            # Calculate gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.network.parameters(),max_norm=5.0)
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
            # Add loss
            losses.append(loss.cpu().data[0])

        loss = sum(losses) / len(self.loader['train'])
        return math.exp(loss)

    def train_hall(self):
        self.network.train()

        for p in self.network.parameters():
            p.requires_grad = False
        for p in self.network.hall_net.parameters():
            p.requires_grad = True

        losses = []
        for i,(pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader['train']):
            # Prepare input & target
            x1 = Variable(pairs_padded[0]).cuda()
            x2 = Variable(pairs_padded[1]).cuda()
            x1_mask = Variable(pairs_mask[0]).cuda()
            x2_mask = Variable(pairs_mask[1]).cuda()
            # Forward propagation
            input1_hiddens_hall = self.network.forward_hall(x2=x2, x2_mask=x2_mask)# [batch * len_o * hidden_size]
            input1_hiddens = self.network.forward_enc1(x1=x1, x1_mask=x1_mask)# [batch * len_o * hidden_size]
            #print('output:',input1_hiddens_hall.size())
            #print('target:',input1_hiddens.size())

            # Calculate MSE loss
            loss = self.criterion(input =input1_hiddens_hall.view(input1_hiddens.size(0),-1),
                                  target = input1_hiddens.view(input1_hiddens.size(0),-1))

            # Calculate gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.network.hall_net.parameters(), max_norm=5.0)
            # Update weights
            self.optimizer_hall.step()
            self.optimizer_hall.zero_grad()
            self.updates += 1
            # Add loss
            losses.append(loss.cpu().data[0])

        loss = sum(losses) / len(self.loader['train'])
        return loss



    def evaluate(self,INPUT1_TYPE="normal",recursive_len=0,DATASET_TYPE='test'):
        self.network.eval()
        losses = []
        exact_matches = []
        f1_scores = []
        for i, (pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader[DATASET_TYPE]):
            # Prepare input & target
            x1 = Variable(pairs_padded[0]).cuda()
            x2 = Variable(pairs_padded[1]).cuda()
            y = Variable(pairs_padded[2]).cuda()
            x1_mask = Variable(pairs_mask[0]).cuda()
            x2_mask = Variable(pairs_mask[1]).cuda()
            y_mask = Variable(pairs_mask[2]).cuda()
            y_length = Variable(pairs_len[2]).cuda()
            # Forward propagation
            outputs, outputs_indices = self.network.forward(x1=x1, x2=x2, x1_mask=x1_mask, x2_mask=x2_mask, y=y,
                                                            INPUT1_TYPE=INPUT1_TYPE,recursive_len=recursive_len)
            # outputs: [ batch x seq_len x #vocab_output ]
            # outputs_indices = [ batch x seq_len ]
            # y: [ Batch x seq_len ]
            # Calculate cross-entropy loss
            loss = masked_cross_entropy(logits=outputs,
                                        target=y,
                                        target_mask=y_mask,
                                        target_length=y_length)
            # loss = self.criterion.forward(input=outputs.view(-1,outputs.size(2)),target=y.view(-1))
            losses.append(loss.cpu().data[0])
            # exact_match: True or False (Bool)
            # f1_score: 0 ~ 1 (Float)
            exact_match, f1_score = self.score(outputs_indices,y)
            exact_matches.append(exact_match)
            f1_scores.append(f1_score)
            #output_sentence_tokens = self.indices2text(outputs_indices)
            #target_sentence_tokens = self.indices2text(y)

        loss = sum(losses) / len(self.loader[DATASET_TYPE])
        exact_match = sum(exact_matches) / len(self.loader[DATASET_TYPE])
        f1_score = sum(f1_scores) / len(self.loader[DATASET_TYPE])
        # return perplexity
        return math.exp(loss), exact_match, f1_score

    def indices2text(self,indices):
        #print(indices.size())
        indices = indices.squeeze(0).cpu().data.numpy() # [ seq_len ]
        #indices = indices.squeeze(axis=1)
        sentence = []
        for index in indices:
            #print(index.shape)
            word_text = self.langs[2].itos[index]
            sentence.append(word_text)

        return sentence


    def score(self, output_tokens, answer_tokens):
        output_tokens = output_tokens.squeeze(0).cpu().data.numpy().tolist()
        answer_tokens = answer_tokens.squeeze(0).cpu().data.numpy().tolist()
        exact_match = output_tokens==answer_tokens
        if exact_match:
            return 1.0, 1.0
        common = Counter(output_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0
        precision = 1. * num_same / len(output_tokens)
        recall = 1. * num_same / len(answer_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return exact_match, f1

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_hall': self.optimizer_hall.state_dict(),
                'updates': self.updates
            },
            'config': self.args,
            'epoch': epoch
        }
        try:
            pathlib.Path('checkpoint/best_model').mkdir(parents=True, exist_ok=True)
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')




class taskplan2taskplanModel(object):
    def __init__(self, args, loader, langs,max_seqlen, embedding=None, state_dict=None):
        self.args = args
        self.updates = state_dict['updates'] if state_dict else 0
        self.langs = langs
        self.loader=loader
        # build network
        self.network = seq2seq(args,
                                input_embedding=embedding,
                                output_lang=langs[2],
                                max_seqlen=max_seqlen)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'],strict=False)

        # build optimizer for seq2seq
        base_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(base_params, args.learning_rate, weight_decay=0)
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])


        # calculate total number of parameters in the network
        num_params = sum(p.data.numel() for p in self.network.parameters()
                         if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr())
        print ("{} parameters".format(num_params))

    def cuda(self):
        self.network.cuda()

    def train(self):
        self.network.train()
        losses = []
        for i,(pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader['train']):
            # Prepare input & target
            y = Variable(pairs_padded[2]).cuda()
            y_mask= Variable(pairs_mask[2]).cuda()
            y_length = Variable(pairs_len[2]).cuda()
            # Forward propagation
            outputs, outputs_indices = self.network.forward(y_in=y, y_in_mask=y_mask, y_out=y)
            # Calculate cross-entropy loss
            # outputs: [ batch x seq_len x #vocab_output ]
            # y: [ Batch x seq_len ]
            loss = masked_cross_entropy(logits=outputs,
                                        target=y,
                                        target_mask=y_mask,
                                        target_length=y_length)
            # loss = self.criterion.forward(input=outputs.view(-1,outputs.size(2)),target=y.view(-1))

            # Calculate gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.network.parameters(),max_norm=5.0)
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
            # Add loss
            losses.append(loss.cpu().data[0])

        loss = sum(losses) / len(self.loader['train'])
        return math.exp(loss)

    def evaluate(self,INPUT1_TYPE="normal"):
        self.network.eval()
        losses = []
        exact_matches = []
        f1_scores = []
        for i, (pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader['test']):
            # Prepare input & target
            y = Variable(pairs_padded[2]).cuda()
            y_mask = Variable(pairs_mask[2]).cuda()
            y_length = Variable(pairs_len[2]).cuda()
            # Forward propagation
            outputs, outputs_indices = self.network.forward(y_in=y, y_in_mask=y_mask, y_out=y)
            # outputs: [ batch x seq_len x #vocab_output ]
            # outputs_indices = [ batch x seq_len ]
            # y: [ Batch x seq_len ]
            # Calculate cross-entropy loss
            loss = masked_cross_entropy(logits=outputs,
                                        target=y,
                                        target_mask=y_mask,
                                        target_length=y_length)
            # loss = self.criterion.forward(input=outputs.view(-1,outputs.size(2)),target=y.view(-1))
            losses.append(loss.cpu().data[0])
            # exact_match: True or False (Bool)
            # f1_score: 0 ~ 1 (Float)
            exact_match, f1_score = self.score(outputs_indices,y)
            exact_matches.append(exact_match)
            f1_scores.append(f1_score)
            #output_sentence_tokens = self.indices2text(outputs_indices)
            #target_sentence_tokens = self.indices2text(y)

        loss = sum(losses) / len(self.loader['test'])
        exact_match = sum(exact_matches) / len(self.loader['test'])
        f1_score = sum(f1_scores) / len(self.loader['test'])
        # return perplexity
        return math.exp(loss), exact_match, f1_score

    def indices2text(self,indices):
        #print(indices.size())
        indices = indices.squeeze(0).cpu().data.numpy() # [ seq_len ]
        #indices = indices.squeeze(axis=1)
        sentence = []
        for index in indices:
            #print(index.shape)
            word_text = self.langs[2].itos[index]
            sentence.append(word_text)

        return sentence


    def score(self, output_tokens, answer_tokens):
        output_tokens = output_tokens.squeeze(0).cpu().data.numpy().tolist()
        answer_tokens = answer_tokens.squeeze(0).cpu().data.numpy().tolist()
        exact_match = output_tokens==answer_tokens
        if exact_match:
            return 1.0, 1.0
        common = Counter(output_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0
        precision = 1. * num_same / len(output_tokens)
        recall = 1. * num_same / len(answer_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return exact_match, f1

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.args,
            'epoch': epoch
        }
        try:
            pathlib.Path('checkpoint/best_model').mkdir(parents=True, exist_ok=True)
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')




class order2taskplanAdversarialModel(object):
    def __init__(self, args, loader, langs, max_seqlen, autoencoder_state_dict=None, gen_state_dict=None, embedding=None, state_dict=None):
        self.args = args
        self.updates = state_dict['updates'] if state_dict else 0
        self.langs = langs
        self.loader=loader
        # build network
        self.autoencoder = seq2seq(args,
                                input_embedding=embedding,
                                output_lang=langs[2],
                                max_seqlen=max_seqlen)
        self.generator = seqs2seq(args,
                                input_embedding=embedding,
                                output_lang=langs[2],
                                max_seqlen=max_seqlen)

        latent_vector_size = args.latent_vector_size if args.latent_vector_size is not None else 0
        decoder_input_size = args.hidden_size*5 + latent_vector_size
        self.discriminator = AttentionRNNDiscriminator(input_size=decoder_input_size,
                                                             hidden_size=args.hidden_size,
                                                             embedding_dim = embedding.vectors.size(1),
                                                             num_layers=args.num_layers,
                                                             max_seqlen=max_seqlen,
                                                             lang=langs[2],
                                                             padding_idx=0,
                                                             dropout_rate=0,
                                                             dropout_output=False,
                                                             packing=False,
                                                             latent_vector_size=latent_vector_size)

        # build optimizer for generator
        base_params = [p for p in self.generator.parameters() if p.requires_grad]
        self.optimizer_g = optim.Adamax(base_params, args.learning_rate, weight_decay=0)
        # build optimizer for discriminator
        base_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        self.optimizer_d = optim.Adamax(base_params, args.learning_rate, weight_decay=0)

        '''
        Loading Pre-trained model
        '''
        if autoencoder_state_dict:
            new_state = set(self.autoencoder.state_dict().keys())
            for k in list(autoencoder_state_dict['network'].keys()):
                if k not in new_state:
                    del autoencoder_state_dict['network'][k]
            self.autoencoder.load_state_dict(autoencoder_state_dict['network'],strict=False)
        if gen_state_dict:
            new_state = set(self.generator.state_dict().keys())
            for k in list(gen_state_dict['network'].keys()):
                if k not in new_state:
                    del gen_state_dict['network'][k]
            self.generator.load_state_dict(gen_state_dict['network'], strict=False)


        if state_dict:
            new_state = set(self.generator.state_dict().keys())
            for k in list(state_dict['generator'].keys()):
                if k not in new_state:
                    del state_dict['generator'][k]
            self.generator.load_state_dict(state_dict['generator'],strict=False)
            new_state = set(self.discriminator.state_dict().keys())
            for k in list(state_dict['discriminator'].keys()):
                if k not in new_state:
                    del state_dict['discriminator'][k]
            self.discriminator.load_state_dict(state_dict['discriminator'], strict=False)
            self.optimizer_g.load_state_dict(state_dict['optimizer_g'])
            self.optimizer_d.load_state_dict(state_dict['optimizer_d'])



        #self.criterion = torch.nn.MSELoss()



        # calculate total number of parameters in the network
        num_params = sum(p.data.numel() for p in self.autoencoder.parameters()
                         if p.data.data_ptr() != self.autoencoder.embedding.weight.data.data_ptr())
        print ("{} autoencoder parameters".format(num_params))
        num_params = sum(p.data.numel() for p in self.generator.parameters()
                         if p.data.data_ptr() != self.generator.embedding.weight.data.data_ptr())
        print ("{} generator parameters".format(num_params))
        num_params = sum(p.data.numel() for p in self.discriminator.parameters())
        print ("{} discriminator parameters".format(num_params))


    def cuda(self):
        self.autoencoder.cuda()
        self.generator.cuda()
        self.discriminator.cuda()

    def train(self):
        self.autoencoder.train()
        self.generator.train()
        self.discriminator.train()

        losses_g = []
        losses_d = []
        for i,(pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader['train']):
            # Prepare input & target
            x1 = Variable(pairs_padded[0]).cuda()
            x2 = Variable(pairs_padded[1]).cuda()
            y = Variable(pairs_padded[2]).cuda()
            x1_mask = Variable(pairs_mask[0]).cuda()
            x2_mask = Variable(pairs_mask[1]).cuda()
            y_mask = Variable(pairs_mask[2]).cuda()
            y_length = Variable(pairs_len[2]).cuda()
            '''
            Train Discriminator
            '''
            # Forward propagation
            #Real_rnn_outputs = self.autoencoder.forwardToHidden(y_in=y, y_in_mask=y_mask, y_out=y)

            Real_rnn_outputs,x1_hiddens, x2_hiddens = self.generator.forwardToHidden(x1=x1, x2=x2,
                                                                                     x1_mask=x1_mask, x2_mask=x2_mask,
                                                                                     y=y,teacher_forcing=True)
            Fake_rnn_outputs,x1_hiddens, x2_hiddens = self.generator.forwardToHidden(x1=x1, x2=x2,
                                                                                     x1_mask=x1_mask, x2_mask=x2_mask,
                                                                                     y=y,teacher_forcing=False)
            Real_D_outputs=self.discriminator.forward(encoder1_hiddens=x1_hiddens, encoder2_hiddens=x2_hiddens,
                                                      x1_mask = x1_mask, x2_mask=x2_mask,y=Real_rnn_outputs)
            Fake_D_outputs=self.discriminator.forward(encoder1_hiddens=x1_hiddens, encoder2_hiddens=x2_hiddens,
                                                      x1_mask = x1_mask, x2_mask=x2_mask,y=Fake_rnn_outputs)

            if self.args.loss_type == 'log_prob':
                Loss_D = -torch.mean(torch.log(Real_D_outputs+1e-8) + torch.log(1 - Fake_D_outputs+1e-8))
            elif self.args.loss_type == 'least_square':
                Loss_D = 0.5 * (torch.mean((Real_D_outputs - 1)**2) + torch.mean(Fake_D_outputs**2))

            Loss_D.backward()
            torch.nn.utils.clip_grad_norm(self.discriminator.parameters(),max_norm=5.0)
            # Update weights
            self.optimizer_d.step()
            self.optimizer_d.zero_grad()
            self.optimizer_g.zero_grad()
            '''
            Train Generator
            '''

            # Forward propagation
            Fake_rnn_outputs,x1_hiddens,x2_hiddens = self.generator.forwardToHidden(x1=x1, x2=x2,
                                                                                    x1_mask=x1_mask, x2_mask=x2_mask,
                                                                                    y=y, teacher_forcing=False)
            Fake_D_outputs = self.discriminator.forward(encoder1_hiddens=x1_hiddens, encoder2_hiddens=x2_hiddens,
                                                        x1_mask = x1_mask, x2_mask=x2_mask,y=Fake_rnn_outputs)
            if self.args.loss_type == 'log_prob':
                Loss_G = -torch.mean(torch.log(Fake_D_outputs+1e-8))
            elif self.args.loss_type == 'least_square':
                Loss_G = 0.5 * torch.mean((Fake_D_outputs -1)**2)
            Loss_G.backward()
            torch.nn.utils.clip_grad_norm(self.generator.parameters(), max_norm=5.0)
            # Update weights
            self.optimizer_g.step()
            self.optimizer_d.zero_grad()
            self.optimizer_g.zero_grad()

            self.updates += 1
            # Add loss
            losses_d.append(Loss_D.cpu().data[0])
            #losses_g.append(Loss_G.cpu().data[0])


        loss_d = sum(losses_d) / len(self.loader['train'])
        loss_g = sum(losses_g) / len(self.loader['train'])

        return loss_d, loss_g

    def evaluate(self,INPUT1_TYPE="normal"):
        self.generator.eval()
        losses = []
        exact_matches = []
        f1_scores = []
        for i, (pairs_padded, pairs_len, pairs_mask) in enumerate(self.loader['test']):
            # Prepare input & target
            x1 = Variable(pairs_padded[0],volatile=True).cuda()
            x2 = Variable(pairs_padded[1],volatile=True).cuda()
            y = Variable(pairs_padded[2],volatile=True).cuda()
            x1_mask = Variable(pairs_mask[0],volatile=True).cuda()
            x2_mask = Variable(pairs_mask[1],volatile=True).cuda()
            y_mask = Variable(pairs_mask[2],volatile=True).cuda()
            y_length = Variable(pairs_len[2],volatile=True).cuda()
            # Forward propagation
            outputs, outputs_indices = self.generator.forward(x1=x1, x2=x2, x1_mask=x1_mask, x2_mask=x2_mask, y=y)
            # outputs: [ batch x seq_len x #vocab_output ]
            # outputs_indices = [ batch x seq_len ]
            # y: [ Batch x seq_len ]
            # Calculate cross-entropy loss
            loss = masked_cross_entropy(logits=outputs,
                                        target=y,
                                        target_mask=y_mask,
                                        target_length=y_length)
            # loss = self.criterion.forward(input=outputs.view(-1,outputs.size(2)),target=y.view(-1))
            losses.append(loss.cpu().data[0])
            # exact_match: True or False (Bool)
            # f1_score: 0 ~ 1 (Float)
            exact_match, f1_score = self.score(outputs_indices,y)
            exact_matches.append(exact_match)
            f1_scores.append(f1_score)
            #output_sentence_tokens = self.indices2text(outputs_indices)
            #target_sentence_tokens = self.indices2text(y)

        loss = sum(losses) / len(self.loader['test'])
        exact_match = sum(exact_matches) / len(self.loader['test'])
        f1_score = sum(f1_scores) / len(self.loader['test'])
        # return perplexity
        return math.exp(loss), exact_match, f1_score

    def indices2text(self,indices):
        #print(indices.size())
        indices = indices.squeeze(0).cpu().data.numpy() # [ seq_len ]
        #indices = indices.squeeze(axis=1)
        sentence = []
        for index in indices:
            #print(index.shape)
            word_text = self.langs[2].itos[index]
            sentence.append(word_text)

        return sentence


    def score(self, output_tokens, answer_tokens):
        output_tokens = output_tokens.squeeze(0).cpu().data.numpy().tolist()
        answer_tokens = answer_tokens.squeeze(0).cpu().data.numpy().tolist()
        exact_match = output_tokens==answer_tokens
        if exact_match:
            return 1.0, 1.0
        common = Counter(output_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0
        precision = 1. * num_same / len(output_tokens)
        recall = 1. * num_same / len(answer_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return exact_match, f1

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimizer_d': self.optimizer_d.state_dict(),
                'optimizer_g': self.optimizer_g.state_dict(),
                'updates': self.updates
            },
            'config': self.args,
            'epoch': epoch
        }
        try:
            pathlib.Path('checkpoint/best_model').mkdir(parents=True, exist_ok=True)
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')


