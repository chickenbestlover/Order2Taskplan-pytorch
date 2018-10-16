# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

import cuda_functional as MF

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False,
                 concat_layers=False, packing=False):
        super(StackedBRNN, self).__init__()
        self.padding = packing
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(nn.LSTM(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))
            # self.rnns.append(MF.SRUCell(input_size, hidden_size,
            #                           dropout=dropout_rate,
            #                           rnn_dropout=dropout_rate,
            #                           use_tanh=1,
            #                           bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        #if self.padding or not self.training:
        #    return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
#            if self.dropout_rate > 0:
#                rnn_input = F.dropout(rnn_input,
#                                      p=self.dropout_rate,
#                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output.contiguous()

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class AttentionRNNDecoder_double(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers, max_seqlen, lang,
                 padding_idx=0, dropout_rate=0, dropout_output=False, packing=False,teacher_forcing_ratio=0.5,latent_vector_size=0):
        super(AttentionRNNDecoder_double, self).__init__()

        self.max_seqlen = max_seqlen
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.lang = lang
        self.latent_vector_size=latent_vector_size
        self.rnn = StackedLSTM(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout_rate=dropout_rate,
                                       dropout_output=dropout_output,
                                       packing=packing)

        self.input1_attn = attn_Bahdanau(max_seqlen=max_seqlen[0],
                                         rnn_hidden_size= hidden_size,
                                         embedding_dim=embedding_dim)
        self.input2_attn = attn_Bahdanau(max_seqlen=max_seqlen[1],
                                         rnn_hidden_size= hidden_size,
                                         embedding_dim=embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=len(lang.stoi),
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.linear = nn.Linear(in_features=hidden_size,out_features=lang.n_words)

    def forward(self,encoder1_hiddens, encoder2_hiddens, x1_mask, x2_mask,
                y=None,output_hidden=False,teacher_forcing=None,recursive_len=0):

        # generate initial word embedding (SOS token)
        batch_size = encoder1_hiddens.size(0)
        x = Variable(torch.zeros(batch_size,1).fill_(self.lang.stoi['SOS']).long()).cuda()
        #hidden = None
        hidden = self.rnn.init_hidden(batch_size=batch_size)
        outputs = [] # output vectors
        outputs_indices = [] # output indices
        rnn_outputs = []
        USE_TEACHER_FORCING = True if teacher_forcing else random.random() < self.teacher_forcing_ratio
        #USE_TEACHER_FORCING = True
        if self.latent_vector_size>0:
            latent_vector = Variable(torch.randn(batch_size,self.latent_vector_size)).cuda()
        for i in range(self.max_seqlen[2]):
            x_emb = self.embedding(x)
            #print('x_emb:',x_emb.size())
            # generate attention-applied hidden states for both encoders
            encoder1_attn_scores = self.input1_attn.forward(input = x_emb.squeeze(1),
                                                          hidden = hidden[0][0],
                                                          mask = x1_mask)
            encoder1_attn_hidden = torch.bmm(encoder1_attn_scores.unsqueeze(1),encoder1_hiddens).squeeze(1)

            encoder2_attn_scores = self.input2_attn.forward(input = x_emb.squeeze(1),
                                                          hidden = hidden[0][0],
                                                          mask = x2_mask)
            encoder2_attn_hidden = torch.bmm(encoder2_attn_scores.unsqueeze(1),encoder2_hiddens).squeeze(1)
            #print('encoder1_attn_hidden:',encoder1_attn_hidden.size())
            #print('encoder2_attn_hidden:', encoder2_attn_hidden.size())
            if self.latent_vector_size>0:
                rnn_input = torch.cat([encoder1_attn_hidden, encoder2_attn_hidden, x_emb.squeeze(1),latent_vector], dim=1)
            else:
                rnn_input = torch.cat([encoder1_attn_hidden,encoder2_attn_hidden,x_emb.squeeze(1)],dim=1)
            #print('rnn_input:',rnn_input.size())
            #print(self.rnn)
            rnn_output, hidden = self.rnn.forward(input=rnn_input,hidden=hidden)
            #print('rnn_output:', rnn_output.size())
            output = self.linear.forward(rnn_output)
            #print('output:', output.size())
            outputs.append(output.unsqueeze(1))
            rnn_outputs.append(rnn_output.unsqueeze(1))
            topValue, topIndex = output.data.topk(k=1, dim=1)
            outputs_indices.append(Variable(topIndex))
            if self.training and USE_TEACHER_FORCING:
                # Target becomes the next input
                # print('target:',y[:,i].unsqueeze(1))
                x = y[:, i].unsqueeze(1) # Next target is next input
            elif i < recursive_len:
                x = y[:, i].unsqueeze(1) # Next target is next input

            else:
                # Network output becomes the next input
                #print('topvalue:',topValue)
                #print('topIndex:',topIndex)
                x = Variable(topIndex) # [ batch x 1 ]

        if output_hidden:
            rnn_outputs = torch.cat(rnn_outputs,dim=1)
            return rnn_outputs

        outputs = torch.cat(outputs, dim=1) # [batch x seq_len x num_classes ]
        outputs_indices = torch.cat(outputs_indices,dim=1) # [ batch x seq_len ]
        #print('outputs:', outputs.size())

        return outputs, outputs_indices

    def forward_for_hall(self,encoder1_hiddens, encoder2_hiddens, x1_mask, x2_mask, y):

        # generate initial word embedding (SOS token)
        batch_size = encoder1_hiddens.size(0)
        x = Variable(torch.zeros(batch_size,1).fill_(self.lang.stoi['SOS']).long()).cuda()
        #hidden = None
        hidden = self.rnn.init_hidden(batch_size=batch_size)
        outputs = [] # output vectors
        outputs_indices = [] # output indices

        USE_TEACHER_FORCING = random.random() < self.teacher_forcing_ratio
        #USE_TEACHER_FORCING = True

        for i in range(self.max_seqlen[2]):
            x_emb = self.embedding(x)
            #print('x_emb:',x_emb.size())
            # generate attention-applied hidden states for both encoders
            encoder1_attn_scores = self.input1_attn.forward(input = x_emb.squeeze(1),
                                                          hidden = hidden[0][0],
                                                          mask = x1_mask)
            encoder1_attn_hidden = torch.bmm(encoder1_attn_scores.unsqueeze(1),encoder1_hiddens).squeeze(1)

            encoder2_attn_scores = self.input2_attn.forward(input = x_emb.squeeze(1),
                                                          hidden = hidden[0][0],
                                                          mask = x2_mask)
            encoder2_attn_hidden = torch.bmm(encoder2_attn_scores.unsqueeze(1),encoder2_hiddens).squeeze(1)
            #print('encoder1_attn_hidden:',encoder1_attn_hidden.size())
            #print('encoder2_attn_hidden:', encoder2_attn_hidden.size())
            rnn_input = torch.cat([encoder1_attn_hidden,encoder2_attn_hidden,x_emb.squeeze(1)],dim=1)
            #print('rnn_input:',rnn_input.size())
            #print(self.rnn)
            rnn_output, hidden = self.rnn.forward(input=rnn_input,hidden=hidden)
            #print('rnn_output:', rnn_output.size())
            output = self.linear.forward(rnn_output)
            #print('output:', output.size())
            outputs.append(output.unsqueeze(1))
            topValue, topIndex = output.data.topk(k=1, dim=1)
            outputs_indices.append(Variable(topIndex))
            if i<y.size(1):
                # Target becomes the next input
                #print('target:',y[:,i].unsqueeze(1))
                x = y[:,i].unsqueeze(1) #Next target is next input
            else:
                # Network output becomes the next input
                #print('topvalue:',topValue)
                #print('topIndex:',topIndex)
                x = Variable(topIndex) # [ batch x 1 ]


        outputs = torch.cat(outputs, dim=1) # [batch x seq_len x num_classes ]
        outputs_indices = torch.cat(outputs_indices,dim=1) # [ batch x seq_len ]
        #print('outputs:', outputs.size())

        return outputs, outputs_indices

class AttentionRNNDecoder_hall(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, max_seqlen,
                 dropout_rate=0, dropout_output=False, packing=False):
        super(AttentionRNNDecoder_hall, self).__init__()

        self.max_seqlen = max_seqlen
        self.hidden_size = hidden_size
        self.rnn = StackedLSTM(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout_rate=dropout_rate,
                                       dropout_output=dropout_output,
                                       packing=packing)
        self.input2_attn = attn_Bahdanau(max_seqlen=max_seqlen[1],
                                         rnn_hidden_size=hidden_size,
                                         embedding_dim=hidden_size*2)
        self.linear = nn.Linear(in_features=hidden_size,out_features=hidden_size*2)

    def forward(self,encoder2_hiddens, x2_mask):

        # generate initial word embedding (SOS token)
        batch_size = encoder2_hiddens.size(0)
        #hidden = None
        hidden = self.rnn.init_hidden(batch_size=batch_size)
        outputs = [] # output vectors
        output = Variable(torch.FloatTensor(batch_size,self.hidden_size*2).fill_(0)).cuda()

        for i in range(self.max_seqlen[0]):
            # generate attention-applied hidden states for both encoders
            encoder2_attn_scores = self.input2_attn.forward(input = output,
                                                          hidden = hidden[0][0],
                                                          mask = x2_mask)
            encoder2_attn_hidden = torch.bmm(encoder2_attn_scores.unsqueeze(1),encoder2_hiddens).squeeze(1)
            #print('encoder1_attn_hidden:',encoder1_attn_hidden.size())
            #print('encoder2_attn_hidden:', encoder2_attn_hidden.size())
            rnn_input = encoder2_attn_hidden
            #print('rnn_input:',rnn_input.size())
            #print(self.rnn)
            rnn_output, hidden = self.rnn.forward(input=rnn_input,hidden=hidden)
            #print('rnn_output:', rnn_output.size())
            output = self.linear.forward(rnn_output)
            #print('output:', output.size())
            outputs.append(output.unsqueeze(1))


        outputs = torch.cat(outputs, dim=1) # [batch x seq_len x num_classes ]
        #print('outputs:', outputs.size())

        return outputs


class AttentionRNNDecoder_single(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers, max_seqlen, lang,
                 padding_idx=0, dropout_rate=0, dropout_output=False, packing=False,teacher_forcing_ratio=0.5):
        super(AttentionRNNDecoder_single, self).__init__()

        self.max_seqlen = max_seqlen
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.lang = lang

        self.rnn = StackedLSTM(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout_rate=dropout_rate,
                                       dropout_output=dropout_output,
                                       packing=packing)

        self.input_attn = attn_Bahdanau(max_seqlen=max_seqlen[2],
                                        rnn_hidden_size=hidden_size,
                                        embedding_dim=embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=len(lang.stoi),
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.linear = nn.Linear(in_features=hidden_size,out_features=lang.n_words)

    def forward(self, encoder_hiddens, y_mask, y= None, output_hidden= False):

        # generate initial word embedding (SOS token)
        batch_size = encoder_hiddens.size(0)
        x = Variable(torch.zeros(batch_size,1).fill_(self.lang.stoi['SOS']).long()).cuda()
        #hidden = None
        hidden = self.rnn.init_hidden(batch_size=batch_size)
        outputs = [] # output vectors
        outputs_indices = [] # output indices
        rnn_outputs = []
        USE_TEACHER_FORCING = random.random() < self.teacher_forcing_ratio
        #USE_TEACHER_FORCING = True

        for i in range(self.max_seqlen[2]):
            x_emb = self.embedding(x)
            #print('x_emb:',x_emb.size())
            # generate attention-applied hidden states for both encoders
            encoder1_attn_scores = self.input_attn.forward(input = x_emb.squeeze(1),
                                                           hidden =hidden[0][-1],
                                                           mask = y_mask)
            encoder1_attn_hidden = torch.bmm(encoder1_attn_scores.unsqueeze(1), encoder_hiddens).squeeze(1)
            #print('encoder1_attn_hidden:',encoder1_attn_hidden.size())
            #print('encoder2_attn_hidden:', encoder2_attn_hidden.size())
            rnn_input = torch.cat([encoder1_attn_hidden,x_emb.squeeze(1)],dim=1)
            #print('rnn_input:',rnn_input.size())
            #print(self.rnn)
            rnn_output, hidden = self.rnn.forward(input=rnn_input,hidden=hidden)
            #print('rnn_output:', rnn_output.size())
            output = self.linear.forward(rnn_output)
            #print('output:', output.size())
            outputs.append(output.unsqueeze(1))
            rnn_outputs.append(rnn_output.unsqueeze(1))
            topValue, topIndex = output.data.topk(k=1, dim=1)
            outputs_indices.append(Variable(topIndex))
            if self.training and USE_TEACHER_FORCING:
                # Target becomes the next input
                #print('target:',y[:,i].unsqueeze(1))
                x = y[:,i].unsqueeze(1) #Next target is next input
            else:
                # Network output becomes the next input
                #print('topvalue:',topValue)
                #print('topIndex:',topIndex)
                x = Variable(topIndex) # [ batch x 1 ]


        outputs = torch.cat(outputs, dim=1) # [batch x seq_len x num_classes ]
        #print('outputs:', outputs.size())
        outputs_indices = torch.cat(outputs_indices,dim=1) # [ batch x seq_len ]

        if output_hidden:
            rnn_outputs = torch.cat(rnn_outputs,dim=1)
            return rnn_outputs

        return outputs, outputs_indices


class AttentionRNNDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers, max_seqlen, lang,
                 padding_idx=0, dropout_rate=0, dropout_output=False, packing=False,latent_vector_size=0):
        super(AttentionRNNDiscriminator, self).__init__()

        self.max_seqlen = max_seqlen
        self.lang = lang
        self.latent_vector_size=latent_vector_size
        self.rnn = StackedLSTM(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout_rate=dropout_rate,
                                       dropout_output=dropout_output,
                                       packing=packing)

        self.input1_attn = attn_Bahdanau(max_seqlen=max_seqlen[0],
                                         rnn_hidden_size= hidden_size,
                                         embedding_dim=hidden_size)
        self.input2_attn = attn_Bahdanau(max_seqlen=max_seqlen[1],
                                         rnn_hidden_size= hidden_size,
                                         embedding_dim=hidden_size)
        self.embedding = nn.Embedding(num_embeddings=len(lang.stoi),
                                      embedding_dim=hidden_size,
                                      padding_idx=padding_idx)
        self.linear = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self,encoder1_hiddens, encoder2_hiddens, x1_mask, x2_mask, y=None):

        # generate initial word embedding (SOS token)
        batch_size = encoder1_hiddens.size(0)
        x = Variable(torch.zeros(batch_size,1).fill_(self.lang.stoi['SOS']).long()).cuda()
        #hidden = None
        hidden = self.rnn.init_hidden(batch_size=batch_size)
        outputs = [] # output vectors
        outputs_indices = [] # output indices
        rnn_outputs = []
        #USE_TEACHER_FORCING = True
        if self.latent_vector_size>0:
            latent_vector = Variable(torch.randn(batch_size,self.latent_vector_size)).cuda()
        for i in range(self.max_seqlen[2]):
            #print('x:',x.size())

            x_emb = self.embedding(x) if i==0 else y[:,i-1].unsqueeze(1)
            #print('x_emb:',x_emb.size())
            # generate attention-applied hidden states for both encoders
            encoder1_attn_scores = self.input1_attn.forward(input = x_emb.squeeze(1),
                                                          hidden = hidden[0][0],
                                                          mask = x1_mask)
            encoder1_attn_hidden = torch.bmm(encoder1_attn_scores.unsqueeze(1),encoder1_hiddens).squeeze(1)

            encoder2_attn_scores = self.input2_attn.forward(input = x_emb.squeeze(1),
                                                          hidden = hidden[0][0],
                                                          mask = x2_mask)
            encoder2_attn_hidden = torch.bmm(encoder2_attn_scores.unsqueeze(1),encoder2_hiddens).squeeze(1)
            #print('encoder1_attn_hidden:',encoder1_attn_hidden.size())
            #print('encoder2_attn_hidden:', encoder2_attn_hidden.size())
            if self.latent_vector_size>0:
                rnn_input = torch.cat([encoder1_attn_hidden, encoder2_attn_hidden, x_emb.squeeze(1),latent_vector], dim=1)
            else:
                rnn_input = torch.cat([encoder1_attn_hidden,encoder2_attn_hidden,x_emb.squeeze(1)],dim=1)
            #print('rnn_input:',rnn_input.size())
            #print(self.rnn)
            rnn_output, hidden = self.rnn.forward(input=rnn_input,hidden=hidden)
            #print('rnn_output:', rnn_output.size())
            output = self.linear.forward(rnn_output)
            #print('output:', output.size())
            outputs.append(output.unsqueeze(1))
            rnn_outputs.append(rnn_output.unsqueeze(1))



        outputs = torch.cat(outputs, dim=1) # [batch x seq_len x 1 ]
        #print('outputs:', outputs.size())

        return outputs



class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, packing=False):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden=None):
        if hidden is None:
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        1,
                                                        input.size(0),
                                                        self.hidden_size).zero_(), requires_grad=True)
            if next(self.parameters()).is_cuda:
                hx = hx.cuda()

            hidden = (hx, hx)

        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

    def init_hidden(self,batch_size):
        hx = Variable(torch.FloatTensor(self.num_layers*1,batch_size,self.hidden_size).zero_(), requires_grad=True)
        if next(self.parameters()).is_cuda:
            hx = hx.cuda()
        hidden = (hx, hx)
        return hidden

class attn_Bahdanau(nn.Module):
    def __init__(self, max_seqlen, rnn_hidden_size=300, embedding_dim=300):
        super(attn_Bahdanau,self).__init__()

        self.input_size = rnn_hidden_size + embedding_dim
        self.output_size = max_seqlen
        self.linear = nn.Linear(self.input_size,self.output_size)

    def forward(self, hidden, mask,input=None):
        """Inputs:
        input = a word vector             [batch * embedding_dim]
        hidden = RNN hidden state         [batch * rnn_hidden_size]
        mask = encoder padding mask               [batch * max_seqlen]
        """
        #print(input.size())
        #print(hidden.size())
        #print(self.linear)
        #print(mask.size())

        if input is not None:
            hidden = torch.cat([input, hidden], dim=1)
        scores = self.linear(hidden)
        scores.data.masked_fill_(mask.byte().data,-float('inf'))
        scores = F.softmax(scores,dim=1)

        return scores


class selfAttention(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(selfAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, x_mask=None):
        """
        x = batch * len * input_size
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1)) # x_flat = (batch x len) * input_size
        hidden = torch.nn.functional.tanh(self.linear1(x_flat)) # hidden = (batch x len) * hidden_size
        scores = self.linear2(hidden).view(x.size(0), x.size(1), self.output_size) # scores = batch * len * output_size

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(2).expand_as(scores)
            scores.data.masked_fill_(x_mask.data, -float('inf'))

        alpha = F.softmax(scores,dim=1)
        return alpha.transpose(1,2) # batch * output_size * len