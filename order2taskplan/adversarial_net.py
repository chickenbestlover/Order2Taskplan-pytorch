import torch.nn as nn
from order2taskplan.seqs2seq import seqs2seq
from order2taskplan.seq2seq import seq2seq
from order2taskplan.layers import AttentionRNNDecoder_single

class adversarial_net(nn.Module):
    def __init__(self, args, langs, max_seqlen, embedding=None, state_dict=None):
        super(adversarial_net, self).__init__()
        self.args = args
        self.langs = langs

        self.generator= seqs2seq(args,
                                 input_embedding=embedding,
                                 output_lang=langs[2],
                                 max_seqlen=max_seqlen)

        self.autoencoder = seq2seq(args,
                                   input_embedding=embedding,
                                   output_lang=langs[2],
                                   max_seqlen=max_seqlen)
        self.discriminator =



    def forward(self, x1, x2, x1_mask, x2_mask, y, INPUT1_TYPE='normal'):
        """Inputs:
        x1 = order word indices             [batch * len_o]
        x1_mask = order padding mask        [batch * len_o]
        x2 = environment word indices       [batch * len_e]
        x2_mask = environment padding mask  [batch * len_e]
        """
        # Embed both order and environment

        x1_emb = self.embedding(x1) # [batch * len_o * embed_dim]
        x2_emb = self.embedding(x2) # [batch * len_e * embed_dim]
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Encode order with RNN
        if INPUT1_TYPE=='normal':
            x1_hiddens = self.input1_rnn_encoder.forward(x1_emb, x1_mask) # [batch * len_o * hidden_size]
        elif INPUT1_TYPE=='hall':
            x1_hiddens = self.hall_net.forward(x2_emb, x2_mask)  # [batch * len_o * hidden_size]
        else:
            x1_hiddens = self.input1_rnn_encoder.forward(x1_emb.fill_(0), x1_mask.fill_(0))
            #print('x1_hiddens:', x1_hiddens.size())
        # Encode environment with RNN
        x2_hiddens = self.input2_rnn_encoder.forward(x2_emb, x2_mask) # [batch * len_e * hidden_size]
        #print('x2_hiddens:',x2_hiddens.size())

        outputs, outputs_indices = self.rnn_decoder.forward(x1_hiddens, x2_hiddens, x1_mask, x2_mask, y)

        return outputs, outputs_indices


    def forward_hall(self,x2,x2_mask):
        x2_emb = self.embedding(x2) # [batch * len_e * embed_dim]
        if self.args.dropout_emb > 0:
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)
        x1_hiddens = self.hall_net.forward(x2_emb, x2_mask)  # [batch * len_o * hidden_size]
        return x1_hiddens

    def forward_enc1(self, x1, x1_mask):
        """Inputs:
        x1 = environment word indices       [batch * len_e]
        x1_mask = environment padding mask  [batch * len_e]
        """
        # Embed both order and environment

        x1_emb = self.embedding(x1)  # [batch * len_e * embed_dim]
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
        # Encode order with RNN
        x1_hiddens = self.input1_rnn_encoder.forward(x1_emb, x1_mask)  # [batch * len_o * hidden_size]

        return x1_hiddens
