import torch.nn as nn

from order2taskplan import layers


class seq2seq(nn.Module):
    def __init__(self, args, input_embedding, output_lang, max_seqlen, padding_idx=0):
        super(seq2seq,self).__init__()
        self.args=args

        '''
        Word Embedding
        '''
        self.embedding = nn.Embedding(num_embeddings=input_embedding.vectors.size(0),
                                      embedding_dim=input_embedding.vectors.size(1),
                                      padding_idx=padding_idx)
        self.embedding.weight.data = input_embedding.vectors
        if args.fix_embed:
            for p in self.embedding.parameters():
                p.requires_grad=False

        '''
        User Order & Environment Encoding
        '''
        # RNN taskplan encoder
        self.input1_rnn_encoder = layers.StackedBRNN(
            input_size = input_embedding.vectors.size(1),
            hidden_size = args.hidden_size,
            num_layers = args.num_layers,
            dropout_rate = args.dropout_rnn,
            dropout_output = args.dropout_rnn_output,
            concat_layers = False,
            packing = args.packing)

        '''
        Attention & Decoding
        '''
        decoder_input_size =input_embedding.vectors.size(1)+ args.hidden_size*2
        self.rnn_decoder = layers.AttentionRNNDecoder_single(input_size=decoder_input_size,
                                                             hidden_size=args.hidden_size,
                                                             embedding_dim = input_embedding.vectors.size(1),
                                                             num_layers=args.num_layers,
                                                             max_seqlen=max_seqlen,
                                                             lang=output_lang,
                                                             padding_idx=0,
                                                             dropout_rate=0,
                                                             dropout_output=False,
                                                             packing=False,
                                                             teacher_forcing_ratio=args.teacher_forcing_ratio)



    def forward(self, y_in, y_in_mask, y_out, INPUT1_TYPE='normal'):
        """Inputs:
        y_in = taskplan indices             [batch * len_o]
        y_in_mask = taskplan mask        [batch * len_o]
        """
        # Embed both order and environment

        y_in_emb = self.embedding(y_in) # [batch * len_o * embed_dim]
        if self.args.dropout_emb > 0:
            y_in_emb = nn.functional.dropout(y_in_emb, p=self.args.dropout_emb,
                                               training=self.training)


        y_in_hiddens = self.input1_rnn_encoder.forward(y_in_emb, y_in_mask) # [batch * len_o * hidden_size]
        #print('y_in_hiddens:', y_in_hiddens.size())


        outputs, outputs_indices = self.rnn_decoder.forward(y_in_hiddens, y_in_mask, y_out)

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

    def forwardToHidden(self, y_in, y_in_mask, y_out):
        """Inputs:
        y_in = taskplan indices             [batch * len_o]
        y_in_mask = taskplan mask        [batch * len_o]
        """
        # Embed both order and environment

        y_in_emb = self.embedding(y_in) # [batch * len_o * embed_dim]
        if self.args.dropout_emb > 0:
            y_in_emb = nn.functional.dropout(y_in_emb, p=self.args.dropout_emb,
                                               training=self.training)


        y_in_hiddens = self.input1_rnn_encoder.forward(y_in_emb, y_in_mask) # [batch * len_o * hidden_size]
        #print('y_in_hiddens:', y_in_hiddens.size())


        rnn_outputs = self.rnn_decoder.forward(y_in_hiddens,y_in_mask,y_out,output_hidden=True)

        return rnn_outputs



class halluciation_net(nn.Module):
    def __init__(self, args, input_embedding, max_seqlen, padding_idx=0):
        super(halluciation_net,self).__init__()
        self.args=args

        '''
        Word Embedding
        '''
        self.embedding = nn.Embedding(num_embeddings=input_embedding.vectors.size(0),
                                      embedding_dim=input_embedding.vectors.size(1),
                                      padding_idx=padding_idx)
        self.embedding.weight.data = input_embedding.vectors
        if args.fix_embed:
            for p in self.embedding.parameters():
                p.requires_grad=False

        '''
        Environment Encoding
        '''
        # RNN environment encoder
        self.input2_rnn_encoder = layers.StackedBRNN(
            input_size=input_embedding.vectors.size(1),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            packing=args.packing)


        '''
        Attention & Decoding
        '''
        decoder_input_size = args.hidden_size*2
        self.rnn_decoder = layers.AttentionRNNDecoder_hall(input_size=decoder_input_size,
                                                           hidden_size=args.hidden_size,
                                                           num_layers=args.num_layers,
                                                           max_seqlen=max_seqlen,
                                                           dropout_rate=0,
                                                           dropout_output=False,
                                                           packing=False)


    def forward(self, x2_emb, x2_mask):
        """Inputs:
        x2_emb = environment word vectors       [batch * len_e * embedding_dim]
        x2_mask = environment padding mask  [batch * len_e]
        """
        # Encode environment with RNN
        x2_hiddens = self.input2_rnn_encoder.forward(x2_emb, x2_mask) # [batch * len_e * hidden_size]
        #print('x2_hiddens:',x2_hiddens.size())

        outputs = self.rnn_decoder.forward(x2_hiddens, x2_mask)

        return outputs
