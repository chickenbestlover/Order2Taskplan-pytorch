import torch.nn as nn

from order2taskplan import layers


class AttnRNNDiscriminator(nn.Module):
    def __init__(self, args, input_embedding, output_lang, max_seqlen, padding_idx=0):
        super(AttnRNNDiscriminator, self).__init__()
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
        # RNN order encoder
        self.input1_rnn_encoder = layers.StackedBRNN(
            input_size = input_embedding.vectors.size(1),
            hidden_size = args.hidden_size,
            num_layers = args.num_layers,
            dropout_rate = args.dropout_rnn,
            dropout_output = args.dropout_rnn_output,
            concat_layers = False,
            packing = args.packing)

        # RNN environment encoder
        self.input2_rnn_encoder = layers.StackedBRNN(
            input_size=input_embedding.vectors.size(1),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            packing=args.packing)

        # RNN order encoder by halluciation
        decoder_input_size =input_embedding.vectors.size(1)+ args.hidden_size*2




        '''
        Attention & Decoding
        '''
        latent_vector_size = args.latent_vector_size if args.latent_vector_size is not None else 0
        decoder_input_size =input_embedding.vectors.size(1)+ args.hidden_size*4+latent_vector_size
        self.rnn_decoder = layers.AttentionRNNDecoder_double(input_size=decoder_input_size,
                                                             hidden_size=args.hidden_size,
                                                             embedding_dim = input_embedding.vectors.size(1),
                                                             num_layers=args.num_layers,
                                                             max_seqlen=max_seqlen,
                                                             lang=output_lang,
                                                             padding_idx=0,
                                                             dropout_rate=0,
                                                             dropout_output=False,
                                                             packing=False,
                                                             teacher_forcing_ratio=args.teacher_forcing_ratio,
                                                             latent_vector_size=latent_vector_size)


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


    def forwardToHidden(self, x1, x2, x1_mask, x2_mask, y, INPUT1_TYPE='normal'):
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

        rnn_outputs = self.rnn_decoder.forward(x1_hiddens, x2_hiddens, x1_mask, x2_mask, y,output_hidden=True)

        return rnn_outputs


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