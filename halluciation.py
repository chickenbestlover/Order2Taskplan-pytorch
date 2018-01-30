import torch
import torch.nn as nn
import layers
class halluciationEncoder(nn.Module):
    def __init__(self, args, input_embedding, padding_idx=0):
        super(halluciationEncoder,self).__init__()
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
        User Order & Encoding By Halluciation
        '''
        # RNN order encoder
        self.hall_rnn_encoder = layers.StackedBRNN(
            input_size = input_embedding.vectors.size(1),
            hidden_size = args.hidden_size,
            num_layers = args.num_layers,
            dropout_rate = args.dropout_rnn,
            dropout_output = args.dropout_rnn_output,
            concat_layers = False,
            packing = args.packing)




    def forward(self, x2, x2_mask):
        """Inputs:
        x2 = environment word indices       [batch * len_e]
        x2_mask = environment padding mask  [batch * len_e]
        """
        # Embed both order and environment

        x2_emb = self.embedding(x2) # [batch * len_e * embed_dim]
        if self.args.dropout_emb > 0:
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)


        # Encode environment with RNN
        x1_hiddens = self.hall_rnn_encoder.forward(x2_emb, x2_mask) # [batch * len_e * hidden_size]
        #print('x2_hiddens:',x2_hiddens.size())

        return x1_hiddens