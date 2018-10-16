import torch.nn as nn
from order2taskplan import layers


class selfAttnDiscriminator(nn.Module):
    def __init__(self, attn_input_size, attn_hidden_size):
        super(selfAttnDiscriminator,self).__init__()

        '''
        Self attention layers for dimension reduction
        '''
        self.selfAttention = layers.selfAttention(input_size=attn_input_size,
                                                  hidden_size=attn_hidden_size,
                                                  output_size=1)
        '''
        Discriminator 
        '''
        self.linear = nn.Linear(in_features=attn_input_size,out_features=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, seq, seq_mask=None):
        """Inputs:
        seq = sequence of vectors  [batch * len * hidden_size]
        seq_mask = mask of seq        [batch * len * hidden_size]
        """
        # dimension reduction (sequence of vectors -> single vector)
        scores = self.selfAttention.forward(x=seq,x_mask=seq_mask) # scores = [batch * 1 * len]
        vec = scores.bmm(seq) # vec = [batch * 1 * hidden_size]
        # Discriminator (output size = 1)
        out = self.sigmoid(self.linear.forward(vec.squeeze(1))) # out = [batch * 1]

        return out
