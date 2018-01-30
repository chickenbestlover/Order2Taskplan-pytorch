import torch
from torch.nn import functional
from torch.autograd import Variable

def masked_cross_entropy(logits, target, target_mask,target_length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        target_mask: A Variable containing a ByteTensor of size (batch, max_len)
            which contains the mask of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    #print('logits_flat_grad', logits_flat.requires_grad)

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat,dim=1)
    #print('log_probs_grad', log_probs_flat.requires_grad)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    #print('target_flat_grad', target.requires_grad)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    #print('losses_flat_grad', losses_flat.requires_grad)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)

    losses.data.masked_fill_(target_mask.data.byte(),0)
    #print('losses_grad', losses.requires_grad)

    loss = losses.sum() / target_length.float().sum()
    #print('loss_grad',loss.requires_grad)
    return loss