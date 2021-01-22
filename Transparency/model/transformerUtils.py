import torch
import os
from datetime import datetime

####### IMPORTANCE RANKING CALCULATIONS
def delete_weights(weights, lengths, delete_prop):
  """
  Deletes the delete_prop of values with highest weights. Takes into account the unpadded sequence lengths.
  """

  assert weights.size(0) == lengths.size(0), "The dimension of the lengths does not match the b_size*num_heads!"

  indices = torch.argsort(torch.argsort(weights, dim = 2, descending = True)) # get the indicies of weights from smallest to largest
  #indices += 1
  for i in range(weights.size(0)):
    mask = ~indices[i,:,:].ge(delete_prop*(lengths[i]))
    #print(torch.sum(mask))
    weights[i,:,:][mask] = float("-inf") # set the appropriate proportion of attention weights to 0
  
def get_conicity_mask(var1, lengths):
    b, t, h = var1.size()
    mask = torch.ones((b,t))
    for batch_dim in range(b):
        #print(batch_dim)
        #print(int(lengths[batch_dim]))
        mask[batch_dim, lengths[batch_dim]-1:] = 0
    return mask


##### CONICITY CALCULATION HELPERS
def _conicity(hidden, masks, lengths):
  """
  Calculates the concity of a set with shape [batch_size, seq_length, emb_dim]. 

  Parameters:
    ----------------------------

  hidden: torch.Tensor
    Assuming shape [batch_size, seq_length, emb_dim] 
  masks: torch.Tensor
    mask [batch_size, seq_length] with 0's indiciating that the value has been padded
  lengths: torch.Tensor
    Shape [batch_size]. Contains unpadded sequence lenghts. Used to normalize conicity.

  Output:

  conicity: torch.Tensor
    Shape [batch_size]

  """
  hidden_states = hidden#.to(device)    # [batch size, seq_length, hiddem_dim]
  b,l,h = hidden_states.size()
  masks = masks.float()#.to(device) #[batch_size, hidden dim]
  lengths = (lengths.float() - 2) ## (B)

  hidden_states = hidden_states* (masks.unsqueeze(2))
  mean_state = hidden_states.sum(1) / lengths.unsqueeze(1)
  mean_state = mean_state.unsqueeze(1) #.repeat(1,l,1) #(B,L,H)
  cosine_sim = torch.abs(torch.nn.functional.cosine_similarity(hidden_states, mean_state, dim=2, eps=1e-6))  #(B,L)
  cosine_sim = cosine_sim*(masks)

  conicity = cosine_sim.sum(1) / lengths  # (B)
  return conicity

#### TRAINING LOOP HELPERS

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def eval_acc(model, iterator, mx = 512):
  """ Calculates the accuracy of the model on an iterator"""
  model.eval()
  tot, cor= 0.0, 0.0

  for batch in iterator:

    input = [X, X_unpadded_len]
    label = y

    if input[0].size(1) > mx:
        input[0] = input[0][:, :mx]
    out = model(input).argmax(dim=1)
    #print("Out",out)
    #print("Label", label)
    tot += float(input[0].size(0))
    cor += float((label == out).sum().item())
  acc = cor / tot
  return acc