import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
#from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip
import os
from Transparency.model.transformerUtils import get_curr_time, delete_weights, _conicity, d, eval_acc, get_conicity_mask
from Transparency.model.transformerDataHandling import vectorize_data, datasets
from Transparency.model.modules.transformer import CTransformer

def go(arg):
    """
    Creates and trains a basic transformer for the sentiment classification task.
    """


    #torch.set_deterministic(True)
    torch.manual_seed(arg.seed)

    # Used for converting between nats and bits
    #LOG2E = math.log2(math.e)
    NUM_CLS = 2
    SAVE_PATH = "model_acc{}.pt"


    _start_time = get_curr_time()
    #tbw = SummaryWriter(log_dir=arg.tb_dir + "_{}".format(_start_time)) # Tensorboard logging

    # Create the dataset
    dataset_args = {}
    dataset_args['padding_length'] = arg.max_length
    dataset_args['padding_token']  = 0
    dataset_args['batch_size'] = arg.batch_size

    vec = vectorize_data(os.path.join("preprocess", arg.dataset_name.upper(), "{}_dataset.csv".format(arg.dataset_name)), min_df=1) 
    dataset = datasets[arg.dataset_name](vec, dataset_args)

    if arg.max_length < 0:
        mx = max([input.size(1) for inpu,_,_ in dataset.train_data])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=vec.vocab_size,
     num_classes=NUM_CLS, max_pool=arg.max_pool, delete_prop=arg.delete_prop)

    if torch.cuda.is_available():
        print("Cuda baby")
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    best_acc = 0
    for e in range(arg.num_epochs):
        mean_conicity_values = []
        mean_nll_loss = []
        print(f'\n epoch {e}')
        model.train(True)

        for i, batch in enumerate(tqdm.tqdm(dataset.train_data, position=0, leave=True)):

            opt.zero_grad()
            X, y, X_unpadded_len = batch

            input = [X, X_unpadded_len]
            label = y
            #print("LABEL shape", label.shape)
            #print(label)

            out = model(input)

            # CONICITY CONSTRAINT ON THE VALUES OF THE ATTENTION HEADS
            heads_batch_conicity =  torch.stack([_conicity(model.tblocks[-1].attention.values[:,:,i,:],\
                                                                  get_conicity_mask(model.tblocks[-1].attention.values[:,:,i,:], input[1]) , \
                                                                  input[1]) for i in range(model.tblocks[-1].heads) ]) # [#heads, batch_size]
            mean_batch_conicity  = torch.mean(heads_batch_conicity,(-1,0)) # [#scalar]. first takes the mean of a each head's batch, then of heads

            #print("Sample conicity val", batch_conicity)
            #print("Sample conicity val per head", mean_batch_conicity)
            mean_conicity_values.append(mean_batch_conicity)

            nll = F.nll_loss(out, label)

            #tbw.add_scalar("nll_loss", nll, (e+1)*(i+1))
            #tbw.add_scalar("mean_batch_conicity", mean_batch_conicity, (e+1)*(i+1))

            # DIVERSITY COMPOMNENT
            if arg.diversity_transformer == True:
              loss = nll + arg.diversity_weight*mean_batch_conicity
            else: 
              loss = nll

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            sch.step()

            seen += input[0].size(0)
            #tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
        
        with torch.no_grad():
            
            acc = eval_acc(model, dataset.dev_data)
            if acc > best_acc:
              best_acc = acc
              print("new best valiadtion achieved: {}. Saving model ...".format(best_acc))
              torch.save({
              'epoch': e,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': opt.state_dict(),
              'loss': loss,
              }, SAVE_PATH.format(round(acc,3)))

            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            mean_epoch_conicity = torch.mean(torch.stack(mean_conicity_values))
            print("Mean concity value {}".format(mean_epoch_conicity))
            # Upon epoch completion, write to tensoraboard
            #tbw.add_scalar('mean_epoch_conicity', mean_epoch_conicity, e)
            #tbw.add_scalar('classification/test-loss', float(loss.item()), e)
