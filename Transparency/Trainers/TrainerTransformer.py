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
from Transparency.model.transformerDataHandling import vectorize_data, datasets, dataset_config
from Transparency.model.modules.transformer import CTransformer
import pickle
import time

def go(config):
    """
    Creates and trains a basic transformer for the sentiment classification task.
    """
    
    start_time = time.time()

    arg = dataset_config[config.dataset_name]["arg"]
    dataset_config[config.dataset_name]["arg"].diversity_transformer = True if config.model_type=="diversity_transformer" else False
    print("Diversity",  dataset_config[config.dataset_name]["arg"].diversity_transformer ) 

    logging_path = os.path.join("experiments", arg.dataset_name, config.model_type, f'experiment_{get_curr_time()}.txt')

    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    log_file = open(os.path.join(logging_path, "log.txt"), "w+" )
    log_file.write(f"Options: {arg}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log_file.write(f'Using device {device}')

    #torch.set_deterministic(True)
    torch.manual_seed(arg.seed)

    # Used for converting between nats and bits
    NUM_CLS = 2
    SAVE_PATH = os.path.join(logging_path, "model.pt")

    # Create the dataset
    dataset_args = {}
    dataset_args['padding_length'] = arg.max_length
    dataset_args['padding_token']  = 0
    dataset_args['batch_size'] = arg.batch_size
    
    #vec = vectorize_data(os.path.join("preprocess", arg.dataset_name.upper(), "{}_dataset.csv".format(arg.dataset_name)), min_df=1) 
    print(os.path.join("preprocess", arg.dataset_name.upper(), "vec_{}.p".format(arg.dataset_name)) )
    

    #vec = pickle.load(open(os.path.join("preprocess", arg.dataset_name.upper(), "vec_{}.p".format(arg.dataset_name)), "rb") )
    vec = pickle.load(open("preprocess/20News/vec_20news_sports.p", "rb") )
    dataset = datasets[arg.dataset_name](vec, dataset_args)

    if arg.max_length < 0:
        mx = max([input.size(1) for inpu,_,_ in dataset.train_data])
        mx = mx + 0.25*mx # add 25% slack
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=vec.vocab_size,
     num_classes=NUM_CLS)

    model.token_embedding.weight.data.copy_(torch.from_numpy(vec.embeddings))
    model.token_embedding.weight.requires_grad = False

    if torch.cuda.is_available():
        print("[INFO]: GPU training enabled")
        model.to(device)

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters(), weight_decay=0.00001)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    best_acc = 0
    for e in range(arg.num_epochs):
        mean_conicity_values = []
        print(f'\n epoch {e+1}')
        log_file.write(f'\n epoch {e+1}')
        model.train(True)

        for i, batch in enumerate(tqdm.tqdm(dataset.train_data, position=0, leave=True)):

            opt.zero_grad()
            X, y, X_unpadded_len = batch

            input = [X.to(device), X_unpadded_len.to(device)]
            label = y.to(device)
            out = model(input)

            # CONICITY CONSTRAINT ON THE VALUES OF THE ATTENTION HEADS
            heads_batch_conicity =  torch.stack([_conicity(model.tblocks[-1].attention.values[:,:,i,:], 
                                                                  get_conicity_mask(model.tblocks[-1].attention.values[:,:,i,:], input[1]) , \
                                                                  input[1]) for i in range(model.tblocks[-1].heads) ]) # [#heads, batch_size], only looks at the last trans block
            mean_batch_conicity  = torch.mean(heads_batch_conicity,(-1,0)) # [#scalar]. first takes the mean of a each head's batch, then of heads
            mean_conicity_values.append(mean_batch_conicity)

            nll = F.nll_loss(out, label)

            #tbw.add_scalar("nll_loss", nll, (e+1)*(i+1))
            #tbw.add_scalar("mean_batch_conicity", mean_batch_conicity, (e+1)*(i+1))

            # DIVERSITY COMPOMNENT
            if arg.diversity_transformer == "diversity_transformer":
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
            #print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(opt.param_groups[0]["lr"],3))

            seen += input[0].size(0)
            #tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
        
        with torch.no_grad():
            
            acc = eval_acc(model, dataset.dev_data)
            log_file.write(f'-- {"validation"} accuracy {acc:.3}')
            print(f'-- {"validation"} accuracy {acc:.3}')
            mean_epoch_conicity = torch.mean(torch.stack(mean_conicity_values))
            print("Mean concity value {}".format(mean_epoch_conicity))
            log_file.write(f'-- mean conicity in epoch {mean_epoch_conicity:.4}')
            log_file.write(f'-- loss  {float(loss.item()):.3}')
            if acc > best_acc:
                best_acc = acc
                print("new best valiadtion accuracy achieved:. Saving model ...")
                torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
                }, SAVE_PATH.format(round(acc,3)))
                log_file.write("Saving model with best validation accuracy so far...")
            # Upon epoch completion, write to tensoraboard
            #tbw.add_scalar('mean_epoch_conicity', mean_epoch_conicity, e)
            #tbw.add_scalar('classification/test-loss', float(loss.item()), e)
    
    checkpoint = torch.load(SAVE_PATH)
    best_model = CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=vec.vocab_size,
     num_classes=NUM_CLS, max_pool=arg.max_pool)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    test_acc = eval_acc(best_model, dataset.test_data)
    print(f"Training concluded. Best model with validation accuracy {best_acc:.3} achieved test accuracy: {test_acc}")
    total_time = time.time() - start_time
    print(f"Execution took {total_time:.3} seconds")
    log_file.write(f"\n total time {total_time:.3} [seconds]")