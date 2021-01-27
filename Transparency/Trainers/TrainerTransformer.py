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
from Transparency.model.transformerDataHandling import vectorize_data, dataset_config, datasets

from Transparency.Trainers.DatasetBC import datasets
from Transparency.model.modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths

from Transparency.model.modules.transformer import CTransformer
import pickle
import time
from sklearn.utils import shuffle

def go(config):
    """
    Creates and trains a basic transformer for the sentiment classification task.
    """
    
    start_time = time.time()

    arg = dataset_config[config.dataset_name]["arg"]
    dataset_config[config.dataset_name]["arg"].diversity_transformer = True if config.model_type=="diversity_transformer" else False
    print("Options:", arg)

    logging_path = os.path.join("experiments", arg.dataset_name, config.model_type, f'experiment_{get_curr_time()}.txt')

    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    log_file = open(os.path.join(logging_path, "log.txt"), "w+" )
    log_file.write(f"Options: {arg}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    use_tqdm = True if torch.cuda.is_available() else False
    log_file.write(f'Using device {device}')

    #torch.set_deterministic(True)
    #torch.manual_seed(arg.seed)

    # Used for converting between nats and bits
    NUM_CLS = 2
    SAVE_PATH = os.path.join(logging_path, "model.pt")
    dataset = datasets[arg.dataset_name]()

    if dataset.max_length is None:
        mx = max([len(input) for input in dataset.train_data.X])
        mx = int(mx + 0.25*mx) # add 25% slack
        print(f'- maximum sequence length: {mx}')
    else:
        mx = dataset.max_length + 2 # +2 for the special tokens

    # create the model
    model = CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=dataset.vec.vocab_size,
     num_classes=NUM_CLS)

    if dataset_config[config.dataset_name]["use_emb"]:
        print("Using pretrained embeddings ... ")
        model.token_embedding.weight.data.copy_(torch.from_numpy(dataset.vec.embeddings))
        model.token_embedding.weight.requires_grad = False

    if torch.cuda.is_available():
        print("[INFO]: GPU training enabled")
        model.to(device)

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters(), weight_decay=0.00001)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    best_acc = 0
    for e in range(arg.num_epochs):
        mean_conicity_values = []
        print(f'\n epoch {e+1}')
        log_file.write(f'\n epoch {e+1}')
        model.train(True)

        data_in = dataset.train_data.X
        target_in = dataset.train_data.y
    
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]
        N = len(data)
        bsize = arg.batch_size
        batches = list(range(0, N, bsize))
        batches = shuffle(batches)
 
        for i,n in enumerate(tqdm.tqdm(batches, position = 0, leave = True, disable = use_tqdm)):
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            X = batch_data.seq
            X_unpadded_len =  batch_data.lengths
            batch_target = target[n:n+bsize]
            y = torch.FloatTensor(batch_target).to(torch.int64)

            opt.zero_grad()
            input = [X.to(device), X_unpadded_len.to(device)]
            label = y.to(device)
            out = model(input)

            # CONICITY CONSTRAINT ON THE VALUES OF THE ATTENTION HEADS
            #heads_batch_conicity =  torch.stack([_conicity(model.tblocks[-1].attention.values[:,:,i,:], 
            #                                                      get_conicity_mask(model.tblocks[-1].attention.values[:,:,i,:], input[1]) , \
            #                                                      input[1]) for i in range(model.tblocks[-1].heads) ]) # [#heads, batch_size], only looks at the last trans block
            heads_batch_conicity =  torch.stack([_conicity(model.tblocks[-1].attention.values[:,:,i,:], 
                                                                  batch_data.masks , \
                                                                  input[1]) for i in range(model.tblocks[-1].heads) ]) 
            mean_batch_conicity  = torch.mean(heads_batch_conicity,(-1,0)) # [#scalar]. first takes the mean of a each head's batch, then of heads
            mean_conicity_values.append(mean_batch_conicity)

            nll = F.nll_loss(out, label)


            # DIVERSITY COMPOMNENT
            if arg.diversity_transformer:
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

            #tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
        with torch.no_grad():
            
            data = dataset.dev_data.X
            target = dataset.dev_data.y
            N = len(data)
            batches = list(range(0, N, bsize))
            tot, cor= 0.0, 0.0
            for i,n in enumerate(tqdm.tqdm(batches, position = 0, leave = True, disable = use_tqdm)):
                batch_doc = data[n:n+bsize]
                batch_data = BatchHolder(batch_doc)
                X = batch_data.seq
                X_unpadded_len =  batch_data.lengths
                batch_target = target[n:n+bsize]
                y = torch.FloatTensor(batch_target).to(torch.int64)

                input = [X.to(device), X_unpadded_len.to(device)]
                label = y.to(device)
                out = model(input).argmax(dim=1)
                tot += float(input[0].size(0))
                cor += float((label == out).sum().item())
                #print(tot)
            acc = float(cor)/float(tot)

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
    best_model = CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=dataset.vec.vocab_size,
     num_classes=NUM_CLS)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    #test_acc = eval_acc(best_model, dataset.test_data)
    data = dataset.test_data.X
    target = dataset.test_data.y

    #sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
    #data = [data_in[i] for i in sorting_idx]
    #target = [target_in[i] for i in sorting_idx]
    N = len(data)
    batches = list(range(0, N, bsize))
    tot, cor= 0.0, 0.0
    mean_conicity_values = []
    for i,n in enumerate(tqdm.tqdm(batches, position = 0, leave = True, disable = use_tqdm)):
        batch_doc = data[n:n+bsize]
        batch_data = BatchHolder(batch_doc)
        X = batch_data.seq
        X_unpadded_len =  batch_data.lengths
        batch_target = target[n:n+bsize]
        y = torch.FloatTensor(batch_target).to(torch.int64)

        input = [X.to(device), X_unpadded_len.to(device)]
        label = y.to(device)
        out = model(input).argmax(dim=1)

        heads_batch_conicity =  torch.stack([_conicity(model.tblocks[-1].attention.values[:,:,i,:], 
                                                                  batch_data.masks , \
                                                                  input[1]) for i in range(model.tblocks[-1].heads) ]) 
        mean_batch_conicity  = torch.mean(heads_batch_conicity,(-1,0)) # [#scalar]. first takes the mean of a each head's batch, then of heads
        mean_conicity_values.append(mean_batch_conicity)

        tot += float(input[0].size(0))
        cor += float((label == out).sum().item())

    mean_epoch_conicity = torch.mean(torch.stack(mean_conicity_values))
    test_acc = float(cor)/float(tot)
    print(f"Training concluded. Best model with validation accuracy {best_acc:.3} achieved test accuracy: {test_acc}")
    total_time = time.time() - start_time
    print(f"Execution took {total_time:.3} seconds")
    log_file.write(f"\n total time {total_time:.3} [seconds]")

    return best_acc, test_acc, mean_epoch_conicity