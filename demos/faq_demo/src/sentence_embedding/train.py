import torch
from torch import nn
from torch.autograd import Variable

from src.sentence_embedding.data_loader import DataLoader
from src.sentence_embedding.model import UniSkip
from src.sentence_embedding.config import *
from datetime import datetime, timedelta

d = DataLoader(text_file = './data/faq.txt')

mod = UniSkip()
if USE_CUDA:
    mod.cuda(CUDA_DEVICE)

lr = 3e-4
optimizer = torch.optim.Adam(params=mod.parameters(), lr=lr)

loss_trail = []
last_best_loss = None
current_time = datetime.utcnow()

def debug(i, loss, prev, ne, prev_pred, next_pred):
    global loss_trail
    global last_best_loss
    global current_time

    this_loss = loss.data.item()
    loss_trail.append(this_loss)
    loss_trail = loss_trail[-20: ]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time-current_time)
    current_time = new_current_time
    print("Iteration {}: time = {} last best loss = {}, this_loss = {}".format(
        i, time_elapsed, last_best_loss, this_loss
    ))
    print("prev = {}\nnext = {}\npred_prev={}\npred_next = {} ".format(
        d.convert_indices_to_sentences(prev),
        d.convert_indices_to_sentences(nex),
        d.convert_indices_to_sentences(prev_pred),
        d.convert_indices_to_sentences(next_pred),
    ))

    try:
        trail_loss = sum(loss_trail) / len(loss_trail)
        if last_best_loss is None or last_best_loss > trail_loss:
            print("Loss improved from {} to {}".format(last_best_loss, trail_loss))

            save_loc = "./saved_models/skip-best".format(lr, VOCAB_SIZE)
            print("saving model at {}".format(save_loc))
            torch.save(mod.state_dict(), save_loc)

            last_best_loss = trail_loss

    except Exception as e:
        print("Can't save model because {}".format(e))


print("start training..")
# a hundred iterations
for i in range(0, 100):
    sentences, lengths = d.fetch_batch(64)
    # print(sentences, lengths)
    # print(sentences.type(torch.LongTensor))
    sentences = sentences.type(torch.LongTensor)
    loss, prev, nex, prev_pred, next_pred = mod(sentences, lengths)

    if i % 10 == 0:
        debug(i, loss, prev, nex, prev_pred, next_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()