# A quick intro to implement the Distributed Data Parallel (DDP) training in Pytorch.
# To simply this example, we directly load the ResNet50 using ```torch.hub.load()```,
# and train it from the scratch using the CIFAR10 dataset.

# Run this python script in terminal like "python3 DDP_training.py -n 1 -g 8 -nr 0"

import os
from datetime import datetime
import argparse
import torch
import torchtext
import torch.nn as nn
import functools
import datasets
import pickle as pkl

TIMES = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=256, type=int,
                        help='batch size')
    parser.add_argument('-d', '--directory', default=16, type=int,
                        help='parent directory of pickle dump')
    parser.add_argument('-f', '--frac', default=.35, type=float,
                        help='per process memory fraction')
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.frac)
    train(args, gpu = args.nr)


def train(args, gpu):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # Data loading code
    trainset = datasets.load_dataset('imdb', split='train')

    ###tokenizer things as outlined https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2_lstm.ipynb
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    max_length = 256

    trainset = trainset.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

    min_freq = 5
    special_tokens = ['<unk>', '<pad>']
    vocab = torchtext.vocab.build_vocab_from_iterator(trainset['tokens'],
                                                    min_freq=min_freq,
                                                    specials=special_tokens)
    unk_index = vocab['<unk>']
    pad_index = vocab['<pad>']
    vocab.set_default_index(unk_index)

    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 300
    output_dim = len(trainset.unique('label'))
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.5

    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 
                pad_index)
    model.apply(initialize_weights)

    trainset = trainset.map(numericalize_data, fn_kwargs={'vocab': vocab})
    trainset = trainset.with_format(type='torch', columns=['ids', 'label', 'length'])
    vectors = torchtext.vocab.FastText()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    batch_size = args.batch

    collate2 = functools.partial(collate, pad_index=pad_index)

    model.cuda(gpu)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              collate_fn = collate2)

    total_step = min(len(trainloader), 240)
    train_start = datetime.now()
    trainload_time = datetime.now()
    model.train()
    for epoch in range(args.epochs):
        start = datetime.now()
        for i, batch in enumerate(trainloader):
            if i > total_step: 
                break
            ids = batch['ids'].cuda(non_blocking=True)
            length = batch['length']
            label = batch['label'].cuda(non_blocking=True)

            # Forward pass
            outputs = model(ids, length)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
                TIMES.append((datetime.now() - start).microseconds)
                print("Training complete in: " + str(TIMES[-1]))
                start = datetime.now()

    print("Training done, total epoch {}, total time {}".format(args.epochs, datetime.now()-train_start))
    print('===========================')
    print(sum(TIMES) / len(TIMES))
    print('===========================')

def tokenize_data(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    length = len(tokens)
    return {'tokens': tokens, 'length': length}

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                    dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length.cpu(), batch_first=True, 
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction



#other functions

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'length': batch_length,
             'label': batch_label}
    return batch

if __name__ == '__main__':
    main()