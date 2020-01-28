import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify
from models import Seq2Seq2Decoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify, load_models
import shutil

parser = argparse.ArgumentParser(description='ARAE for Yelp transfer')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='yelp_example',
                    help='output directory name')
parser.add_argument('--load_vocab', type=str, default="",
                    help='path to load vocabulary from')
parser.add_argument('--corpus_name', type=str, required=True)

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=30000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=25,
                    help='maximum sentence length')
parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                    help='not lowercase all text')
parser.set_defaults(lowercase=True)

# Other
parser.add_argument('--epochs', type=int, default=25,
                    help='maximum number of epochs')
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--no-cuda', dest='cuda', action='store_true',
                    help='not using CUDA')
parser.set_defaults(cuda=True)
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
print(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

# make output directory if it doesn't already exist
if not os.path.isdir(args.outf):
    os.makedirs(args.outf)

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

def evaluate_generator(whichdecoder, noise):
    gan_gen.eval()
    autoencoder.eval()

    for d in en_data:    
        indices, _, lengths = d 
        indices = indices.cuda()
        hidden = autoencoder.encode(indices, lengths, noise=noise)
        max_indices = \
            autoencoder.generate(whichdecoder, hidden, maxlen=50, sample=args.sample)

        with open("%s/%s_generated_%s.txt" % (args.outf, whichdecoder, args.epochs), "a") as f:
            max_indices = max_indices.data.cpu().numpy()
            for idx in max_indices:
                # generated sentence
                words = [corpus.dictionary.idx2word[x] for x in idx]
                # truncate sentences to first occurrence of <eos>
                truncated_sent = []
                for w in words:
                    if w != '<eos>':
                        truncated_sent.append(w)
                    else:
                        break
                chars = " ".join(truncated_sent)
                f.write(chars)
                f.write("\n")

# Load data
label_ids = {"pos": 1, "neg": 0}
id2label = {1:"pos", 0:"neg"}
datafiles = [(args.data_path, args.corpus_name, False),
            ]

with open(os.path.join(args.outf,"vocab.json"), "r") as f:
    vocabdict = json.load(f)
vocabdict = {k: int(v) for k, v in vocabdict.items()}
corpus = Corpus(datafiles,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                vocab=vocabdict)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens

eval_batch_size = 100
en_data = batchify(corpus.data[args.corpus_name], eval_batch_size, shuffle=False)
print(len(en_data))
print("Loaded data!")

model_args, idx2word, autoencoder, gan_gen, gan_disc = load_models(args.outf, args.epochs, twodecoders=True)

if args.cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()

one = to_gpu(args.cuda, torch.FloatTensor([1]))
mone = one * -1

evaluate_generator(1, False)
