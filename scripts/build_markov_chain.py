import sys
from count_and_score_kmers import yield_kmers
from count_and_score_kmers import get_file_list_hash
from count_and_score_kmers import get_char2idx
import numpy as np
import os
import pickle

def load_alphabet(f):
    alphabet = []
    with open(f) as fin:
        for line in fin:
            line = line.strip()
            alphabet.append(line)
    return alphabet

def next_token(kmer, sorted_alphabet):
    token = None
    for ii in range(len(kmer)):
        possible_token = ''.join(kmer[:ii+1])
        for c in alphabet:
            if possible_token == c:
                token = possible_token
                continue
    assert token is not None
    
    # fast forward for the length of the token so we don't count the 
    # same letters twice
    skip = len(token) - 1
    return token, skip

def build_markov_chain(kmers_stream, alphabet, order=1):

    # need a mapping from token to index in the transition matrix
    char2idx = get_char2idx(alphabet)

    # the sorted function is guaranteed to be stable, so we can sort 
    # alphabetically then by length to get a list sorted by length 
    # and also alphabetically within each length
    sorted_alphabet = sorted(alphabet)
    sorted_alphabet = sorted(sorted_alphabet, 
            key=lambda x: len(x), reverse=True)

    # need this to make sure that our kmers are longer than the tokens
    max_token_length = max([len(x) for x in alphabet])

    # store the transtion matrix (or tensor)
    transition_counts = np.zeros([len(alphabet)] * (order + 1), dtype='int64')

    skip = 0
    # hold on to the last n tokens, where n is the order of the model + 1
    last_tokens = []
    for kmer in kmers_stream:
        if skip > 0:
            skip = skip - 1
            continue
        if not max_token_length <= len(kmer):
            raise Exception("The kmers must be longer than the longest token")

        if skip > 0:
            skip = skip - 1
            continue

        curr_token, skip = next_token(kmer, alphabet)
        last_tokens.append(curr_token)
        if len(last_tokens) < order + 1:
            continue
        idxs = tuple([char2idx[x] for x in last_tokens])
        transition_counts[idxs] += 1
        last_tokens.pop(0)

    return transition_counts

def build_markov_chain_corpus(corpus, alphabet, order=1):

    # need a mapping from token to index in the transition matrix
    char2idx = get_char2idx(alphabet)

    # the sorted function is guaranteed to be stable, so we can sort 
    # alphabetically then by length to get a list sorted by length 
    # and also alphabetically within each length
    sorted_alphabet = sorted(alphabet)
    sorted_alphabet = sorted(sorted_alphabet, 
            key=lambda x: len(x), reverse=True)

    # need this to make sure that our kmers are longer than the tokens
    max_token_length = max([len(x) for x in alphabet])

    # store the transtion matrix (or tensor)
    transition_counts = np.zeros([len(alphabet)] * (order + 1), dtype='int64')

    skip = 0
    # hold on to the last n tokens, where n is the order of the model + 1
    last_tokens = []
    for kmer in kmers_stream:
        if skip > 0:
            skip = skip - 1
            continue
        if not max_token_length <= len(kmer):
            raise Exception("The kmers must be longer than the longest token")

        if skip > 0:
            skip = skip - 1
            continue

        curr_token, skip = next_token(kmer, alphabet)
        last_tokens.append(curr_token)
        if len(last_tokens) < order + 1:
            continue
        idxs = tuple([char2idx[x] for x in last_tokens])
        transition_counts[idxs] += 1
        last_tokens.pop(0)

    return transition_counts

def tokenize_corpus(corpus, alphabet):
    for token in alphabet:
        for ii in range(len(corpus)):
            if corpus[ii:ii+len(token)] == token:
                corpus[ii:ii+len(token)] = [token]
    return corpus

def count_matrix_to_prob(n):

    # start with some LaPlace smoothing by adding one to all counts
    n = n + 1

    # need to get the sum of counts across all possible decisions 
    ndims = len(n.shape)
    new_shape = list(n.shape[:-1])
    new_shape.append(1)
    sums = np.sum(n, axis=ndims-1).reshape(new_shape)
    reps = [1] * (ndims - 1)
    reps.append(n.shape[-1])
    tiled = np.tile(sums, reps)

    # divide the counts by the sum to get the probability
    return np.divide(n, tiled)

if __name__ == '__main__':
    alphabet_file = sys.argv[1]
    training_name = sys.argv[2]
    order = int(sys.argv[3])
    files = sys.argv[4:]
    alphabet = load_alphabet(alphabet_file)
    max_token_length = max([len(x) for x in alphabet])
    storage_dir = 'markov_model_counts'
    files_hash = get_file_list_hash(files)
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)
    fname = f"{training_name}__{os.path.basename(alphabet_file)}__" \
            f"{order}__{files_hash}.pkl"
    if not os.path.exists(fname):
        transition_tensor = build_markov_chain(yield_kmers(files, 
            max_token_length, verbose=True), alphabet, order=order)
        pickle.dump((alphabet, transition_tensor), 
                open(os.path.join(storage_dir, fname), 'wb'))
