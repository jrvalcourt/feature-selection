import pickle
import sys
from build_markov_chain import count_matrix_to_prob
from build_markov_chain import load_alphabet
from random import random
import numpy as np

def draw_token(prob, alphabet):
    if not np.array(prob).shape == np.array(alphabet).shape:
        raise Exception(f"prob ({np.array(prob).shape}) and alphabet " +
                        f"must be same shape ({np.array(alphabet).shape})")
    r = random()
    cumprob = 0
    for ii in range(len(prob)):
        cumprob += prob[ii]
        if r < cumprob:
            return alphabet[ii]

def simulate(prob, alphabet, N=1000):
    # need a mapping from token to index in the transition matrix
    char2idx = {}
    for ii, c in enumerate(alphabet):
        char2idx[c] = ii
    
    last_tokens = seed
    sim_text = ''.join(seed)
    for ii in range(N):
        idxs = tuple([char2idx[x] for x in last_tokens])
        next_token = draw_token(prob[idxs], alphabet)
        sim_text = sim_text + next_token
    return sim_text

if __name__ == '__main__':
    file_in = sys.argv[1]
    N = int(sys.argv[2]) # number of tokens to emit
    seed = sys.argv[3:]
    
    # load the transition matrix
    alphabet, transition = pickle.load(open(file_in, 'rb'))
    prob = count_matrix_to_prob(transition)

    if not len(seed) == len(prob.shape) - 1:
        raise Exception(f"Please provide a seed consisting of " +
                        f"{len(prob.shape) - 1} tokens")

    print(simulate(prob, alphabet))
