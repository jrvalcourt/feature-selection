from count_and_score_kmers import get_char2idx
from count_and_score_kmers import score_kmers_corpus
from count_and_score_kmers import yield_all_text
from count_and_score_kmers import get_count
from count_and_score_kmers import count_kmers_corpus
from build_markov_chain import load_alphabet
from build_markov_chain import build_markov_chain_corpus
import sys
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def candidate_breaks(log_p_trace, offset):
    log_p_trace = np.insert(log_p_trace, 0, [0] * offset)
    diffed = np.diff(log_p_trace)
    diffeddiffed = np.diff(diffed)
    diffeddiffed = np.insert(diffeddiffed, 0, [0])
    m = diffed * np.roll(diffed, 1)
    breaks = np.argwhere(np.logical_and(m < 0, diffeddiffed > 0))
    return [x[0] for x in breaks]

def guess_next_token(corpus, breaks, alphabet, counts, total_kmers):
    words = {}
    betas = []
    for ii in range(len(breaks) - 1):
        if ii < 1:
            continue

        min_log_p_ratio = 1000000
        min_b_curr = None
        min_chunk1 = None
        min_chunk2 = None
        for w in [0]:

            b_prev = breaks[ii-1]
            b_curr = breaks[ii] + w
            b_next = breaks[ii+1]

            chunk1 = corpus[b_prev:b_curr]
            chunk2 = corpus[b_curr:b_next]

            if len(chunk1) < 1 or len(chunk2) < 1:
                continue

            # trim the chunks if they're too long for the 
            # kmers we counted
            while len(chunk1) + len(chunk2) > maxK:
                if len(chunk1) > len(chunk2):
                    chunk1 = chunk1[1:]
                else:
                    chunk2 = chunk2[:-1]

            count_full = get_count(counts[len(chunk1 + chunk2)], 
                    chunk1 + chunk2)
            count_1    = get_count(counts[len(chunk1)], chunk1)
            count_2    = get_count(counts[len(chunk2)], chunk2)

            log_p_full = np.log(count_full) - np.log(total_kmers)
            log_p_1 = np.log(count_1) - np.log(total_kmers)
            log_p_2 = np.log(count_2) - np.log(total_kmers)
            log_p_ratio = log_p_full - (log_p_1 + log_p_2)

            if log_p_ratio < min_log_p_ratio:
                min_log_p_ratio = log_p_ratio
                min_b_curr = b_curr
                min_chunk1 = chunk1
                min_chunk2 = chunk2

        betas.append(min_log_p_ratio)
        breaks[ii] = min_b_curr
        candidate_word = '_'.join(corpus[b_prev:b_curr])
        if not candidate_word in words:
            words[candidate_word] = 0
        words[candidate_word] += 1

    w_list = [k for k in words]
    w_counts = [words[k] for k in words]
    sorted_words = sorted(zip(w_list, w_counts), key=lambda x: x[1], 
            reverse=True)
    return (sorted_words[0][0].split('_'), sorted_words[0][1], betas)

def replace_token_in_corpus(corpus, new_token):
    for ii in range(len(corpus)):
        if corpus[ii:ii+len(new_token)] == new_token:
            corpus[ii:ii+len(new_token)] = [''.join(new_token)]
    return corpus

if __name__ == '__main__':
    input_file = sys.argv[1]
    alphabet_file = sys.argv[2]
    maxK = int(sys.argv[3])

    alphabet = load_alphabet(alphabet_file)
    corpus = [x for x in yield_all_text([input_file])]

    save_points = [0, 1, 2, 5, 10, 20, 50, 75, 100, 125, 150, 175, 200, 300, 400, 
                   500, 600, 750, 1000, 1500, 2000]

    # lists to hold the descriptive stats at each round
    stds = []
    means = []
    medians = []
    cvs = []
    frac_lt_zero = []

    ii = -1
    while True:
        plt.close()
        ii += 1
        char2idx = get_char2idx(alphabet)
        counts, total_kmers = count_kmers_corpus(corpus, maxK, alphabet)
        results = score_kmers_corpus(corpus, maxK, counts, total_kmers)
        breaks = candidate_breaks(results, maxK//2)
        new_token, n, betas = guess_next_token(corpus, breaks, alphabet, 
                counts, total_kmers)
        corpus = replace_token_in_corpus(corpus, new_token)
        alphabet.append(''.join(new_token))
        print(ii, new_token, n)
        sys.stdout.flush()

        betas = np.array(betas)

        stds.append(np.std(betas))
        means.append(np.mean(betas))
        medians.append(np.median(betas))
        cvs.append(np.std(betas) / np.mean(betas))
        frac_lt_zero.append(np.sum(betas < 0)/len(betas))

        if (ii in save_points) or (ii % 1000 == 0):
            plt.figure(figsize=(12,6))
            plt.hist(betas, bins=np.linspace(-4, 20, num=100))
            plt.savefig(f'plots/beta_hists/beta_hist_{ii}.png', dpi=300)
            plt.close()
            
            if ii < 5:
                continue

            plt.figure()
            plt.plot(range(ii+1), stds, 'k-')
            plt.savefig(f'plots/beta_stds/beta_stds_{ii}.png', dpi=300)
            plt.close()

            plt.figure()
            plt.plot(range(ii+1), means, 'k-')
            plt.savefig(f'plots/beta_means/beta_means_{ii}.png', dpi=300)
            plt.close()

            plt.figure()
            plt.plot(range(ii+1), medians, 'k-')
            plt.savefig(f'plots/beta_medians/beta_medians_{ii}.png', dpi=300)
            plt.close()

            plt.figure()
            plt.plot(range(ii+1), cvs, 'k-')
            plt.savefig(f'plots/beta_cvs/beta_cvs_{ii}.png', dpi=300)
            plt.close()

            plt.figure()
            plt.plot(range(ii+1), frac_lt_zero, 'k-')
            plt.savefig(f'plots/beta_frac_lt_zero/beta_frac_lt_zero_{ii}.png', dpi=300)
            plt.close()

            with open(f'alphabets/basic_round{ii}.txt', 'w') as fout:
                first = True
                for token in alphabet:
                    if not first:
                        fout.write('\n')
                    fout.write(token)
                    first = False
            
