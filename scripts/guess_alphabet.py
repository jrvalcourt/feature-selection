from count_and_score_kmers import get_char2idx
from count_and_score_kmers import score_kmers
from count_and_score_kmers import yield_all_text
from count_and_score_kmers import get_count
from build_markov_chain import load_alphabet
import sys
import pickle
import os
import numpy as np

def candidate_breaks(log_p_trace, offset):
    log_p_trace = np.insert(log_p_trace, 0, [0] * offset)
    diffed = np.diff(log_p_trace)
    diffeddiffed = np.diff(diffed)
    diffeddiffed = np.insert(diffeddiffed, 0, [0])
    m = diffed * np.roll(diffed, 1)
    breaks = np.argwhere(np.logical_and(m < 0, diffeddiffed > 0))
    return [x[0] for x in breaks]

words = {}

if __name__ == '__main__':
    test_file = sys.argv[1]
    alphabet_file = sys.argv[2]
    train_data = sys.argv[3]

    maxK = int(os.path.basename(train_data).split('__')[1][1:])
    counts, total_kmers = pickle.load(open(train_data, 'rb'))
    results = score_kmers([test_file], maxK, counts, total_kmers)
    breaks = candidate_breaks(results, maxK//2)

    alphabet = load_alphabet(alphabet_file)
    char2idx = get_char2idx(alphabet)

    text = ''.join(yield_all_text([test_file]))
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

            chunk1 = text[b_prev:b_curr]
            chunk2 = text[b_curr:b_next]

            if len(chunk1) < 1 or len(chunk2) < 1:
                continue

            # trim the chunks if they're too long for the 
            # kmers we counted
            while len(chunk1) + len(chunk2) > maxK:
                if len(chunk1) > len(chunk2):
                    chunk1 = chunk1[1:]
                else:
                    chunk2 = chunk2[:-1]

            count_full = get_count(counts[len(chunk1 + chunk2)], chunk1 + chunk2)
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

        breaks[ii] = min_b_curr
        candidate_word = text[b_prev:b_curr]
        if not candidate_word in words:
            words[candidate_word] = 0
        words[candidate_word] += 1
#        print(min_chunk1, min_chunk2, min_log_p_ratio)

    w_list = [k for k in words]
    w_counts = [words[k] for k in words]

    sorted_words = sorted(zip(w_list, w_counts), key=lambda x: x[1], reverse=True)
    print(sorted_words[:100])

