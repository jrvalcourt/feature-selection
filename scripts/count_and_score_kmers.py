import gzip
import re
import string
import sys
import numpy as np
from numbers import Number
import percache
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# remove all non-letter characters and make them all uppercase
def clean_line(l):
    l = l.decode('UTF-8')
    l = l.upper()
    regex = re.compile('[^A-Z]')
    l = regex.sub('', l)
    return l

# a generator that concatenates a bunch of files and yields them as a series 
# of k-mers consisting of just uppercase letters
def yield_kmers(files, K, verbose=False):
    queue = []
    for ii, f in enumerate(files):
        if verbose:
            print(f'\r{ii / len(files) * 100:.3f}%', end='')
        with gzip.open(f) as fin:
            for line in fin:
                line = clean_line(line)
                queue.extend(line)
                while len(queue) >= K:
                    kmer = queue[:K]
                    queue.pop(0)
                    yield kmer
    if verbose:
        print('\rDone.                        ')

# a generator that concatenates a bunch of files and yields them as a series 
# of characters
def yield_all_text(files, verbose=False):
    queue = []
    for ii, f in enumerate(files):
        if verbose:
            print(f'\r{ii / len(files) * 100:.3f}%', end='')
        with gzip.open(f) as fin:
            for line in fin:
                line = clean_line(line)
                for c in line:
                    yield c
    if verbose:
        print('\rDone.                        ')

# insert the kmer into the count dictionary
def add_to_count_dict(d, kmer):
    temp = d
    for ii, c in enumerate(kmer):
        if c in temp:
            if ii == len(kmer) - 1:
                temp[c] += 1
            else:
                temp = temp[c]
        else:
            if ii == len(kmer) - 1:
                temp[c] = 1
            else:
                temp[c] = {}
                temp = temp[c]

# returns a dictionary with the frequency counts of every kmer in the corpuses
# contained across the given files
def count_kmers(files, K):
    counts = {}
    for ii in range(K):
        counts[ii+1] = {}
    total_kmers = 0
    for kmer in yield_kmers(files, K, verbose=True):
        total_kmers += 1
        for ii in range(K):
            add_to_count_dict(counts[ii+1], kmer[:ii+1])
    return counts, total_kmers

# retrieve the count associated with the given kmer
def get_count(d, kmer):
    temp = d
    for c in kmer:
        temp = temp[c]
    assert isinstance(temp, Number)
    return temp
    
# score the kmers from a given file
def score_kmers(files, maxK, counts, total_kmers):
    ks = np.linspace(2, maxK, num=((maxK-2)//2)+1, dtype='int32')
    results = {}
    for K in ks:
        results[K] = []
        for kmer in yield_kmers(files, K):
            first_half  = kmer[:K//2]
            second_half = kmer[K//2:]
            
            # retrieve the number of times we saw the full kmer and each half
            n_full = get_count(counts[K],    kmer)
            n_1    = get_count(counts[K//2], first_half)
            n_2    = get_count(counts[K//2], second_half)

            # figure out the probabilities
            # we have to work in log space for probs to prevent underflow
            log_p_kmer = np.log(n_full) - np.log(total_kmers)
            log_p_1    = np.log(n_1)    - np.log(total_kmers)
            log_p_2    = np.log(n_2)    - np.log(total_kmers)
            log_p_ratio = log_p_kmer - (log_p_1 + log_p_2)

            results[K].append(log_p_ratio)
    return results

if __name__ == '__main__':
    ALPHABET = string.ascii_uppercase
    maxK = int(sys.argv[1])
    if not maxK % 2 == 0:
        sys.exit()
    print(f"Using maxK={maxK}")
    test_file = sys.argv[2]
    train_files = sys.argv[3:]
    counts, total_kmers = count_kmers(train_files, maxK)
    results = score_kmers([test_file], maxK, counts, total_kmers)
    
    for ii, K in enumerate(np.linspace(2, maxK, num=((maxK-2)//2)+1, dtype='int32')):
        plt.plot(np.array(range(len(results[K]))) + ii + 0.5, 
                 results[K],
                 label=f'k = {K}')
    plt.legend()
    for ii, c in enumerate(yield_all_text([test_file])):
        plt.text(ii, 0, c, ha='center')
    plt.ylabel('log(p(full kmer) / (p(firsthalf) * p(secondhalf)))')
    plt.xlabel('position')
    plt.show()