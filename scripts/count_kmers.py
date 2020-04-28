import gzip
import re
import string
import sys
import numpy as np
from numbers import Number
import percache

ALPHABET = string.ascii_uppercase
cache = percache.Cache("/tmp/my-cache")

# remove all non-letter characters and make them all uppercase
def clean_line(l):
    l = l.upper()
    regex = re.compile(b'[^A-Z]')
    l = regex.sub(b'', l)
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
                while len(queue) > K:
                    kmer = queue[:K]
                    queue.pop(0)
                    yield kmer
    if verbose:
        print('\rDone.                        ')

# insert the kmer into the count dictionary
def add_to_count_dict(counts, kmer):
    temp_counts = counts
    for ii, c in enumerate(kmer):
        if c in temp_counts:
            if ii == len(kmer) - 1:
                temp_counts[c] += 1
            else:
                temp_counts = temp_counts[c]
        else:
            if ii == len(kmer) - 1:
                temp_counts[c] = 1
            else:
                temp_counts[c] = {}
                temp_counts = temp_counts[c]

# returns a dictionary with the frequency counts of every kmer in the corpuses
# contained across the given files
@cache
def count_kmers(files, K):
    counts = {}
    counts_half = {}
    total_kmers = 0
    for kmer in yield_kmers(files, K, verbose=True):
        total_kmers += 1
        add_to_count_dict(counts, kmer)
        add_to_count_dict(counts_half, kmer[:K//2])
    return counts, total_kmers, counts_half

# retrieve the count associated with the given kmer
def get_count(counts, kmer):
    temp_counts = counts
    for c in kmer:
        temp_counts = temp_counts[c]
    assert isinstance(temp_counts, Number)
    return temp_counts
    
# score the kmers from a given file
def score_kmers(files, K, counts, total_kmers, counts_half):
    for kmer in yield_kmers(files, K):
        first_half = kmer[:K//2]
        second_half = kmer[K//2:]
        
        # retrieve the number of times we saw the full kmer and each half
        n_full = get_count(counts,      kmer)
        n_1    = get_count(counts_half, first_half)
        n_2    = get_count(counts_half, second_half)

        # figure out the probabilities
        # we have to work in log space for probs to prevent underflow
        log_p_kmer = np.log(n_full) - np.log(total_kmers)
        log_p_1    = np.log(n_1)    - np.log(total_kmers)
        log_p_2    = np.log(n_2)    - np.log(total_kmers)
        log_p_ratio = log_p_kmer - (log_p_1 + log_p_2)

        if log_p_ratio < 0:
            print(kmer, log_p_ratio)

K = int(sys.argv[1])
if not K % 2 == 0:
    sys.exit()
test_file = sys.argv[2]
train_files = sys.argv[3:]
counts, total_kmers, counts_half = count_kmers(train_files, K)
score_kmers(test_file, K, counts, total_kmers, counts_half)

