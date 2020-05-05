import count_and_score_kmers
import sys
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_file = sys.argv[1]
    train_data = sys.argv[2]

    maxK = int(os.path.basename(train_data).split('__')[1][1:])
    counts, total_kmers = pickle.load(open(train_data, 'rb'))
    results = count_and_score_kmers.score_kmers_krange([test_file], maxK, counts, total_kmers)
    print(results)

    plt.figure(figsize=(12,6))
    for ii, K in enumerate(np.linspace(2, maxK, num=((maxK-2)//2)+1, dtype='int32')):
        plt.plot(np.array(range(len(results[K]))) + ii + 0.5,
                 results[K],
                 label=f'k = {K}')
    plt.legend()
    for ii, c in enumerate(count_and_score_kmers.yield_all_text([test_file])):
        plt.text(ii, 0, c, ha='center')
    plt.ylabel('log(p(full kmer) / (p(firsthalf) * p(secondhalf)))')
    plt.xlabel('position')
    plt.savefig(sys.argv[3], dpi=300)
