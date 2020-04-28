import count_and_score_kmers
import pickle
import hashlib
import os
import sys

if __name__ == '__main__':

    outdir = "trained_models/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    maxK = int(sys.argv[1])
    if not (maxK % 2 == 0 and maxK > 0):
        print("k must be a positive even number")
        sys.exit()

    name = sys.argv[2]
    train_files = sys.argv[3:]
    file_list_hash = count_and_score_kmers.get_file_list_hash(train_files) 

    tmp_path = os.path.join(outdir, f"{name}__k{maxK}__" + file_list_hash + '.pkl')
    if not os.path.exists(tmp_path):
        print(f"Training using maxK={maxK}")
        counts, total_kmers = count_and_score_kmers.count_kmers(train_files, 
                maxK, verbose=True)
        print(f"Storing in {tmp_path}...")
        pickle.dump((counts, total_kmers), open(tmp_path, 'wb'))
        print("Done.")
    else:
        print("Combination of files and K is already trained.")
