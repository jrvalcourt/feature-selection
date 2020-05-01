# workon biopython
python scripts/train.py 10 totc gutenberg/books/Charles\ Dickens___A\ Tale\ of\ Two\ Cities.txt.gz
python scripts/plot_varying_ks.py test_sets/totc_extra_short.txt.gz trained_models/totc__k10__6b286cb247bf92ba499b9e524fbb8e68.pkl plots/varying_k__trained_on_totc.png

python scripts/train.py 10 alldickens gutenberg/books/Charles\ Dickens___*.txt.gz
python scripts/plot_varying_ks.py test_sets/totc_extra_short.txt.gz trained_models/alldickens__k10__038307874a07e8f6c73b1fbd5e52aa5b.pkl plots/varying_k__trained_on_alldickens.png

python scripts/build_markov_chain.py alphabets/basic.txt dickensnototc 1 gutenberg/collections/dickensnototc.txt.gz
