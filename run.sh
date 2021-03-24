Train Transformer Encoder
python main.py -l ES -dp data/hateval2019/hateval2019_es_train.csv -mode tEncoder -tmode offline -bs 16 -epoches 2
python main.py -l EN -dp data/hateval2019/hateval2019_en_train.csv -mode tEncoder -tmode offline -bs 16 -epoches 2

Encode from Transformers
python main.py -l ES -dp data/pan21-author-profiling-training-2021-03-14 -wp bestmodelo_split_ES_1.pt -mode encode -tmode offline -bs 16 -epoches 3
python main.py -l EN -dp data/pan21-author-profiling-training-2021-03-14 -wp bestmodelo_split_EN_1.pt -mode encode -tmode offline -bs 16 -epoches 3

Train Siamese
python main.py -l ES  -mode tSiamese -bs 64 -epoches 100 -loss triplet -lr 1e-2 -decay 0
python main.py -l EN  -mode tSiamese -bs 64 -epoches 100 -loss triplet -lr 1e-2 -decay 0

Get Siamese Encoders
python main.py -l ES  -mode eSiamese -bs 64 -epoches 50 -loss triplet -lr 1e-2 -decay 0 -wp BSM_split_ES.pt
python main.py -l EN  -mode eSiamese -bs 64 -epoches 50 -loss triplet -lr 1e-2 -decay 0 -wp BSM_split_EN.pt

#Impostors Evaluation
# python main.py -l ES  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.1 -metric cosine
# python main.py -l EN  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.25 -metric euclidean


#Reynier
#python main.py -l EN -dp data/hateval2019/training.tsv -mode tEncoder -tmode offline -bs 16 -epoches 3
