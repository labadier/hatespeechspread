# python main.py -l ES  -mode tSiamese -bs 64 -epoches 50 -loss triplet -lr 1e-2 -decay 0
python main.py -l ES  -mode eSiamese -bs 64 -epoches 50 -loss triplet -lr 1e-2 -decay 0 -wp BSM_split_ES.pt
# python main.py -l ES  -mode tSiamese -bs 64 -epoches 50 -loss contrastive -lr 3e-4 -decay 0
# python main.py -l ES -dp data/pan20-author-profiling-training-2020-02-23 -wp bestmodelo_split_ES_1.pt -mode encode -tmode offline -bs 16 -epoches 3
# python main.py -l EN -dp data/pan20-author-profiling-training-2020-02-23 -wp bestmodelo_split_EN_1.pt -mode encode -tmode offline -bs 16 -epoches 3
# python main.py -l ES -dp data/hateval2019/hateval2019_es_train.csv -mode tEncoder -tmode offline -bs 16 -epoches 3
# python main.py -l EN -dp data/hateval2019/hateval2019_en_train.csv -mode tEncoder -tmode offline -bs 16 -epoches 3
