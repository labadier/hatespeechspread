clear
# # Train Transformer Encoder
# python main.py -l ES -dp data/hateval2019/hateval2019_es_train.csv -mode tEncoder -tmode offline -bs 16 -epoches 2
# python main.py -l EN -dp data/hateval2019/hateval2019_en_train.csv -mode tEncoder -tmode offline -bs 16 -epoches 2

# # Encode from Transformers
# python main.py -l ES -dp data/pan21-author-profiling-training-2021-03-14 -wp logs/bestmodelo_split_ES_1.pt -mode encode -tmode offline -bs 16 -pahse train
# python main.py -l EN -dp data/pan21-author-profiling-training-2021-03-14 -wp logs/bestmodelo_split_EN_1.pt -mode encode -tmode offline -bs 16 -pahse train
# python main.py -l ES -dp data/pan21-author-profiling-training-2021-03-14 -wp logs/bestmodelo_split_ES_1.pt -mode encode -tmode offline -bs 16 -pahse test
# python main.py -l EN -dp data/pan21-author-profiling-training-2021-03-14 -wp logs/bestmodelo_split_EN_1.pt -mode encode -tmode offline -bs 16 -pahse test

# Train Siamese
# python main.py -l ES  -mode tSiamese -bs 64 -epoches 100 -loss contrastive -lr 1e-3 -decay 0 -phase train
# python main.py -l EN  -mode tSiamese -bs 64 -epoches 100 -loss contrastive -lr 1e-3 -decay 0 -phase train

# # Get Siamese Encoders
# python main.py -l ES  -mode eSiamese -bs 64 -wp logs/siamese_encoder_ES.pt -phase train
# python main.py -l EN  -mode eSiamese -bs 64 -wp logs/siamese_encoder_EN.pt -phase train
# python main.py -l ES  -mode eSiamese -bs 64 -wp logs/siamese_encoder_ES.pt -phase test
# python main.py -l EN  -mode eSiamese -bs 64 -wp logs/siamese_encoder_EN.pt -phase test

# Metric Learning
# python main.py -l ES  -mode metriclearn -bs 64 -epoches 120 -loss contrastive -lr 1e-3 -decay 0 -wp logs/BSM_split_ES.pt -dp data/pan21-author-profiling-training-2021-03-14 -phase train
# python main.py -l EN  -mode metriclearn -bs 64 -epoches 120 -loss contrastive -lr 1e-3 -decay 0 -wp logs/BSM_split_EN.pt -dp data/pan21-author-profiling-training-2021-03-14 -phase train

#Impostors Evaluation
python main.py -l EN  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.4 -metric cosine -ecnImp trasformer -dt data/pan21-author-profiling-training-2021-03-14 -output logs
python main.py -l ES  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.45 -metric cosine -ecnImp trasformer -dt data/pan21-author-profiling-training-2021-03-14 -output logs

#FCNN Train
# python main.py -l EN  -mode tfcnn -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 120 -lr 1e-3 -decay 0 -phase train
# python main.py -l ES  -mode tfcnn -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 120 -lr 1e-3 -decay 0 -phase train


#FCNN pred
# python main.py -l EN  -mode tfcnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase test -output logs
# python main.py -l ES  -mode tfcnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase test -output logs

#Reynier
#python main.py -l EN -dp data/hateval2019/training.tsv -mode tEncoder -tmode offline -bs 16 -epoches 3
