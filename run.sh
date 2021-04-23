clear
# # Train Transformer Encoder
# python main.py -l ES -dp data/hateval2019/hateval2019_es_train.csv -mode tEncoder -tmode offline -bs 20 -epoches 12 -interm_layer 96
# python main.py -l EN -dp data/hateval2019/hateval2019_en_train.csv -mode tEncoder -tmode offline -bs 64 -epoches 12 -interm_layer 96

# Encode from Transformers
# python main.py -l ES -dp data/pan21-author-profiling-training-2021-03-14 -wp logs -mode encode -tmode offline -bs 200 -phase train -interm_layer 96
# python main.py -l EN_ -dp data/pan21-author-profiling-training-2021-03-14 -wp logs -mode encode -tmode online -bs 200 -phase train -interm_layer 96
# python main.py -l ES -dp data/pan21-author-profiling-training-2021-03-14 -wp logs -mode encode -tmode offline -bs 200 -phase test -interm_layer 96
# python main.py -l EN -dp data/pan21-author-profiling-training-2021-03-14 -wp logs -mode encode -tmode offline -bs 200 -phase test -interm_layer 96

# Train Siamese
# python main.py -l ES  -mode tSiamese -bs 64 -epoches 100 -loss triplet -lr 1e-5 -decay 0 -phase train
# python main.py -l EN  -mode tSiamese -bs 64 -epoches 100 -loss triplet -lr 1e-5 -decay 0 -phase train

# # Get Siamese Encoders
# python main.py -l ES  -mode eSiamese -bs 64 -wp logs/siamese_encoder_ES.pt -phase train
# python main.py -l EN  -mode eSiamese -bs 64 -wp logs/siamese_encoder_EN.pt -phase train
# python main.py -l ES  -mode eSiamese -bs 64 -wp logs/siamese_encoder_ES.pt -phase test
# python main.py -l EN  -mode eSiamese -bs 64 -wp logs/siamese_encoder_EN.pt -phase test

# # Metric Learning
python main.py -l ES  -mode metriclearn -bs 64 -epoches 80 -loss contrastive -lr 4e-5 -decay 1e-6 -wp logs/BSM_split_ES.pt -dp data/pan21-author-profiling-training-2021-03-14 -phase train -interm_layer 96
python main.py -l EN  -mode metriclearn -bs 64 -epoches 80 -loss contrastive -lr 1e-4 -wp logs/BSM_split_EN.pt -dp data/pan21-author-profiling-training-2021-03-14 -phase train -interm_layer 96

# # #Impostors Evaluation
# python main.py -l EN  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.4 -metric cosine -up random -ecnImp transformer -dt data/pan21-author-profiling-training-2021-03-14 -output logs -interm_layer 96
# python main.py -l ES  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.35 -metric cosine -up random -ecnImp transformer -dt data/pan21-author-profiling-training-2021-03-14 -output logs -interm_layer 96

#Impostors Evaluation deepmetric
python main.py -l EN  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 1 -up prototipical -metric deepmetric -ecnImp transformer -dt data/pan21-author-profiling-training-2021-03-14 -output logs -interm_layer 96
python main.py -l ES  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 1 -up prototipical -metric deepmetric -ecnImp transformer -dt data/pan21-author-profiling-training-2021-03-14 -output logs -interm_layer 96

#FCNN Train
# python main.py -l EN  -mode tfcnn -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 80 -lr 1e-4 -decay 0 -phase train -interm_layer 96
# python main.py -l ES  -mode tfcnn -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 80 -lr 1e-4 -decay 0 -phase train -interm_layer 96


#FCNN pred
# python main.py -l EN  -mode tfcnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase test -output logs -interm_layer 96
# python main.py -l ES  -mode tfcnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase test -output logs -interm_layer 96

#GCN
# python main.py -l EN  -mode cgnn -dp data/pan21-author-profiling-training-2021-03-14 -epoches 60 -lr 1e-3 -decay 0 -phase train -bs 16 -interm_layer 128
# python main.py -l ES  -mode cgnn -dp data/pan21-author-profiling-training-2021-03-14 -epoches 60 -lr 1e-3 -decay 0 -phase train -bs 16 -interm_layer 128

# #GCN pred
# python main.py -l EN  -mode cgnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase test -output logs -interm_layer 128
# python main.py -l ES  -mode cgnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase test -output logs -interm_layer 128
