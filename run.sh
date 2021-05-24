#!/bin/bash
clear
hs=64
data_test=data/pan21-author-profiling-test-without-gold

# # Train Transformer Encoder
# python main.py -l ES -dp data/hateval2019/hateval2019_es_train.csv -mode tEncoder -tmode online -bs 64 -epoches 12 -interm_layer $hs
# python main.py -l EN -dp data/hateval2019/hateval2019_en_train.csv -mode tEncoder -tmode online -bs 64 -epoches 12 -interm_layer $hs

# Encode from Transformers
# python main.py -l ES -dp data/pan21-author-profiling-training-2021-03-14 -wp logs -mode encode -tmode online -bs 200 -phase train -interm_layer $hs
# python main.py -l EN -dp data/pan21-author-profiling-training-2021-03-14 -wp logs -mode encode -tmode online -bs 200 -phase train -interm_layer $hs
# python main.py -l ES -dp $data_test -wp logs -mode encode -tmode online -bs 200 -phase test -interm_layer $hs
# python main.py -l EN -dp $data_test -wp logs -mode encode -tmode online -bs 200 -phase test -interm_layer $hs

# # Train Siamese
# python main.py -l ES  -mode tSiamese -bs 64 -epoches 100 -loss triplet -lr 1e-5 -decay 0 -phase train
# python main.py -l EN  -mode tSiamese -bs 64 -epoches 100 -loss triplet -lr 1e-5 -decay 0 -phase train

# # Get Siamese Encoders
# python main.py -l ES  -mode eSiamese -bs 64 -wp logs/siamese_encoder_ES.pt -phase train
# python main.py -l EN  -mode eSiamese -bs 64 -wp logs/siamese_encoder_EN.pt -phase train
# python main.py -l ES  -mode eSiamese -bs 64 -wp logs/siamese_encoder_ES.pt -phase test
# python main.py -l EN  -mode eSiamese -bs 64 -wp logs/siamese_encoder_EN.pt -phase test

# Metric Learning
# python main.py -l ES  -mode metriclearn -bs 64 -epoches 80 -loss contrastive -lr 4e-5 -decay 1e-6 -wp logs/BSM_split_ES.pt -dp data/pan21-author-profiling-training-2021-03-14 -phase train -interm_layer $hs
# python main.py -l EN  -mode metriclearn -bs 64 -epoches 80 -loss contrastive -lr 1e-4 -decay 1e-6 -wp logs/BSM_split_EN.pt -dp data/pan21-author-profiling-training-2021-03-14 -phase train -interm_layer $hs

# # # #Impostors Evaluation
python main.py -l EN  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.40 -metric cosine -up random -ecnImp transformer -dt $data_test -output logs -interm_layer 64
python main.py -l ES  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 0.45 -metric cosine -up random -ecnImp transformer -dt $data_test -output logs -interm_layer 64

#Impostors Evaluation deepmetric
# python main.py -l EN  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 1 -up prototipical -metric deepmetric -ecnImp transformer -dt data/pan21-author-profiling-test-without-gold -output logs -interm_layer $hs
# python main.py -l ES  -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp 1 -up prototipical -metric deepmetric -ecnImp transformer -dt data/pan21-author-profiling-test-without-gold -output logs -interm_layer $hs

# #FCNN Train
# python main.py -l EN  -mode tfcnn -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 80 -lr 1e-4 -decay 0 -phase train -interm_layer $hs
# python main.py -l ES  -mode tfcnn -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 80 -lr 1e-4 -decay 0 -phase train -interm_layer $hs

#FCNN pred
# python main.py -l EN  -mode tfcnn -dt data/pan21-author-profiling-test-without-gold -bs 32 -phase test -output logs -interm_layer 96
# python main.py -l ES  -mode tfcnn -dt data/pan21-author-profiling-test-without-gold -bs 32 -phase test -output logs -interm_layer 96

#FCNN encode
# python main.py -l EN  -mode tfcnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase encode -output logs -interm_layer $hs
# python main.py -l ES  -mode tfcnn -dt data/pan21-author-profiling-training-2021-03-14 -bs 32 -phase encode -output logs -interm_layer $hs
# python main.py -l EN  -mode tfcnn -dt data/pan21-author-profiling-test-without-gold -bs 32 -phase encode -output logs -interm_layer $hs
# python main.py -l ES  -mode tfcnn -dt data/pan21-author-profiling-test-without-gold -bs 32 -phase encode -output logs -interm_layer $hs


#GCN
# python main.py -l EN  -mode cgnn -dp data/pan21-author-profiling-training-2021-03-14 -epoches 60 -lr 1e-4 -decay 0 -phase train -bs 16 -interm_layer $hs
# python main.py -l ES  -mode cgnn -dp data/pan21-author-profiling-training-2021-03-14 -epoches 60 -lr 1e-4 -decay 0 -phase train -bs 16 -interm_layer $hs

# #GCN pred
# python main.py -l EN  -mode cgnn -dt data/pan21-author-profiling-test-without-gold -bs 32 -phase test -output logs -interm_layer 64
# python main.py -l ES  -mode cgnn -dt data/pan21-author-profiling-test-without-gold -bs 32 -phase test -output logs -interm_layer 64

# #GCN encode
# python main.py -l EN -wp logs -mode cgnn -dp data/pan21-author-profiling-training-2021-03-14 -phase encode -bs 16 -interm_layer $hs
# python main.py -l ES -wp logs -mode cgnn -dp data/pan21-author-profiling-training-2021-03-14 -phase encode -bs 16 -interm_layer $hs
# python main.py -l EN -wp logs -mode cgnn -dp data/pan21-author-profiling-test-without-gold -phase encode -bs 16 -interm_layer $hs
# python main.py -l ES -wp logs -mode cgnn -dp data/pan21-author-profiling-test-without-gold -phase encode -bs 16 -interm_layer $hs


# LSTM Train
# python main.py -l EN  -mode lstm -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 60 -lr 1e-3 -decay 0 -phase train -interm_layer $hs -lstm_size 64
# python main.py -l ES  -mode lstm -dp data/pan21-author-profiling-training-2021-03-14 -bs 32 -epoches 60 -lr 4e-4 -decay 0 -phase train -interm_layer $hs -lstm_size 64

#LSTM Pred
# python main.py -l EN  -mode lstm -dt data/pan21-author-profiling-test-without-gold -output logs -phase test -interm_layer $hs -lstm_size 64
# python main.py -l ES  -mode lstm -dt data/pan21-author-profiling-test-without-gold -output logs -phase test -interm_layer $hs -lstm_size 64

#LSTM encode
# python main.py -l EN  -mode lstm -dt data/pan21-author-profiling-training-2021-03-14 -output logs -phase encode -interm_layer $hs -lstm_size 64
# python main.py -l ES  -mode lstm -dt data/pan21-author-profiling-training-2021-03-14 -output logs -phase encode -interm_layer $hs -lstm_size 64
# python main.py -l EN  -mode lstm -dt data/pan21-author-profiling-test-without-gold -output logs -phase encode -interm_layer $hs -lstm_size 64
# python main.py -l ES  -mode lstm -dt data/pan21-author-profiling-test-without-gold -output logs -phase encode -interm_layer $hs -lstm_size 64


#ML
# python main.py -mode svm -l EN -phase train -dp data/pan21-author-profiling-training-2021-03-14
# python main.py -mode svm -l ES -phase train -dp data/pan21-author-profiling-training-2021-03-14

# CPP
# python main.py -mode cpp -dp logs/lol -output /home/nitro/Desktop

# python main.py -l EN -mode gmu -phase train -dp data/pan21-author-profiling-training-2021-03-14  -bs 32 -epoches 80 -lr 1e-4 -decay 0 -phase train -interm_layer $hs
# python main.py -l ES -mode gmu -phase train -dp data/pan21-author-profiling-training-2021-03-14  -bs 32 -epoches 80 -lr 1e-4 -decay 0 -phase train -interm_layer $hs