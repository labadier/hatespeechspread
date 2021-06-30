Accompanying code for participating system at PAN 2021: Profiling Hate Speech Spreaders on Twitter. See this [link](http://ceur-ws.org/Vol-2765/paper134.pdf) for our paper with the system description.

### Usage

For training or testing the model, run the main.py script with the corresponding parameters. It must be taken into account that we presented different approaches for modeling the users' profiles, for this some parameters differ from one model to another. Also, our architecture is modular, hence every module is trained independently.

#### Tweets Encoder
This module is based on pre-trained models from HuggingFace Transformer Library and pre-trained based models will be downloaded after the training process. To encode independently each tweet you need to train the Encoders as follows:

```shell
    python main.py -l <language> -dp <data-path> -mode tEncoder [-bs <bsize>]
```
Where:
- `language` is the data language. For using adapters and build a multilingual model, add a `_` at the end of the language (e.g., either `ES_` or `EN_`)
- `data-path` is the path for the training data
- `bsize` is the size of the batch for training the models. By default, it is set to 64 and 200 for training and encoding phases respectively.

To Encode the tweets you need to run the same command as before, but setting `-mode encode` and `data-path` with the path of data to be encoded.

#### Profile Modeling
In this step, we are assuming that the isolated tweets have already been encoded by the Tweets Encoder. The profile is modeled by means of an FCNN or a Spectral Graph Neural Net to train these models you must run:

```shell

    python main.py -l <language> -mode <model> -epoches <epoch> -phase train -bs 32
```

Where `model` is the kind of deep model to be employed (`cgnn` or `tfcnn` for convolutional o FCNN respectively) and `epoch` can be set to `-1` to use the default epoch hyperparameter. To encode the profiles just need to set `-phase` to `encode`.

#### Deep Impostor Method

This method needs to have available the modeling of the training profiles for making predictions on test data. For making predictions run:

```shell
    python main.py -l <language>  -mode tImpostor -dp <data-path> -rp <proportion> -metric cosine -up <pselection> -ecnImp <encoding source> -dt <data-test> -output <out>
```
Here again, we are assumed that profile modelings have been computed for training as well as for test data.

- `data-path` is the path of data for training
- `proportion` is the proportion of positive and negative classes to be taken as prototypes.
- `pselection` can be set to `random` or `prototipical` to change the approach for selecting the prototypes from random to PCA method exposed in the paper.
- `encoding source` is the way in which the profiles have been modeled, it can be set to `transformer` to use the average method exposed in the paper, `gcn` to employ the Convolutional Graph-based method or `fcnn` to use the FCNN method.
- `data-test` is the path of testing data.
- `out` is the path to save predictions.


