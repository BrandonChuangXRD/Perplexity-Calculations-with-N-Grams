usage: main.py [-h] [--smooth] [--ngram NGRAM] [--train TRAIN] [--predict PREDICT]

options:
  -h, --help            show this help message and exit
  --smooth, -s          enable smoothing for the MLE
  --ngram NGRAM, -g NGRAM
                        choose between unigram(1), bigram(2), and trigram(3)
  --train TRAIN, -t TRAIN
                        choose training token file
  --predict PREDICT, -p PREDICT
                        choose token file to predict on

UNIGRAM FIT TIME: 0.40364766120910645 seconds
BIGRAM FIT TIME: 106.7597439289093 seconds
