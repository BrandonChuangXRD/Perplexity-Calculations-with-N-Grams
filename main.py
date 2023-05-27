import pandas as pd
from classifiers import *
from ngram import *
import numpy as np
import time
import argparse

#copied from asgn1 starter code
# def accuracy(pred, labels):
#     correct = (np.array(pred) == np.array(labels)).sum()
#     accuracy = correct/len(pred)
#     print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))

def test_perplexity_phrase(mle, ngram_features):
    phrase = ["HDTV ."]
    phrase = ngram_features.transform_list(phrase)
    perp = mle.eval_perplexity(ngram_features, phrase)
    print("\"HDTV .\" perplexity:", perp)
    pass

def calc_perplexity(mle, ngram_features, text_file):
    tokenized = ngram_features.transform_list(text_file)
    perp = mle.eval_perplexity(ngram_features, tokenized)
    print("perplexity:", perp)
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", "-s", action="store_true", help="enable smoothing for the MLE")
    parser.add_argument("--ngram", "-g", type=int, default=1, help="choose between unigram(1), bigram(2), and trigram(3)")
    parser.add_argument("--train", "-t", default="data/1b_benchmark.train.tokens", help="choose training token file")
    parser.add_argument("--predict", "-p", default="data/1b_benchmark.dev.tokens", help="choose token file to predict on")
    parser.add_argument("--alpha", "-a", default = 0, type=int, help="Choose value for add smoothing")
    args = parser.parse_args()

    ngram_features = None
    if args.ngram == 3:
        ngram_features = TrigramFeature()
    elif args.ngram == 2:
        ngram_features = BigramFeature()
    elif args.ngram == 1:
        ngram_features = UnigramFeature()
    else:
        print("ERROR: ngram must be 1, 2 or 3")
        return 1
    ngram_features.set_alpha(args.alpha)
    train_file_name = args.train
    predict_file_name = args.predict

    mle = MaxLikelihoodEst()

    train_file = open(train_file_name, "r")
    predict_file = open(predict_file_name, "r")
    #predict_file = open(predict_file_name, "r")

    ngram_features.fit(train_file)
    #print(ngram_features.word_count)
    test_perplexity_phrase(mle, ngram_features)

    #calc_perplexity(mle, ngram_features, predict_file)
    #calc_perplexity(mle, ngram_features, predict_file)

    #run this again to make sure the add 1 smoothing does not change when adding more tokens
    test_perplexity_phrase(mle, ngram_features)

    train_file.seek(0)

    

    return 0



if __name__ == '__main__':
    main()