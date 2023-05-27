import pandas as pd
from classifiers import *
from ngram import *
import numpy as np
import time
import argparse

BILLION_TEST_FILE = "data/1b_benchmark.test.tokens"
BILLION_TRAIN_FILE = "data/1b_benchmark.train.tokens"
BILLION_DEV_FILE = "data/1b_benchmark.dev.tokens"

def test_perplexity_phrase(mle, ngram_features):
    phrase = ["HDTV ."]
    phrase = ngram_features.transform_list(phrase)
    perp = mle.eval_perplexity(ngram_features, phrase)
    print("\"HDTV .\" perplexity:", perp)
    pass

def calc_perplexity(mle, ngram_features, text_file):
    tokenized = ngram_features.transform_list(text_file)
    perp = mle.eval_perplexity(ngram_features, tokenized)
    return perp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram", "-g", type=int, default=1, help="choose between unigram(1), bigram(2), and trigram(3)")
    parser.add_argument("--train", "-t", default=BILLION_TRAIN_FILE, help="choose training token file")
    parser.add_argument("--predict", "-p", default=BILLION_DEV_FILE, help="choose token file to predict on")
    parser.add_argument("--full", action="store_true", help="find perplexities of sanity, train, and dev tokens (excludes test)")
    parser.add_argument("--test", action="store_true", help="find perplexities of test tokens")
    parser.add_argument("--linear", action="store_true", help="toggles linear interpolation (superseeds add-1 smoothing)")
    parser.add_argument("--smooth", "-s", action="store_true", help="enable smoothing for the MLE")
    parser.add_argument("--sanity", action="store_true", help="test case for \"HDTV .\"")
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
    if args.linear and args.alpha > 0:
        print("WARNING: --alpha negated by --linear (conflicting flags)")
    if not args.linear:
        ngram_features.set_alpha(args.alpha)
    train_file_name = args.train
    predict_file_name = args.predict

    mle = MaxLikelihoodEst()

    train_file = open(train_file_name, "r")

    ngram_features.fit(train_file)
    train_file.close()
    print("----------------------------------------------------")
    print("")
    print("")

    if args.sanity:
        test_perplexity_phrase(mle, ngram_features)
        print("")

    if args.full:
        if not args.sanity:
            test_perplexity_phrase(mle, ngram_features)
            print("")
        
        train_predict = open(BILLION_TRAIN_FILE, "r")
        val = calc_perplexity(mle, ngram_features, train_predict)
        print(BILLION_TRAIN_FILE, "perplexity:", val)
        print("")
        train_predict.close()

        dev_predict = open(BILLION_DEV_FILE, "r")
        val = calc_perplexity(mle, ngram_features, dev_predict)
        print(BILLION_DEV_FILE, "perplexity:", val)
        print("")
        dev_predict.close()

    if not args.full:
        predict_file = open(predict_file_name, "r")
        perp = calc_perplexity(mle, ngram_features, predict_file)
        print(args.predict, "perplexity:", perp)
        predict_file.close()

    if args.test:
        test_file = open(BILLION_TEST_FILE, "r")
        perp = calc_perplexity(mle, ngram_features, test_file)
        print(BILLION_TEST_FILE, "perplexity:", perp)
        test_file.close()


    return 0



if __name__ == '__main__':
    main()