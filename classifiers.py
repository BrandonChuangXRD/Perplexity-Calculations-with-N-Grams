import numpy as np
import math
from ngram import *

class MaxLikelihoodEst:
    def eval_linear_MLE(self, tri, lambdas, Xi):
        bi = tri.bi
        uni = bi.uni
        prob = 0
        # print(Xi)
        for wi in Xi:
            sum = 0.0
            #TODO do the probability of each word
            #uni probability
            sum += lambdas[0] * uni.get_prob(wi[0])
            #bi probability
            sum += lambdas[1] * bi.get_prob((wi[0], wi[1]))
            #tri probability
            sum += lambdas[2] * tri.get_prob(wi)
            prob += math.log(sum, 2)

        # print(f"MLE:", prob)
        return prob
    
    def eval_linear_perplexity(self, tri, lambdas, X):
        #print("X:", X)
        print("lambdas:", lambdas)

        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += len(Xi)
            curr_mle = self.eval_linear_MLE(tri, lambdas, Xi)
            if curr_mle == 0:
                return float("inf")
            sum += curr_mle
        # print("corpus count:", corpus_count)
        sum *= -1 / corpus_count
        # print(sum)
        return 2**sum

    # this should still work with bigram and trigrams, but you need to transform it.
    def eval_MLE(self, ngram, Xi):
        prob = 0
        # print(Xi)
        for wi in Xi:
            if ngram.get_prob(wi) == 0:
                return float("-inf")
            prob += math.log(ngram.get_prob(wi), 2)
        # print(f"MLE:", prob)
        return prob

    # takes
    def eval_perplexity(self, ngram_feats, X):
        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += len(Xi)
            curr_mle = self.eval_MLE(ngram_feats, Xi)
            if curr_mle == 0:
                return float("inf")
            sum += curr_mle
        # print("corpus count:", corpus_count)
        sum *= -1 / corpus_count
        # print(sum)
        return 2**sum

