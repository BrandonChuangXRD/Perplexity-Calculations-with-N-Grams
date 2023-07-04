import numpy as np
import math
from ngram import *


class MaxLikelihoodEst:
    def eval_linear_MLE(self, tri, lambdas, Xi):
        bi = tri.bi
        uni = bi.uni
        prob = 0
        for wi in Xi:
            sum = 0.0
            # uni probability
            sum += lambdas[0] * uni.get_prob(wi[0])
            # bi probability
            sum += lambdas[1] * bi.get_prob((wi[0], wi[1]))
            # tri probability
            sum += lambdas[2] * tri.get_prob(wi)
            prob += math.log(sum, 2)
        return prob

    def eval_linear_perplexity(self, tri, lambdas, X):
        print("lambdas:", lambdas)
        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += len(Xi)
            curr_mle = self.eval_linear_MLE(tri, lambdas, Xi)
            if curr_mle == 0:
                return float("inf")
            sum += curr_mle
        sum *= -1 / corpus_count
        return 2**sum

    def eval_MLE(self, ngram, Xi):
        prob = 0
        for wi in Xi:
            if ngram.get_prob(wi) == 0:
                return float("-inf")
            prob += math.log(ngram.get_prob(wi), 2)
        return prob

    def eval_perplexity(self, ngram_feats, X):
        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += len(Xi)
            curr_mle = self.eval_MLE(ngram_feats, Xi)
            if curr_mle == 0:
                return float("inf")
            sum += curr_mle
        sum *= -1 / corpus_count
        return 2**sum
