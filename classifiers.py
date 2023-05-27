import numpy as np
import math
from ngram import *

class MaxLikelihoodEst():
    #this should still work with bigram and trigrams, but you need to transform it.
    def eval_MLE(self, ngram, Xi): 
        prob = 0
        #print(Xi)
        for wi in Xi:
            if ngram.get_prob(wi) == 0:
                return float("-inf")
            prob += math.log(ngram.get_prob(wi), 2)
        #print(f"MLE:", prob)
        return prob

    #takes 
    def eval_perplexity(self, ngram_feats, X):
        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += len(Xi)
            curr_mle = self.eval_MLE(ngram_feats, Xi)
            if curr_mle == 0:
                return float("inf")
            sum += curr_mle
        print("corpus count:", corpus_count)
        sum *= -1 / corpus_count
        print(sum)
        return 2**sum
