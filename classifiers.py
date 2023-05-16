import numpy as np
import math
from ngram import *

class MaxLikelihoodEst():
    #this should still work with bigram and trigrams, but you need to transform it.
    def eval_MLE(self, ngram_feats, Xi): 
        prob = 1
        for wi in range(len(Xi)):
            #ignore "<START>"
            if wi == ngram_feats.start_index:
                continue
            if Xi[wi] > 0:
                print("token found")
                prob *= (ngram_feats.token_prob[wi])*Xi[wi]
        print(f"MLE:", prob)
        return prob

    def eval_perplexity(self, ngram_feats, X):
        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += Xi.sum()
            sum += math.log(self.eval_MLE(ngram_feats, Xi), 2)
        sum *= -1 / corpus_count
        print(sum)
        return 2**sum
