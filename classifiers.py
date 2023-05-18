import numpy as np
import math
from ngram import *

class MaxLikelihoodEst():
    #this should still work with bigram and trigrams, but you need to transform it.
    def eval_MLE(self, ngram_feats, Xi): 
        prob = 1
        for wi in Xi:
            print(ngram_feats.token_prob[wi])
            prob *= ngram_feats.token_prob[wi]
        print(f"MLE:", prob)
        return prob

    def eval_perplexity(self, ngram_feats, X):
        sum = 0
        corpus_count = 0
        for Xi in X:
            corpus_count += len(Xi)
            sum += math.log(self.eval_MLE(ngram_feats, Xi), 2)
        print("corpus count:", corpus_count)
        sum *= -1 / corpus_count
        print(sum)
        return 2**sum
