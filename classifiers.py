import numpy as np
import math
from ngram import *

class MaxLikelihoodEst():
    #this should still work with bigram and trigrams, but you need to transform it.
    def eval_MLE(self, ngram_feats, Xi): 
        prob = 0
        #print(Xi)
        for wi in Xi:
            #! this will be changed in additive smoothing
            if wi == -1:
                print("unknown combination")
                return 0
            # print(wi)
            # print("unigram tokens", ngram_feats.grams[wi])
            # print("words", ngram_feats.unigrams.grams[ug[0]], "given", ngram_feats.unigrams.grams[ug[1]])
            # print("count", ngram_feats.unigrams.token_counts[ug[0]], ngram_feats.unigrams.token_counts[ug[1]])
            # print("bigram count: ", ngram_feats.token_counts[wi])
            # print("probability", ngram_feats.token_prob[wi])
            if(ngram_feats.token_prob[wi] == 0):
                print("Probability of Zero")
                return 0
            prob += math.log(ngram_feats.token_prob[wi], 2)
        #print(f"MLE:", prob)
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
        print("corpus count:", corpus_count)
        sum *= -1 / corpus_count
        print(sum)
        return 2**sum
