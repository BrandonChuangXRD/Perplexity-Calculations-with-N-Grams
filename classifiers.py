import numpy as np
import math
from ngram import *

class MaxLikelihoodEst():
    #this should still work with bigram and trigrams, but you need to transform it.
    def eval_MLE(self, ngram_feats, Xi): 
        prob = 1
        print(Xi)
        for wi in Xi:
            # print(wi)
            # print("unigram tokens", ngram_feats.grams[wi])
            ug = ngram_feats.grams[wi]
            # print("words", ngram_feats.unigrams.grams[ug[0]], "given", ngram_feats.unigrams.grams[ug[1]])
            # print("count", ngram_feats.unigrams.token_counts[ug[0]], ngram_feats.unigrams.token_counts[ug[1]])
            # print("bigram count: ", ngram_feats.token_counts[wi])
            # print("probability", ngram_feats.token_prob[wi])
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
        sum *= -1 / 3
        print(sum)
        return 2**sum
