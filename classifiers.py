import numpy as np
import math
from ngram import *

class MaxLikelihoodEst():
    #this should still work with bigram and trigrams, but you need to transform it.
    def eval_perplexity(self, ngram_feats, X):
        #print("start word count", ngram_feats.start_tokens)
        # for i in X:
        #     if i > 0:
        #         print(i)
        sum = 0
        #print(ngram_feats.unknown_index)
        #print(len(ngram_feats.token_counts.keys()), len(ngram_feats.grams))
        
        no_start_word_count = ngram_feats.word_count-ngram_feats.start_tokens
        X[ngram_feats.stop_index] += 1
        for i in range(len(X)):
            i_val = X[i]
            if i_val > 0:
                i_count = ngram_feats.token_counts[i]
                print(i_count, ":", no_start_word_count, (i_count/no_start_word_count))
                sum += math.log((i_count/no_start_word_count), 2)*i_val
            else:
                sum += math.log((ngram_feats.token_counts[ngram_feats.unknown_index]/no_start_word_count), 2)*i_val
        sum /= (-1*no_start_word_count)
        return 2**sum
