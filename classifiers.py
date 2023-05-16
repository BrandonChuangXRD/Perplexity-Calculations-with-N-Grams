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
        print(no_start_word_count)
        for i, count in ngram_feats.token_counts.items():
            if (ngram_feats.grams[i] == "<START>"):
                continue
            print(ngram_feats.grams[i], count)
            sum += math.log((count / no_start_word_count), 2) * count
        sum *= -1 / no_start_word_count
        print(sum)
        return 2**sum
