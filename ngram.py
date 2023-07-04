import numpy as np
import time


class UnigramFeature:
    def __init__(self):
        self.grams_index = {}  # word: index
        self.grams_word = {}  # index: word
        self.token_counts = {}  # use in probability, uses index as the key
        self.unique_tokens = 0  # excludes <START> tokens, since its for probability
        self.total_tokens = 0
        self.alpha = 0

    def set_alpha(self, val):
        self.alpha = val
        print("add alpha set to ", self.alpha)

    # calculates probability with add_alpha value
    def get_prob(self, word):
        return (self.token_counts.get(word, 0) + self.alpha) / (
            self.total_tokens + (self.unique_tokens * self.alpha)
        )

    def fit(self, train_file):
        start_time = time.time()
        # make dictionary
        tokens = {}
        tokens["<START>"] = 0
        tokens["<STOP>"] = 0
        for l in train_file:
            tokens["<START>"] += 1
            tokens["<STOP>"] += 1
            self.total_tokens += 1
            for t in l.rstrip("\n").split(" "):
                self.total_tokens += 1
                if t not in tokens.keys():
                    tokens[t] = 0
                tokens[t] += 1
        # parse dictionary
        to_delete = []
        unknowns = 0
        for i in tokens:
            if i == "<START>" or i == "<STOP>":
                continue
            if tokens[i] < 3:
                # if a token shows up less than 3 times, assign it <UNK>
                unknowns += tokens[i]
                to_delete.append(i)
        for i in to_delete:
            tokens.pop(i)
        tokens["<UNK>"] = unknowns

        # create final list and dictionary (dict[index] = token)
        gramlist = list(tokens.keys())
        self.unique_tokens = len(gramlist) - 1
        for i in range(len(gramlist)):
            self.grams_index[gramlist[i]] = i
            self.grams_word[i] = gramlist[i]
            self.token_counts[i] = tokens[gramlist[i]]

        # should have 26602 unique tokens
        print('UNIGRAM: number of tokens minus "<START>":', len(tokens) - 1)
        train_file.seek(0)
        end_time = time.time()
        print("UNIGRAM FIT TIME:", end_time - start_time, "seconds")

    # makes a list in order with the token instead of the word
    def transform(self, text):
        feat = []
        text = text.rstrip("\n")
        for i in text.split(" "):
            if i in self.grams_index.keys():
                where_i = self.grams_index[i]
                feat.append(where_i)
            else:
                feat.append(self.grams_index["<UNK>"])
        feat.append(self.grams_index["<STOP>"])
        return feat

    def transform_list(self, text_set: list):
        fs = []
        print("UNIGRAM: TRANFORMING LIST")
        for i in text_set:
            fs.append(self.transform(i))
        return fs


class BigramFeature:
    # no need for a count of unique tokens
    def __init__(self):
        self.token_counts = {}
        self.alpha = 0  # make a function to change this
        self.uni = UnigramFeature()  # useful shortcut

    # to split the list of unigrams into a list of bigrams
    def bi_splitter(self, X):
        bigram_split = []
        # do this for <START> at the beginning
        bigram_split.append((X[0], self.uni.grams_index["<START>"]))
        for i in range(1, len(X)):
            # [token, given token] for given probability
            bigram_split.append((X[i], X[i - 1]))
        return bigram_split

    # takes a token (which is a set of two numbers)
    def get_prob(self, word):
        if word not in self.token_counts.keys():
            if self.alpha == 0:
                return 0
            return self.alpha / (
                self.uni.token_counts[word[1]] + (self.uni.unique_tokens * self.alpha)
            )
        return (self.token_counts[word] + self.alpha) / (
            self.uni.token_counts[word[1]] + (self.uni.unique_tokens * self.alpha)
        )

    def set_alpha(self, val):
        self.alpha = val

    def fit(self, train_file):
        start_time = time.time()
        # the values represent the number of observed outcomes under the key.
        # useful for probability calculation
        print("BIGRAM: fitting")

        self.uni.fit(train_file)  # use this to replace OOV in train_file with <UNK>

        # this is the time consuming part
        X = train_file.readlines()
        for Xi in X:
            # TODO try just doing this part manually see if it makes a difference
            Xi_tkn = [self.uni.grams_index["<START>"]] + self.uni.transform(Xi)
            for i in range(1, len(Xi_tkn)):
                curr = (Xi_tkn[i], Xi_tkn[i - 1])
                if curr not in self.token_counts.keys():
                    self.token_counts[curr] = 0
                self.token_counts[curr] += 1

        print(
            'BIGRAM: number of tokens minus "<START>":',
            len(self.token_counts.keys()) - 1,
        )
        train_file.seek(0)
        end_time = time.time()
        print("BIGRAM FIT TIME:", end_time - start_time, "seconds")

    def transform(self, Xi):
        unigram_tkns = self.uni.transform(Xi)
        # this gives a list of lists for w|w-1
        bigram_tkns = self.bi_splitter(unigram_tkns)
        return bigram_tkns

    def transform_list(self, text_set):
        fs = []
        for i in text_set:
            # print(i)
            fs.append(self.transform(i))
        return fs


class TrigramFeature:
    def __init__(self):
        self.token_counts = {}
        self.alpha = 0  # make a function to change this
        self.bi = BigramFeature()  # useful shortcut

    # takes an already unigrammed list of tokens
    # returns a tuple with three unigram indices
    def tri_splitter(self, sentence):
        # print("bigrammed: ", sentence)
        Xi_trigram = [(sentence[1], sentence[0])]
        for i in range(2, len(sentence)):
            Xi_trigram.append((sentence[i], sentence[i - 1], sentence[i - 2]))
        return Xi_trigram

    def set_alpha(self, val):
        self.alpha = val

    # takes a tuple with 3 unigram indexes (w2, w1, w0)
    def get_prob(self, word):
        if len(word) == 2:
            if word in self.token_counts.keys():
                return (self.token_counts[word] + self.alpha) / (
                    self.bi.uni.token_counts[word[1]]
                    + (self.bi.uni.unique_tokens * self.alpha)
                )
            else:
                if self.alpha == 0:
                    return 0
                return self.alpha / (
                    self.bi.uni.token_counts[word[1]]
                    + (self.bi.uni.unique_tokens * self.alpha)
                )

        prior = (word[1], word[2])
        prior_prob = 0
        if prior in self.bi.token_counts.keys():
            prior_prob = self.bi.token_counts[prior] + (
                self.bi.uni.unique_tokens * self.alpha
            )
        else:
            prior_prob = self.bi.uni.unique_tokens * self.alpha
        if word not in self.token_counts.keys():
            if self.alpha == 0:
                return 0
            return (self.alpha) / prior_prob
        return (self.token_counts[word] + self.alpha) / prior_prob

    def fit(self, train_file):
        start_time = time.time()
        print("TRIGRAM: fitting")
        self.bi.fit(train_file)

        X = train_file.readlines()
        for Xi in X:
            Xi_tkn = [self.bi.uni.grams_index["<START>"]] + self.bi.uni.transform(Xi)
            # the first term is a bigram
            if (Xi_tkn[1], Xi_tkn[0]) not in self.token_counts.keys():
                self.token_counts[(Xi_tkn[1], Xi_tkn[0])] = 0
            self.token_counts[(Xi_tkn[1], Xi_tkn[0])] += 1

            for i in range(2, len(Xi_tkn)):
                wi = (Xi_tkn[i], Xi_tkn[i - 1], Xi_tkn[i - 2])
                if wi not in self.token_counts.keys():
                    self.token_counts[wi] = 0
                self.token_counts[wi] += 1

        end_time = time.time()
        print("TRIGRAM FIT TIME:", end_time - start_time, "seconds")

    def transform(self, Xi):
        Xi_unigrammed = self.bi.uni.transform(Xi)
        Xi_unigrammed = [self.bi.uni.grams_index["<START>"]] + Xi_unigrammed
        Xi_trigrammed = self.tri_splitter(Xi_unigrammed)
        return Xi_trigrammed

    def transform_list(self, text_set):
        fs = []
        for i in text_set:
            fs.append(self.transform(i))
        return fs
