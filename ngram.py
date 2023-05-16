##################################################
###  heavily influenced by asgn1 starter code  ###
##################################################
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        self.grams = {}
        self.word_count = 0
        self.start_tokens = 0
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass


class UnigramFeature():
    def __init__(self):
        self.grams = []
        self.token_prob = [] #? Apparently we don't include <START> in these?
                             #? just calculate it and assume its an accident if you use it
        self.token_counts = []
        self.word_count = 0
        self.start_tokens = 0 #to calculate perplexity (subtract from total)
        self.unknown_index = -1
        self.start_index = -1
        self.stop_index = -1
    #this passes a file instead of the text.

    def fit(self, train_file):
        #make dictionary
        tokens = {}
        tokens["<START>"] = 0
        tokens["<STOP>"] = 0
        for l in train_file:
            #print(l)
            self.start_tokens += 1
            tokens["<START>"] += 1
            tokens["<STOP>"] += 1
            self.word_count += 2
            #print(l)
            for t in l.split(" "):
                t = t.rstrip("\n")
                #! this lower probably should be here, but it makes the number of tokens worse
                #t = t.lower()
                self.word_count += 1
                if t not in tokens.keys():
                    tokens[t] = 0
                tokens[t] += 1
        #parse dictionary
        to_delete = []
        unknowns = 0
        for i in tokens:
            if i == "<START>" or i == "<STOP>":
                continue
            if tokens[i] < 3:
                #if a token shows up less than 3 times, assign it <UNK>
                unknowns += tokens[i]
                to_delete.append(i)
        for i in to_delete:
            tokens.pop(i)
        tokens["<UNK>"] = unknowns
        
        #create final list and dictionary (dict[index] = token)
        self.grams = np.array(list(tokens.keys()))
        #assign class variables for reference during perplexity and prediction
        self.unknown_index = np.where(self.grams == "<UNK>")[0][0]
        self.start_index = np.where(self.grams == "<START>")[0][0]
        self.stop_index = np.where(self.grams == "<STOP>")[0][0]
        for i in self.grams:
            self.token_counts.append(tokens[i]) 

        #! calculate probability
        for i in range(len(self.grams)):
            p = self.token_counts[i]/(self.word_count-self.start_tokens)
            self.token_prob.append(p)
        #print(self.token_prob)
        #print(tokens)
        print("number of tokens minus \"<START>\":", len(tokens)-1)
        #should have 26602 unique tokens

    
    def transform(self, text: list):
        
        feat = np.zeros(len(self.grams))
        for i in text.split(" "):
            i = i.rstrip("\n")
            #! use lower() function for word if needed
            if i in self.grams:
                where_i = np.where(self.grams == i)[0][0]
                print(f"{i} found in unigram list")
                feat[where_i] += 1
            else:
                print(f"\"{i}\" not found in unigram list")
                feat[np.where(self.grams == "<UNK>")] += 1
        feat[self.stop_index] = 1
        return feat

    def transform_list(self, text_set: list):
        fs = []
        for i in text_set:
            fs.append(self.transform(i))
        return np.array(fs)

class BigramFeature(FeatureExtractor):
    def __init__(self):
        self.bigrams = {}

class TrigramFeature(FeatureExtractor):
    def __init__(self):
        self.trigrams = {}