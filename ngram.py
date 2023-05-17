##################################################
###  heavily influenced by asgn1 starter code  ###
##################################################
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        self.grams = [] #just save these as a strings and w1|w2|w3 for bigrams/trigrams for easier debugging.
        self.token_prob = [] #should be changed to token count tbh but I'm too lazy
        self.word_count = 0
        
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass


class UnigramFeature():
    def __init__(self):
        self.grams = []
        self.token_prob = [] #? Apparently we don't include <START> in these probability calculations
                             #? just calculate it and assume its an accident if you use the <START> probability
        self.token_counts = [] #useful in building bigrams
        self.word_count = 0 #TODO remove this. Not needed since M does not refer to the training data
        self.start_index = -1
        self.stop_index = -1 #? not sure if this is needed.
    #this passes a file instead of the text.

    def fit(self, train_file):
        #make dictionary
        tokens = {}
        tokens["<START>"] = 0
        tokens["<STOP>"] = 0
        for l in train_file:
            #print(l)
            tokens["<START>"] += 1
            tokens["<STOP>"] += 1
            self.word_count += 1
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
            p = self.token_counts[i]/(self.word_count)
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
        self.grams = []
        self.token_prob = [] #should be changed to token count tbh but I'm too lazy
        self.start_tokens = 0 #basically irrelevant.
        self.word_count = 0
        self.unigrams = UnigramFeature() #!This may provide a useful shortcut.
    #to split the sentence into a list of bigrams
    def sentence_splitter(self, sentence):
        tokens = sentence.split(" ")
        bigram_split = []
        for i in range(1, len(tokens)):
                #! the delimiter to find w_i and w_{i-1} is "<|>"
                bigram_split.append(tokens[i]+"<|>"+tokens[i-1])
        return bigram_split
    
    def fit(self, train_file):
        #the values represent the number of observed outcomes under the key.
        #useful for probability calculation
        #TODO I'm thinking you can just make a unigram class in here and then use that
        #TODO to parse everything since unknowns are recorded as (<UNK>|token) instead
        #TODO of straight unknowns
        self.unigrams.fit(train_file) #use this to replace OOV in train_file with <UNK>
        
        bigrams = {}
        start_tokens = 0
        for Xi in train_file:
            Xi = Xi.rstrip("\n")
            Xi = Xi + " <STOP>"
            #count unigram tokens for future probabilities
            for wi in Xi:
                start_tokens += 0
                if wi not in unigrams.keys():
                    unigrams[wi] = 0
                unigrams[wi] += 1
            Xi_tokenized = "<START> " + Xi
            Xi_tokenized = self.sentence_splitter(Xi_tokenized)
            print(Xi_tokenized)
            for wi in Xi_tokenized:
                if wi not in bigrams.keys():
                    bigrams[wi] = 0
                bigrams[wi] += 1
        #calculate probabilities and store into class values
                    
            
            
                
                




class TrigramFeature(FeatureExtractor):
    def __init__(self):
        self.trigrams = {}