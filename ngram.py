##################################################
###  heavily influenced by asgn1 starter code  ###
##################################################
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        self.grams = [] #just save these as a strings and w1|w2|w3 for bigrams/trigrams for easier debugging.
        self.token_prob = [] #should be changed to token count tbh but I'm too lazy
        
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
        self.token_counts = [] #useful in building bigrams
    #this passes a file instead of the text.

    def fit(self, train_file):
        word_count = 0
        #make dictionary
        tokens = {}
        tokens["<START>"] = 0
        tokens["<STOP>"] = 0
        for l in train_file:
            tokens["<START>"] += 1
            tokens["<STOP>"] += 1
            word_count += 1
            for t in l.split(" "):
                t = t.rstrip("\n")
                word_count += 1
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
        for i in self.grams:
            self.token_counts.append(tokens[i]) 

        #! calculate probability
        for i in range(len(self.grams)):
            p = self.token_counts[i]/(word_count)
            self.token_prob.append(p)

        #should have 26602 unique tokens
        print("number of tokens minus \"<START>\":", len(tokens)-1)
        train_file.seek(0)
        
    #just make a list in order with the token instead of the word
    def transform(self, text: list):
        feat = np.empty((0,), dtype=np.int64)
        for i in text.split(" "):
            i = i.rstrip("\n")
            if i in self.grams:
                where_i = np.where(self.grams == i)[0][0]
                print("wherei", where_i)
                print(f"{i} found in unigram list")
                feat = np.append(feat,where_i)
            else:
                print(f"\"{i}\" not found in unigram list")
                feat = np.append(feat, np.where(self.grams == "<UNK>")[0][0])
        feat = np.append(feat, np.where(self.grams == "<STOP>")[0][0])
        print(feat)
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
        self.token_counts = []
        self.start_tokens = 0 #basically irrelevant.
        self.word_count = 0
        self.unigrams = UnigramFeature() #!This may provide a useful shortcut.
    #to split the sentence into a list of bigrams
    def sentence_splitter(self, sentence):
        tokens = sentence
        bigram_split = []
        for i in range(1, len(tokens)):
                #! the delimiter to find w_i and w_{i-1} is "<|>"
                bigram_split.append([np.where(self.unigrams.grams == tokens[i])[0][0], np.where(self.unigrams.grams == tokens[i-1])[0][0]])
        return bigram_split
    
    def fit(self, train_file):
        #the values represent the number of observed outcomes under the key.
        #useful for probability calculation
        #TODO I'm thinking you can just make a unigram class in here and then use that
        #TODO to parse everything since unknowns are recorded as (<UNK>|token) instead
        #TODO of straight unknowns
        self.unigrams.fit(train_file) #use this to replace OOV in train_file with <UNK>
        print("fitting")
        bigrams = {}
        start_tokens = 0
        train_file.seek(0)
        pcount = 0


        for Xi in train_file:
            #print(Xi)
            if pcount > 0 and pcount % 5000 == 0:
                print(pcount)
            pcount+=1
            Xi = Xi.rstrip("\n")
            Xi = Xi.split(" ")
            #replace OOV with <UNK>
            for wi in range(len(Xi)):
                if Xi[wi] not in self.unigrams.grams:
                    Xi[wi] = "<UNK>"
            #split sentence
            Xi_tokenized = ["<START>"] + Xi + ["<STOP>"]
            Xi_tokenized = self.sentence_splitter(Xi_tokenized)
            #print(Xi_tokenized)
            #print(Xi_tokenized)
            for wi in Xi_tokenized:     
                #print(wi)
                wi = tuple(wi)
                #print(type(bigrams))
                if wi not in bigrams:
                    bigrams[wi] = 0
                bigrams[wi] += 1
        #print(bigrams)
        #finalize variables
        self.grams = np.array(list(bigrams.keys))
        #calculate probabilities and store into class values
        self.unknown_index = np.where(self.grams == "<UNK>")[0][0]
        self.start_index = np.where(self.grams == "<START>")[0][0]
        self.stop_index = np.where(self.grams == "<STOP>")[0][0]
        for i in self.grams:
            self.token_counts.append(bigrams[i]) 
        #TODO calculate probability    
        




class TrigramFeature(FeatureExtractor):
    def __init__(self):
        self.trigrams = {}