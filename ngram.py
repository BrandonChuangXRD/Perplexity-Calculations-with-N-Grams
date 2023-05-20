##################################################
###  heavily influenced by asgn1 starter code  ###
##################################################
import numpy as np
import time

class FeatureExtractor(object):
    def __init__(self):
        self.grams = [] #just save these as a strings and w1|w2|w3 for bigrams/trigrams for easier debugging.
        self.grams_dict = {} #for O(1) lookup time i guess?
        self.token_prob = [] 
        self.token_count = []
        
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass

class UnigramFeature():
    def __init__(self):
        self.grams = []
        self.grams_dict = {}
        self.token_prob = [] #? Apparently we don't include <START> in these probability calculations
        self.token_counts = [] #useful in building bigrams

    def fit(self, train_file):
        start_time = time.time()
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
        for i in range(len(self.grams)):
            self.grams_dict[self.grams[i]] = i
            self.token_counts.append(tokens[self.grams[i]]) 

        #! calculate probability
        for i in range(len(self.grams)):
            p = self.token_counts[i]/(word_count)
            self.token_prob.append(p)

        #should have 26602 unique tokens
        print("UNIGRAM: number of tokens minus \"<START>\":", len(tokens)-1)
        train_file.seek(0)
        end_time = time.time()
        print("UNIGRAM FIT TIME:", end_time-start_time, "seconds")
        
    #just make a list in order with the token instead of the word
    def transform(self, text: list):
        feat = np.empty((0,), dtype=np.int64)
        for i in text.split(" "):
            i = i.rstrip("\n")
            if i in self.grams:
                where_i = self.grams_dict[i]
                #print(f"{i} found in unigram list")
                feat = np.append(feat,where_i)
            else:
                #print(f"\"{i}\" not found in unigram list")
                feat = np.append(feat, self.grams_dict["<UNK>"])
        feat = np.append(feat, self.grams_dict["<STOP>"])
        #print(feat)
        return feat

    def transform_list(self, text_set: list):
        fs = []
        print("UNIGRAM: TRANFORMING LIST")
        lcount = 0
        for i in text_set:
            if lcount != 0 and lcount % 5000 == 0:
                print(lcount)
            lcount +=1
            #print(i)
            fs.append(self.transform(i))
        return fs

class BigramFeature(FeatureExtractor):
    def __init__(self):
        self.grams = []
        self.grams_dict = {}
        self.token_prob = [] #should be changed to token count tbh but I'm too lazy
        self.token_counts = []
        self.unigrams = UnigramFeature() #useful shortcut
    
    #to split the list of unigrams into a list of bigrams
    def bi_splitter(self, X):
        bigram_split = []
        #do this for <START> at the beginning
        bigram_split.append((X[0], self.unigrams.grams_dict["<START>"]))
        for i in range(1, len(X)):
            #[token, given token] for given probability
            bigram_split.append((X[i], X[i-1]))
        return bigram_split
    
    def fit(self, train_file):
        start_time = time.time()
        #the values represent the number of observed outcomes under the key.
        #useful for probability calculation
        #TODO I'm thinking you can just make a unigram class in here and then use that
        #TODO to parse everything since unknowns are recorded as (<UNK>|token) instead
        #TODO of straight unknowns
        print("BIGRAM: fitting")
        self.unigrams.fit(train_file) #use this to replace OOV in train_file with <UNK>
        bigrams = {}
        start_tokens = 0
        pcount = 0
        for Xi in train_file:
            start_tokens+=1
            if pcount > 0 and pcount % 5000 == 0:
                print(pcount)
            pcount+=1
            Xi_tkn = self.unigrams.transform(Xi)
            #split sentence (keep in mind start is not in the unigram transformed language)
            Xi_bigramed = self.bi_splitter(Xi_tkn)
            #print(Xi_bigramed)
            for wi in Xi_bigramed:     
                if wi not in bigrams.keys():
                    bigrams[wi] = 0
                bigrams[wi] += 1
        #print(bigrams)
        #fix finalize variables
        self.grams = list(bigrams.keys())
        for i in range(len(self.grams)):
            self.grams_dict[self.grams[i]] = i
            self.token_counts.append(bigrams[self.grams[i]]) 
        #TODO calculate probability, THIS IS WRONG  
        for i in range(len(self.grams)):
            prior_token = self.grams[i][1]
            self.token_prob.append(self.token_counts[i]/self.unigrams.token_counts[prior_token])


        print("BIGRAM: number of tokens minus \"<START>\":", len(bigrams.keys())-1)
        train_file.seek(0)
        end_time = time.time()
        print("BIGRAM FIT TIME:", end_time-start_time, "seconds")

    def transform(self, Xi):
        unigram_tkns = self.unigrams.transform(Xi)
        bigram_tkns = self.bi_splitter(unigram_tkns)
        for i in range(len(bigram_tkns)):
            bigram_tkns[i] = self.grams_dict[bigram_tkns[i]]
        return bigram_tkns

    def transform_list(self, text_set):
        fs = []
        for i in text_set:
            print(i)
            fs.append(self.transform(i))
        return np.array(fs)

class TrigramFeature(FeatureExtractor):
    def __init__(self):
        self.grams = []
        self.grams_dict = {}
        self.token_prob = [] #should be changed to token count tbh but I'm too lazy
        self.token_counts = []
        self.bigrams = BigramFeature() #useful shortcut

    #takes an already bigrammed list of tokens
    # def tri_splitter(self, sentence):
        

    def fit(self, train_file):
        self.bigrams.fit(train_file)
        start_tokens = 0
        pcount = 0
        for Xi in train_file:
            start_tokens += 1
            if pcount > 0 and pcount % 5000 == 0:
                print(pcount)
            

#     def transform(self, Xi):

#     def transform_list(self, X):
