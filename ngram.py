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
        self.grams = list(tokens.keys())
        #assign class variables for reference during perplexity and prediction
        for i in range(len(self.grams)):
            self.grams_dict[self.grams[i]] = i
            self.token_counts.append(tokens[self.grams[i]]) 

        #calculate probability
        for i in range(len(self.grams)):
            p = self.token_counts[i]/(word_count)
            self.token_prob.append(p)

        #should have 26602 unique tokens
        print("UNIGRAM: number of tokens minus \"<START>\":", len(tokens)-1)
        train_file.seek(0)
        end_time = time.time()
        print("UNIGRAM FIT TIME:", end_time-start_time, "seconds")
        
    #just make a list in order with the token instead of the word
    def transform(self, text):
        feat = []
        text = text.rstrip("\n")
        for i in text.split(" "):
            if i in self.grams:
                where_i = self.grams_dict[i]
                #print(f"{i} found in unigram list")
                feat.append(where_i)
            else:
                #print(f"\"{i}\" not found in unigram list")
                feat.append(self.grams_dict["<UNK>"])
        feat.append(self.grams_dict["<STOP>"])
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
        print("BIGRAM: fitting")

        self.unigrams.fit(train_file) #use this to replace OOV in train_file with <UNK>
        
        bigrams = {}
        start_tokens = 0
        pcount = 0
        #this is the time consuming part
        X = train_file.readlines()
        for Xi in X:
            start_tokens+=1
            if pcount > 0 and pcount % 5000 == 0:
                print(pcount, time.time()-start_time, "seconds")
            pcount+=1
            #TODO try just doing this part manually see if it makes a difference
            Xi_tkn = [self.unigrams.grams_dict["<START>"]] + self.unigrams.transform(Xi)
            #split sentence (keep in mind start is not in the unigram transformed language)
            #Xi_bigramed = self.bi_splitter(Xi_tkn)
            #TODO replace bi_splitter and do the parsing here
            #print(Xi_bigramed)
            # for wi in Xi_bigramed:     
            #     if wi not in bigrams.keys():
            #         bigrams[wi] = 0
            #     bigrams[wi] += 1
            for i in range(1, len(Xi_tkn)):
                curr = (Xi_tkn[i], Xi_tkn[i-1])
                if curr not in bigrams.keys():
                    bigrams[curr] = 0
                bigrams[curr] += 1

        #finalize variables
        self.grams = list(bigrams.keys())
        for i in range(len(self.grams)):
            self.grams_dict[self.grams[i]] = i
            self.token_counts.append(bigrams[self.grams[i]]) 
        for i in range(len(self.grams)):
            prior_token = self.grams[i][1]
            self.token_prob.append(self.token_counts[i]/self.unigrams.token_counts[prior_token])


        print("BIGRAM: number of tokens minus \"<START>\":", len(bigrams.keys())-1)
        train_file.seek(0)
        end_time = time.time()
        print("BIGRAM FIT TIME:", end_time-start_time, "seconds")

    def transform(self, Xi):
        unigram_tkns = self.unigrams.transform(Xi)
        #this gives a list of lists for w|w-1
        bigram_tkns = self.bi_splitter(unigram_tkns)
        #this converts it to the token
        for i in range(len(bigram_tkns)):
            bigram_tkns[i] = self.grams_dict[bigram_tkns[i]]
        return bigram_tkns

    def transform_list(self, text_set):
        fs = []
        for i in text_set:
            print(i)
            fs.append(self.transform(i))
        return fs

class TrigramFeature(FeatureExtractor):
    def __init__(self):
        self.grams = []
        self.grams_dict = {}
        self.token_prob = [] #should be changed to token count tbh but I'm too lazy
        self.token_counts = []
        self.bigrams = BigramFeature() #useful shortcut

    #takes an already unigrammed list of tokens
    #the first token is a bigram for some reason
    #TODO this is broken.
    def tri_splitter(self, sentence):
        #print("bigrammed: ", sentence)
        Xi_trigram = [(sentence[1], sentence[0])]
        for i in range(2, len(sentence)):
            Xi_trigram.append((sentence[i], sentence[i-1], sentence[i-2]))
        return Xi_trigram

    def fit(self, train_file):
        start_time = time.time()
        print("TRIGRAM: fitting")
        self.bigrams.fit(train_file)

        start_tokens = 0
        pcount = 0
        tris = {}
        X = train_file.readlines()
        for Xi in X:
            start_tokens += 1

            if pcount > 0 and pcount % 5000 == 0:
                print(pcount, time.time()-start_time, "seconds")
            pcount += 1
            
            Xi_tkn = [self.bigrams.unigrams.grams_dict["<START>"]] + self.bigrams.unigrams.transform(Xi)
            #hdtvprint = ("HDTV ." in Xi)
            #if hdtvprint:
                #print("HDTV PRINT XI_TKN", Xi_tkn)
            #TODO do tri splitter without the function or bi_splitter
            #print("Xi_trigrammed: ", Xi_trigramed)
            #the first one is a bigram
            if (Xi_tkn[1], Xi_tkn[0]) not in tris.keys():
                tris[(Xi_tkn[1], Xi_tkn[0])] = 0
            tris[(Xi_tkn[1], Xi_tkn[0])] += 1

            for i in range(2, len(Xi_tkn)):
                wi = (Xi_tkn[i], Xi_tkn[i-1], Xi_tkn[i-2])
                if wi not in tris.keys():
                    tris[wi] = 0
                tris[wi] += 1
        #finalize variables
        self.grams = list(tris.keys())
        for i in range(len(self.grams)):
            self.grams_dict[self.grams[i]] = i
            self.token_counts.append(tris[self.grams[i]])
        #TODO fix this probability 
        for i in range(len(self.grams)):
            #this is the one case where theres a bigram
            if len(self.grams[i]) == 2:
                self.token_prob.append(self.bigrams.token_prob[self.bigrams.grams_dict[self.grams[i]]])
                continue
            prior_tokens = (self.grams[i][1], self.grams[i][2])
            self.token_prob.append(self.token_counts[i]/self.bigrams.token_counts[self.bigrams.grams_dict[prior_tokens]])
        #print("self.grams afterwards:", self.grams)
        end_time = time.time()
        print("TRIGRAM FIT TIME:", end_time-start_time, "seconds")
            

    def transform(self, Xi):
        print(Xi)
        Xi_unigrammed = self.bigrams.unigrams.transform(Xi)
        print(Xi_unigrammed)
        Xi_unigrammed = [self.bigrams.unigrams.grams_dict["<START>"]] + Xi_unigrammed
        Xi_trigrammed = self.tri_splitter(Xi_unigrammed)
        #print("after tri_splitter:", Xi_trigrammed)
        print(Xi_trigrammed)
        for i in range(len(Xi_trigrammed)):
            if tuple(Xi_trigrammed[i]) not in self.grams_dict:
                Xi_trigrammed[i] = -1
                continue
            Xi_trigrammed[i] = self.grams_dict[Xi_trigrammed[i]]
        #print(Xi_trigrammed)
        return Xi_trigrammed


    def transform_list(self, text_set):
        fs = []
        for i in text_set:
            #print(i)
            fs.append(self.transform(i))
        return fs
