##################################################
###  heavily influenced by asgn1 starter code  ###
##################################################
import numpy as np
import time

# class FeatureExtractor(object):
#     def __init__(self):
#         self.grams_dict = {} #for O(1) lookup time i guess?
#         self.token_count = {}
#         self.add_alpha = 0 #make a function to change this
        
#     def set_alpha(self, val):
#         pass
#     def fit(self, text_set):
#         pass
#     def transform(self, text):
#         pass  
#     def transform_list(self, text_set):
#         pass

class UnigramFeature():
    def __init__(self):
        self.grams_index = {} #word: index
        self.grams_word = {} #index: word
        self.token_counts = {} #use in probability, uses index as the key
        self.unique_tokens = 0 #excludes <START>
        self.total_tokens = 0
        self.alpha = 0 #make a function to change this

    def set_alpha(self,val):
        self.alpha = val
        print("add alpha set to ", self.alpha)

    #word must be an index
    def get_prob(self, word):
        return (self.token_counts[word]+self.alpha)/(self.total_tokens+(self.unique_tokens*self.alpha))

    def fit(self, train_file):
        start_time = time.time()

        #make dictionary
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
        gramlist = list(tokens.keys())
        self.unique_tokens = len(gramlist)-1
        for i in range(len(gramlist)):
            self.grams_index[gramlist[i]] = i
            self.grams_word[i] = gramlist[i]
            self.token_counts[i] = tokens[gramlist[i]]

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
            if i in self.grams_index.keys():
                where_i = self.grams_index[i]
                #print(f"{i} found in unigram list")
                feat.append(where_i)
            else:
                #print(f"\"{i}\" not found in unigram list")
                feat.append(self.grams_index["<UNK>"])
        feat.append(self.grams_index["<STOP>"])
        #print(feat)
        return feat

    def transform_list(self, text_set: list):
        fs = []
        print("UNIGRAM: TRANFORMING LIST")
        #lcount = 0
        for i in text_set:
            #if lcount != 0 and lcount % 5000 == 0:
                #print(lcount)
            #lcount +=1
            #print(i)
            fs.append(self.transform(i))
        return fs



class BigramFeature():
    #no need for a count of unique tokens
    def __init__(self):
        self.token_counts = {}
        self.alpha = 0 #make a function to change this
        self.uni = UnigramFeature() #useful shortcut

    #to split the list of unigrams into a list of bigrams
    def bi_splitter(self, X):
        bigram_split = []
        #do this for <START> at the beginning
        bigram_split.append((X[0], self.uni.grams_index["<START>"]))
        for i in range(1, len(X)):
            #[token, given token] for given probability
            bigram_split.append((X[i], X[i-1]))
        return bigram_split
   
    #takes a token (which is a set of two numbers)
    def get_prob(self, word):
        return (self.token_counts[word]+self.alpha)/(self.uni.token_counts[word[1]]+(self.uni.unique_tokens*self.alpha))

    def set_alpha(self, val):
        self.alpha = val
        print("add-alpha set to:", self.alpha)

    #TODO FIX
    def fit(self, train_file):
        start_time = time.time()
        #the values represent the number of observed outcomes under the key.
        #useful for probability calculation
        print("BIGRAM: fitting")

        self.uni.fit(train_file) #use this to replace OOV in train_file with <UNK>
        
        #this is the time consuming part
        X = train_file.readlines()
        for Xi in X:
            #TODO try just doing this part manually see if it makes a difference
            Xi_tkn = [self.uni.grams_index["<START>"]] + self.uni.transform(Xi)
            for i in range(1, len(Xi_tkn)):
                curr = (Xi_tkn[i], Xi_tkn[i-1])
                if curr not in self.token_counts.keys():
                    self.token_counts[curr] = 0
                self.token_counts[curr] += 1

        print("BIGRAM: number of tokens minus \"<START>\":", len(self.token_counts.keys())-1)
        train_file.seek(0)
        end_time = time.time()
        print("BIGRAM FIT TIME:", end_time-start_time, "seconds")

    def transform(self, Xi):
        unigram_tkns = self.uni.transform(Xi)
        #this gives a list of lists for w|w-1
        bigram_tkns = self.bi_splitter(unigram_tkns)
        return bigram_tkns

    def transform_list(self, text_set):
        fs = []
        for i in text_set:
            #print(i)
            fs.append(self.transform(i))
        return fs

class TrigramFeature():
    def __init__(self):
        self.token_counts = []
        self.alpha = 0 #make a function to change this
        self.bi = BigramFeature() #useful shortcut

    #takes an already unigrammed list of tokens
    def tri_splitter(self, sentence):
        #print("bigrammed: ", sentence)
        Xi_trigram = [(sentence[1], sentence[0])]
        for i in range(2, len(sentence)):
            Xi_trigram.append((sentence[i], sentence[i-1], sentence[i-2]))
        return Xi_trigram

    def set_alpha(self, val):
        self.alpha = val

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
        for i in range(len(self.grams)):
            #this is the one case where theres a bigram
            if len(self.grams[i]) == 2:
                curr_gram = self.grams[i]
                #this might be wrong.
                self.token_prob.append((self.token_counts[i]+self.add_alpha)/ (self.bigrams.unigrams.token_counts[curr_gram[1]] +  self.add_alpha*(len(self.bigrams.unigrams.grams)-1)))
                continue
            prior_tokens = (self.grams[i][1], self.grams[i][2])
            #now with alpha smoothing!
            self.token_prob.append((self.token_counts[i]+self.add_alpha)/(self.bigrams.token_counts[self.bigrams.grams_dict[prior_tokens]]+((len(self.bigrams.unigrams.grams)-1)*self.add_alpha)))
        #print("self.grams afterwards:", self.grams)
        end_time = time.time()
        print("TRIGRAM FIT TIME:", end_time-start_time, "seconds")
    
    #also needs to add to bigram if its missing.
    def add_trigram(self, wi):
        if len(wi) == 2:
            self.bigrams.add_bigram(wi)
            w_index = len(self.grams)
            self.grams.append(wi)
            self.grams_dict[wi] = w_index
            self.token_counts.append(0)
            #!NOT CORRECT, BUT FIX IT LATER
            self.token_prob.append(self.bigrams.token_prob[self.bigrams.grams_dict[wi]])
            return
        prior_bi = (wi[1], wi[2])
        if prior_bi not in self.bigrams.grams:
            self.bigrams.add_bigram(prior_bi)
        #it is impossible for the trigram to exist if the bigram doesn't exist
        w_index = len(self.grams)
        self.grams.append(wi)
        self.token_counts.append(0)
        self.grams_dict[wi] = w_index
        prior_count = self.bigrams.token_counts[self.bigrams.grams_dict[prior_bi]]
        self.token_prob.append((self.add_alpha) / (prior_count+(self.add_alpha*(len(self.bigrams.unigrams.grams)-1))))

    #TODO must include fixes for additive smoothing
    def transform(self, Xi):
        #print(Xi)
        Xi_unigrammed = self.bigrams.unigrams.transform(Xi)
        #print(Xi_unigrammed)
        Xi_unigrammed = [self.bigrams.unigrams.grams_dict["<START>"]] + Xi_unigrammed
        Xi_trigrammed = self.tri_splitter(Xi_unigrammed)
        #print("after tri_splitter:", Xi_trigrammed)
        #print(Xi_trigrammed)
        for i in range(len(Xi_trigrammed)):
            if tuple(Xi_trigrammed[i]) not in self.grams_dict:
                self.add_trigram(tuple(Xi_trigrammed[i]))
                #print("thing not found")
            Xi_trigrammed[i] = self.grams_dict[Xi_trigrammed[i]]
        #print(Xi_trigrammed)
        return Xi_trigrammed


    def transform_list(self, text_set):
        fs = []
        pcount = 0
        for i in text_set:
            pcount+=1
            if pcount % 100 == 0:   
                print(i)
            fs.append(self.transform(i))
        return fs
