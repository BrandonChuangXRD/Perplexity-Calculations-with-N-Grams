CSE143 Assignment 2: Lev Teytelman, Brandon Chuang

  

### HOW TO RUN:

run ```python3 main.py``` in the console.


### OPTIONS:
```
-h, --help show this help message and exit

--ngram NGRAM, -g NGRAM
    choose between unigram(1), bigram(2), and trigram(3)
  
--train TRAIN, -t TRAIN
    choose training token file

--predict PREDICT, -p PREDICT
    choose token file to predict on

--full find perplexities of sanity, train, and dev tokens (excludes test)

--test find perplexities of test tokens

--linear toggles linear interpolation (superseeds add-1 smoothing)

--lambdas LAMBDAS LAMBDAS LAMBDAS
    for linear interpolation: three floats separated by spaces (in order)

--smooth, -s enable smoothing for the MLE

--sanity test case for "HDTV ."

--alpha ALPHA, -a ALPHA
    Choose value for add smoothing

```
### DEFAULT VALUES:

--train: data/1b_benchmark.train.tokens  

--predict: data/1b_benchmark.dev.tokens

--alpha: 0

### NOTES:

the linear flag will mostly ignore any of the other flags (most imprortantly, the alpha value)  

the lambda flag will do nothing if not run with linear
