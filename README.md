# Perplexity using Unigram, Bigram, and Trigram Feature Extraction
This is code I wrote for an assignment in one of my classes at UC Santa Cruz. It is my implementation of unigrams, bigrams, and trigrams, with a perplexity. I'm unsure if this is how any of this is properly implemented, but it seems decently fast. A design document may be included in the future for this, but very quickly: there is a unigram class inside of the bigram class, and a bigram included in the trigram class, since the (n-1)-gram is required to calculate the prior for any n-gram greater than 1.

There was also a writeup included in the assignment, though I'm going to refrain from including it as I do not want to give away answers that are somewhat arbitrary to the given code.
## USAGE:

run ```python3 main.py``` in the console, followed by any combination of these flags with a few restrictions:

| flag | input | description |
| ------ | ----- | -----------|
| -h, --help | (None) | show this help message and exit |
| --ngram | 1, 2, or 3 | choose between unigram(1), bigram(2), and trigram(3) feature extractors |
| -t, --train | file name | choose a training file |
| -p, -predict | file name | choose a predicting file |
| --full | (None) | find perplexities of sanity, train, and dev tokens (excludes test) |
| --test | (None) | find perplexity of test file |
| --linear | (None) | toggles linear interpolation (superseeds add-1 smoothing) |
| --lambdas | [0-1] [0-1] [0-1] | for linear interpolation: three floats seperated by spaces (in order) to determine the weight of unigram, bigram, and trigram perplexities
| -s, --smooth | (None) | enable smoothing for the MLE in the perplexity calculation |
| --sanity | (None) | sanity test case: "HDTV ." |
| -a, --alpha | Positive Integer | Choose a value for additive smoothing

## DEFAULT VALUES:
| flag | value |
| --- | --- |
| --train | data/1b_benchmark.train.tokens |
| --predict | data/1b_benchmark.dev.tokens |
| --alpha | 0 |

## NOTES:

- the linear flag will mostly ignore any of the other flags (most imprortantly, the alpha value)  

- the lambda flag will do nothing if not run with linear

## CREDITS:
- The included tokens was produced by the WMT 2011 News Crawl data. The text in data_smol includes lyrics from "What is Love" from Haddaway.