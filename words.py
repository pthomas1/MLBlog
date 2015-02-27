# The MIT License (MIT)
# Copyright (c) 2015 Thoughtly, Corp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.



# NLTK provides a lot of functionality, only a small fraction of which will be
# exercised in this demo.  We are mainly interested in accessing built in 
# test corpora.  For this demo we're just using plaintext corpora, though
# NLTK has many additional available.  If you want to provide a corpus of
# your own based on text within a folder, nltk helps us tokenize your
# files to match the text found in the NLTK corpora.
import nltk

# argparse is a standard Python mechanism for handling commandline 
# args while avoiding a bunch of boilerplate code.
import argparse

# This is a module that provides a bunch of simple methods that make
# accessing the filesystem simpler.
from utils import fs, charting

# Python logging allows us to log formatted log messages at different
# log levels.
import logging

# A simple helper for setting up log files based on commandline args
from utils import log

# Used for log method
import math

# numpy is just used for some simple array helpers
import numpy

def main():

    # Build the commandline parser and return entered args.  This also
    # setups up any non-ML/NLP config needed by the script (such as logging)
    args = configure_command_line_arguments()

    # Read text into an array of words based on what source the user entered.
    words_array, corpus_name = load_text_corpus(args)

    # Stem the input.  Stemming will take variations on a word (run, runs) and map
    # them to a single representation of the word (run).  It loses information,
    # but this allows subsequent analysis to be performed on the corpus under
    # the assumption that the information contained in the precise word chosen
    # is less valuable than that in the stem of that word.  Stemming is a
    # relatively naive algorithm which essentially cuts off the ends of words
    # to get them down to their base stem
    if args["stem"]:
        words_array = stem_words_array(words_array)

    # Lemmatization serves a similar purpose as stemming.  Instead of simply
    # cutting ends off of words, lemmatization attempts to map a word to its
    # lemma.  This does include chopping the end off of a word in some cases.
    # In others it is a more complex operation.  For example, lemmatization
    # will try to map am, is, are and were to their lemma, be.  This requires
    # a better concept of the language being lemmatized and is more resource
    # intensive than stemming.  Different information is lost in lemmatization,
    # so different usecases may prefer one over the other.
    elif args["lemma"]:
        words_array = lemmatize_words_array(words_array)

    # Here we want to run through all of the corpora and calculate the uniqque
    #  word counts, stemmed word counts and lemmatized word counts.
    elif args["stemVsLemma"]:
        compare_stemming_to_lemmatization()

    logging.info("The corpus contains " + str(len(words_array)) + " elements after processing");

    # Multiple methods make use of the unique vocabulary within the document.  We pass
    # it into the various methods that can use it.  If it's not provided the relevant
    # methods will calculate it.  Same applies to term frequencies
    unique_vocabulary = None
    term_frequencies = None

    # Calculating the vocabulary size gives you an idea of the overall complexity of a corpus.  It is a quick
    # and easy way to succinctly summarize a corpus down to a single number.  Generally speaking, this doesn't
    # provide a whole lot of information, but it can be a quick way to compare 2 different corpora. It can also
    # quickly illustrate the difference between an original corpus, its stemmed version and its lemmatized version.
    if args["vocabularySize"]:
        unique_vocabulary = calculate_corpus_vocabulary_size(words_array)

    # Term presence allows the user to see a list of all unique tokens in a document.  This allows the user to
    # quickly see what sorts of words appear in a corpus.  It it also useful for examining the effect of
    # tokenization or lemmatization on a corpus.  For some applications it is preferred to use the simple presence
    # of a token as compared to its frequency.
    if args["termPresence"]:
        output_corpus_terms(words_array, unique_vocabulary)

    # Term frequency is a common method of translating a corpus into a word vector.  This method executes a simple
    # count of all instances of each term.  A term that appears 10 times in a corpus will be counted exactly 10 times.
    # Many applications need this raw term frequency to generate simple models of a language.  Other applications
    # make use of the term frequency indirectly as part of a process of vectorizing text.
    if args["termFrequency"]:
        term_frequencies = collect_and_output_corpus_term_frequencies(words_array, corpus_name)

    # Log normalizing term frequencies effectively squashes the output counts of the term frequency process by taking
    # the log of the frequencies of each term.  If a term T appears 10 times more often than some other term X, the log
    # base 10 of T will be 1 greater than that of X.  This calculation essentially says that frequency is important,
    # but words that are more frequent shouldn't be considered to be linearly more (or less) important than those
    # which are less.
    if args["logNormalize"]:
        term_frequencies = collect_and_output_normalized_corpus_term_frequencies(words_array, corpus_name, term_frequencies)

    # Frequency frequency is a bit of an odd metric.  Here we want to know, for example, how many words are used just
    # one time?  How many are used 10?  We calculate all the frequencies of each word, just like when calculating
    # term frequencies above.  We then take those frequencies and count them.  So every time we see a term that is
    # used 1 time we increment the frequency=1 counter.  When we see a term used 10 times we increment the frequency=10
    # counter.  This metric gives us an easy way to get a feel for how repetitive a corpus is regarding word use.  If
    # we see that there are a relatively large number of frequency=1 items then we can say that the corpus does not
    # repeat itself, again, in terms of word use.  Seeing a large number for frequency = 100, for example, would
    # indicate the opposite.  One thing to consider here is stop words.  Naturally we see certain English words occur
    # extremely frequently (the, and, a, or...).  Removing those words from this analysis can yield a bit more
    # useful information than leaving them in.
    if args["frequencyFrequency"]:
        collect_and_output_frequency_frequencies(words_array, corpus_name, term_frequencies)



###############################################################################
#
# Calculating the vocabulary size requires only two simple steps:
#   1) Accumulate all unique words
#   2) Count the unique words accumulated in 1
#
################################################################################

def calculate_corpus_vocabulary_size(corpus):
    unique_vocabulary = collect_unique_terms(corpus)
    logging.debug("The corpus has a total vocabulary of " + str(len(unique_vocabulary))
                    + " unique tokens.")
    return unique_vocabulary



###############################################################################
#
# This method takes or determines the unique_vocabulary for the given corpus.
# This is then output to a CSV file where each row is a single term from the
# corpus.
#
###############################################################################

def output_corpus_terms(corpus, unique_vocabulary=None):
    if unique_vocabulary is None:
        unique_vocabulary = collect_unique_terms(corpus)

    output_csv_file = fs.open_csv_file("corpus_terms.csv", ["Term"])

    for term in unique_vocabulary:
        logging.debug(term)
        output_csv_file.writerow([term])



###############################################################################
#
# This method goes through a corpus of text and outputs the raw frequency
# counts of each unique term.
#
###############################################################################

def collect_and_output_corpus_term_frequencies(corpus, corpus_name):
    term_frequencies = collect_term_counts(corpus)

    output_csv_file = fs.open_csv_file("term_frequencies.csv", ["Term", "Frequency"])

    unsorted_array = [[key,value] for key, value in term_frequencies.iteritems()]
    sorted_array = sorted(unsorted_array, key=lambda term_frequency: term_frequency[1], reverse=True)

    for term, frequency in sorted_array:
        output_csv_file.writerow([term] + [frequency])

    # output a bar chart illustrating the above
    chart_term_frequencies("term_frequencies.png",
                           "Term Frequencies (" + corpus_name + ")",
                           "Term Frequencies",
                           sorted_array, [0, 1, 2, -3, -2, -1])

    return term_frequencies



###############################################################################
#
# This method takes in or generates term frequencies.  Each unique term in the
# corpus is counted.  This first step is identical to the process in
# collect_and_output_corpus_term_frequencies.  Once those words are collected,
# the method iterates through each term/count pair and log normalizes the
# count where normalized = 1 + log10(frequency).  This will result in a value
# of 1 if frequency is 1, 2 if frequency is 10, 3 if frequency is 100, etc.
#
###############################################################################

def collect_and_output_normalized_corpus_term_frequencies(corpus, corpus_name, term_frequencies=None):

    if term_frequencies is None:
        term_frequencies = collect_term_counts(corpus)

    output_csv_file = fs.open_csv_file("normalized_term_frequencies.csv", ["Term", "Log Normalized TF"])

    unsorted_array = []

    for term, frequency in term_frequencies.iteritems():
        normalized_term_frequency = (1 + math.log(frequency, 10))
        unsorted_array.append([term, normalized_term_frequency])
        output_csv_file.writerow([term] + [normalized_term_frequency])

    sorted_array = sorted(unsorted_array, key=lambda term_frequency: term_frequency[1], reverse=True)

    # output a bar chart illustrating the above
    chart_term_frequencies("normalized_term_frequencies.png",
                           "Log Normalized Term Frequencies (" + corpus_name + ")",
                           "Term Frequencies",
                           sorted_array, [0, 1, 2, -3, -2, -1])

    return term_frequencies



###############################################################################
#
# This method first collects the raw frequency counts of each unique term
# in a corpus.  It then iterates through these term/frequency pairs and
# accumulates the frequency of the given frequency from the pair.  This serves
# to calculate the number of terms that are used a given number of times.  For
# example, this method would identify the number of terms that appear once in
# a document.  It will also identify the number of terms that appear 10 times
# in a document, etc.
#
###############################################################################

def collect_and_output_frequency_frequencies(corpus, corpus_name, term_frequencies):
    if term_frequencies is None:
        term_frequencies = collect_term_counts(corpus)

    frequency_frequencies = {}
    for term, frequency in term_frequencies.iteritems():
        if frequency_frequencies.has_key(frequency):
            frequency_frequencies[frequency] += 1
        else:
            frequency_frequencies[frequency] = 1

    unsorted_array = [[key,value] for key, value in frequency_frequencies.iteritems()]
    sorted_array = sorted(unsorted_array, key=lambda frequency_frequency: frequency_frequency[1], reverse=True)

    frequency_frequencies_to_chart = []
    frequencies_to_chart = []
    output_csv_file = fs.open_csv_file("frequency_frequencies.csv", ["Frequency Frequency", "Term Frequency"])

    # we collect frequencies_to_chart and frequency_frequencies_to_chart each into their own single dimensional
    # array.  Then we pass frequency_frequencies_to_chart in an array so that it is 2D as needed by the chart.
    # This means there is exactly 1 data set and 6 columns of data in the set.  There is no second set to compare
    # it to.
    for index, (term_frequency, frequency_frequency) in enumerate(sorted_array):
        output_csv_file.writerow([frequency_frequency] + [term_frequency])
        if index <= 20:
            frequencies_to_chart.extend([term_frequency])
            frequency_frequencies_to_chart.extend([frequency_frequency])

    charting.bar_chart( "frequency_frequencies.png",
                        [frequency_frequencies_to_chart],
                        "Frequency Frequencies (" + corpus_name + ")",
                        frequencies_to_chart,
                        "Frequency Frequency",
                        None,
                        ['#59799e', '#810CE8', '#FF0000', '#12995D', '#FD53FF', '#AA55CC'],
                        0.2, 0.0)


    return frequency_frequencies



###############################################################################
#
# We use a Python dictionary to accomplish this.  Because we are simply
# checking for the existence of a word, it is sufficient to simply add it to
# a dictionary with a value of 1 (really any value would suffice).  Because a
# dictionary will not allow duplicate keys, we can be sure that the accumulated
# words in the dictionary are unique.
#
################################################################################

def collect_unique_terms(corpus):
    unique_vocabulary = {}
    for term in corpus:
        unique_vocabulary[term] = 1;

    return unique_vocabulary;



###############################################################################
#
# This method iterates through the entire corpus and collects counts of all
# unique words.
#
# Similar to collect_unique_words, this method uses a dictionary to track the
# uniqueness of a word.  Unlike collect_unique_words, this method does not
# simply note the existence of a new word.  Instead, each instance of a term
# increments a counter tied to the value of the term.
#
################################################################################

def collect_term_counts(corpus):
    unique_word_counts = {}
    for term in corpus:
        if unique_word_counts.has_key(term):
            unique_word_counts[term] += 1;
        else:
            unique_word_counts[term] = 1

    return unique_word_counts



###############################################################################
#
# Most of this method simply returns the relevant corpus based on the requested
# corpus name passed in via the commandline.  The last option, custom, is used
# when the user chooses to look at a corpus comprised of one or more of their
# own documents.
#
###############################################################################

def load_text_corpus(args):

    if args.has_key("abc") and args["abc"]:
        logging.debug("Loading the ABC corpus.")
        name = "ABC"
        words = nltk.corpus.abc.words()

    elif args.has_key("genesis") and args["genesis"]:
        logging.debug("Loading the Genesis corpus.")
        name = "Genesis"
        words = nltk.corpus.genesis.words()

    elif args.has_key("gutenberg") and args["gutenberg"]:
        logging.debug("Loading the Gutenberg corpus.")
        name = "Gutenberg"
        words = nltk.corpus.gutenberg.words()

    elif args.has_key("inaugural") and args["inaugural"]:
        logging.debug("Loading the Inaugural Address corpus.")
        name = "Inaugural"
        words = nltk.corpus.inaugural.words()

    elif args.has_key("stateUnion") and args["stateUnion"]:
        logging.debug("Loading the State of the Union corpus.")
        name = "Union"
        words = nltk.corpus.state_union.words()

    elif args.has_key("webtext") and args["webtext"]:
        logging.debug("Loading the webtext corpus.")
        name = "Web"
        words = nltk.corpus.webtext.words()

    elif args.has_key("custom") and args["custom"] != None:
        logging.debug("Loading a custom corpus from " + args["custom"])
        name = "Custom"
        words = load_custom_corpus(args["custom"])
    else:
        words = ""
        name = "None"

    if len(words) > 0:
        logging.debug("Read " + str(len(words)) + " words: " + str(words[0:20]))

    return words, name



###############################################################################
#
# Combine all the docs into a single string and then tokenize.
#
###############################################################################    

def load_custom_corpus(path):
    all_custom_files = fs.directory_file_names(path, True, None)
    combined_corpus = ""
    for file_name in all_custom_files:
        combined_corpus = open(file_name).read() + "\n";

    return nltk.word_tokenize(combined_corpus)



###############################################################################
#
# This method simply iterates through a list of words and returns their
# stemmed version.  Stemming serves to map multiple words with the same root
# down to a single word stem.  This allows for a reduction in features.  Also,
# in some applications, it is more useful to consider words by their stems
# rather than consider the actual word.
#
################################################################################

def stem_words_array(words_array):
    stemmer = nltk.PorterStemmer();
    stemmed_words_array = [];
    for word in words_array:
        try:
            stem = stemmer.stem(word);
            stemmed_words_array.append(stem);
        except Exception:
            pass

    return stemmed_words_array;



################################################################################
#
# Lemmatization is implemented similarly to stemming.  We iterate over each
# word in the input array and lemmatize it using the NLTK WordNetLemmatizer.
#
################################################################################

def lemmatize_words_array(words_array):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_words_array = [];
    for word in words_array:
        lemma = lemmatizer.lemmatize(word)
        lemmatized_words_array.append(lemma)

    return lemmatized_words_array;



#################################################################################
#
# This method looks at all of the corpora supported by the app (from NLTK) and
# generates a chart at stemmingVsLemmatization.png that shows each the number of
# words, number of words after stemming and number of words after lemmatization
# for each corpus.
#
#################################################################################
def compare_stemming_to_lemmatization():

    # load each of the corpora
    abc_words = nltk.corpus.abc.words()
    genesis_words = nltk.corpus.genesis.words()
    gutenberg_words = nltk.corpus.gutenberg.words()
    inaugural_words = nltk.corpus.inaugural.words()
    state_union_words = nltk.corpus.state_union.words()
    webtext_words = nltk.corpus.webtext.words()

    all_words = [abc_words, genesis_words, gutenberg_words, inaugural_words,
                 state_union_words, webtext_words]
    corpora_names =["ABC", "Genesis", "Gutenberg", "Inaugural",
                    "Union", "Web"];

    word_counts = []
    lemmatized_counts = []
    stemmed_counts = []

    # iterate through each corpus and generate counts of the unique tokens
    # in each
    for index, words in enumerate(all_words):
        logging.debug("Lemmatizing " + corpora_names[index])
        lemmatized = collect_term_counts(lemmatize_words_array(words))
        logging.debug("Stemming " + corpora_names[index])
        stemmed = collect_term_counts(stem_words_array(words))
        word_counts.extend([len(collect_term_counts(words))])
        lemmatized_counts.extend([len(lemmatized)])
        stemmed_counts.extend([len(stemmed)])

    logging.info("Corpora: " + str(corpora_names))
    logging.info("Word Counts: " + str(word_counts))
    logging.info("Lemmatized Word Counts: " + str(lemmatized_counts))
    logging.info("Stemmed Word Counts: " + str(stemmed_counts))

    # output a bar chart illustrating the above
    charting.bar_chart( "stemming_vs_lemmatization.png",
                        [word_counts, lemmatized_counts, stemmed_counts],
                        "Token Counts for Words, Stems and Lemmas",
                        corpora_names,
                        "Token Counts",
                        ["Words", "Lemmas", "Stems"],
                        ['#59799e', '#810CE8', '#FF0000'],
                        .5)



###############################################################################
#
# Build the commandline parser for the script and return a map of the entered
# options.  In addition, setup logging based on the user's entered log level.
# Specific options are documented inline.
#
###############################################################################

def configure_command_line_arguments():
    # Initialize the commandline argument parser.
    parser = argparse.ArgumentParser(description='Play with words using NLTK.')

    # Configure the log level parser.  Verbose shows some logs, veryVerbose
    # shows more
    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument("-v",
                               "--verbose",
                               help="Set the log level verbose.",
                               action='store_true',
                               required=False)
    logging_group.add_argument("-vv",
                               "--veryVerbose",
                               help="Set the log level verbose.",
                               action='store_true',
                               required=False)

    # In this app we allow the user to choose from a handful of built-in
    # corpora and a user provided one
    corpora_group = parser.add_mutually_exclusive_group(required=True)

    # NLTK supports six built in plaintext corpora.  This allows the user
    # to choose between those six corpora or a seventh option - the
    # corpus the user provided.
    # The first option is a corpus taken from ABC news.
    corpora_group.add_argument('-abc',
                               '--abc',
                               help="ABC news corpus",
                               required=False,
                               action='store_true')

    # The second option is the book of Genesis
    corpora_group.add_argument('-gen',
                               '--genesis', help="The book of Genesis from the Bible.",
                               required=False,
                               action='store_true')

    # Third option is a collection of text from project Gutenberg
    corpora_group.add_argument('-gut',
                               '--gutenberg', help="Text from Project Gutenberg.",
                               required=False,
                               action='store_true')

    # Fourth is text from presidential inaugural addresses
    corpora_group.add_argument('-in',
                               '--inaugural', help="Text from inaugural addresses.",
                               required=False,
                               action='store_true')

    # Fifth is text from the State of the Union
    corpora_group.add_argument('-su',
                               '--stateUnion', help="Text from State of the Union Addresses.",
                               required=False,
                               action='store_true')

    # The final NLTK provided corpus is text from the web
    corpora_group.add_argument('-web',
                               '--webtext', help="Text taken from the web.",
                               required=False,
                               action='store_true')

    corpora_group.add_argument('-svl',
                        '--stemVsLemma',
                        help="Generate chart of corpus length of original, stemmed and lemmatized word",
                        required=False,
                        action='store_true')

    # Tell the parser that there is an optional corpus that can be pulled in.
    # The directory can contain multiple files and directories (if the user
    # also passes --recursive)
    fs.add_filesystem_path_args(parser,
                                '-c',
                                '--custom',
                                help='Directory of files to include in a custom corpus.',
                                required=False,
                                group=corpora_group)

    # Optionally, the user is able to stem or lemmatize the input.
    preprocessing_group = parser.add_mutually_exclusive_group(required=False)

    # Select stemming
    preprocessing_group.add_argument('-s',
                                     '--stem',
                                     help="Stem the input.",
                                     required=False,
                                     action='store_true')

    # Select lemmatization
    preprocessing_group.add_argument('-l',
                                     '--lemma',
                                     help="Lemmatize the input.",
                                     required=False,
                                     action='store_true')

    # What do you want to know?  These params allow one or more calculations to be run on
    # the input data.  In addition, you can ask the app to stem the data before running any
    # of these calculations

    # Calculate the vocabulary size of the selected corpus
    parser.add_argument('-vs',
                        '--vocabularySize',
                        help="Calculate the vocabulary size.",
                        required=False,
                        action='store_true')

    # List all terms found in the corpus
    parser.add_argument('-tp',
                        '--termPresence',
                        help="List all words that are present.",
                        required=False,
                        action='store_true')

    # List the frequency of terms in the corpus
    parser.add_argument('-tf',
                        '--termFrequency',
                        help="Calculate the frequency of each word.",
                        required=False,
                        action='store_true')

    # Log normalize the term frequencies
    parser.add_argument('-ln',
                        '--logNormalize',
                        help="Calculate the log of the frequency.",
                        required=False,
                        action='store_true')

    # Determine the frequency of each frequency of terms
    parser.add_argument('-ff',
                        '--frequencyFrequency',
                        help="Calculate the frequency of each frequency.  For example, 7 words appear once, 5 appear twice, etc.",
                        required=False,
                        action='store_true')

    # Parse the passed commandline args and turn them into a dictionary.
    args = vars(parser.parse_args())

    # Configure the log level based on passed in args to be one of DEBUG, INFO, WARN, ERROR, CRITICAL
    log.set_log_level_from_args(args)

    return args



###############################################################################
#
# A method to help term frequency / log normalized term frequency plot their
# output.
#
###############################################################################

def chart_term_frequencies(file_name, title, y_axis, term_frequencies, indexes=numpy.arange(5)):
    chart_terms=[]
    chart_frequencies=[]
    selected_frequencies = []
    for index in indexes:
        selected_frequencies.append(term_frequencies[index])

    for term, frequency in selected_frequencies:
        chart_terms.extend([term])
        chart_frequencies.append([frequency])

    charting.bar_chart( file_name,
                        chart_frequencies,
                        title,
                        None,
                        y_axis,
                        chart_terms,
                        ['#59799e', '#810CE8', '#FF0000', '#12995D', '#FD53FF', '#AA55CC'],
                        1, 0.2)



###############################################################################    
#
# This is a pythonism.  Rather than putting code directly at the "root"
# level of the file we instead provide a main method that is called 
# whenever this python script is run directly.
#
###############################################################################

if __name__ == "__main__":
    main()
    

