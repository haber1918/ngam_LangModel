# imports go here
import sys
import numpy as np
from collections import Counter
import random

"""
Akhil Sasi Kumar
This is my model for N-gram Text Generation.
"""


# My simple tokenizer
def my_tokenizer(sentence: str):
    """
    Parameters:
     sentence (str): An input string to be converted to a list of words
    Returns:
      list: the list of words
    """
    return sentence.split()


def get_ngrams(n: int, tokens: list) -> list:
    """
        Parameters:
         n (int): The value of n in n-gram
         tokens (list): The list of tokens that'll be converted to n-grams
        Returns:
          list: the list of n-grams
        """
    result = []
    # Special case for 1-gram
    if n == 1:
        return tokens

    # Every generated n-gram is of the form tuple(tuple(n-1 words), nth word)
    # Each n-gram is appended as a member of the return list
    for i in range(len(tokens) - n + 1):
        result.append((tuple(tokens[i:i + n - 1]), tokens[i + n - 1]))
    return result


class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing=False):
        """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether to use Laplace smoothing -> False by default
    """
        self.n_length = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing

        self.n_gram_context = {}
        self.n_gram_counter = {}

        self.vocab_list = {}
        self.vocab = 0

        self.start_tok = "<s>"
        self.end_token = "</s>"
        self.unknown_tok = "<UNK>"

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    None
        """

        with open(training_file_path, 'r', encoding="utf8") as train_file:
            # Replacing 1-freq words with the '<UNK>' token
            train_file_contents = train_file.read()
            word_counts = Counter(train_file_contents.split())
            for key, value in word_counts.items():
                if value == 1:
                    train_file_contents = train_file_contents.replace(" " + key + " ", " " + self.unknown_tok + " ")

            sentences_list = train_file_contents.split("\n")

            # A dictionary to maintain vocabulary and its length
            self.vocab_list = Counter(train_file_contents.split())
            self.vocab = len(self.vocab_list)

        # Training, where the n_gram_context and n_gram_counter is populated
        # n_gram_context is a dictionary -> (keys are n-1 words) (values are the nth word)
        # n_gram_counter is a dictionary -> (keys are the n-grams) (the frequency of this n-gram)
        for sentence in sentences_list:
            temp_ngrams = get_ngrams(self.n_length, my_tokenizer(sentence))

            # Special case for 1-gram
            if self.n_length == 1:
                for temp_ngram in temp_ngrams:
                    if temp_ngram in self.n_gram_counter:
                        self.n_gram_counter[temp_ngram] += 1
                    else:
                        self.n_gram_counter[temp_ngram] = 1
                continue

            # Common for other n-grams
            for temp_ngram in temp_ngrams:
                if temp_ngram[0] in self.n_gram_context:
                    self.n_gram_context[temp_ngram[0]].append(temp_ngram[1])
                else:
                    self.n_gram_context[temp_ngram[0]] = [temp_ngram[1]]

            for temp_ngram in temp_ngrams:
                if temp_ngram in self.n_gram_counter:
                    self.n_gram_counter[temp_ngram] += 1
                else:
                    self.n_gram_counter[temp_ngram] = 1

    def probability_check(self, temp_ngram):
        """Calculates the probability for a given n-gram.
    Parameters:
      temp_ngram (list): It is of the form list[tuple(tuple(n-1 words), (nth word))]
                        For 1-gram, the form is [tuple(word)]
    Returns:
      float: the probability value of the n-gram
    """
        if self.n_length == 1:
            return self.n_gram_counter[temp_ngram] / sum(self.n_gram_counter.values())

        count_of_token = self.n_gram_counter[temp_ngram]
        count_of_context = float(len(self.n_gram_context[temp_ngram[0]]))
        result = count_of_token / count_of_context

        return result

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of

    Returns:
      float: the probability value of the given string for this model
    """

        result = []
        sentence = ' ' + sentence + ' '
        sentence_words = sentence.split()
        for w in sentence_words:
            if w not in self.vocab_list:
                sentence = sentence.replace(" " + w + " ", " <UNK> ")
        temp_ngrams = get_ngrams(self.n_length, my_tokenizer(sentence))
        for temp_ngram in temp_ngrams:

            if self.n_length == 1:
                try:
                    count_of_token = self.n_gram_counter[(temp_ngram)]
                except:
                    count_of_token = self.n_gram_counter["<UNK>"]

                if self.is_laplace_smoothing:
                    result.append((count_of_token + 1) / (sum(self.n_gram_counter.values())
                                                          + self.vocab))
                else:
                    result.append(count_of_token / sum(self.n_gram_counter.values()))
                continue

            try:
                count_of_token = self.n_gram_counter[(temp_ngram[0], temp_ngram[1])]
            except:
                count_of_token = 0
            count_of_context = float(len(self.n_gram_context[temp_ngram[0]]))
            if self.is_laplace_smoothing:
                result.append((count_of_token + 1) / (count_of_context + self.vocab))
            else:
                result.append(count_of_token / count_of_context)

        return np.prod(result)

    def generate_token(self, n_minus1_words):
        """Generates a token(the nth word) for the given n-1 words
            It follows a pseudo-random criteria, where the probabilities are summed and compared against a
            random number
         Parameters:
           n_minus1_words (list): List of n-1 tokens

         Returns:
           string: the generated token
    """
        probability_dictionary = {}
        sum_of_probs = 0
        random_thresh = random.uniform(0, 1)
        if self.n_length == 1:
            keys = list(self.n_gram_counter.keys())
            random.shuffle(keys)
            for token in keys:
                if not self.is_laplace_smoothing:
                    sum_of_probs += self.n_gram_counter[token] / len(self.n_gram_counter)
                else:
                    sum_of_probs += (self.n_gram_counter[token] + 1) / (sum(self.n_gram_counter.values()) + self.vocab)
                if sum_of_probs > random_thresh:
                    return token

        else:
            token_of_interest = self.n_gram_context[n_minus1_words]

            for token in token_of_interest:
                probability_dictionary[token] = self.probability_check((n_minus1_words, token))

            # pseudo-random criteria where the probability values are added and compared with a random no.
            sum_of_probs = 0
            random_thresh = random.uniform(0, 1)
            token_ascending_list = dict(sorted(probability_dictionary.items(), key=lambda item: item[1]))
            for token in token_ascending_list:
                sum_of_probs += probability_dictionary[token]
                if sum_of_probs > random_thresh:
                    return token
        return token

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

    Returns:
      str: the generated sentence
    """
        n_length = self.n_length
        current_sentence = [(n_length - 1) * [self.start_tok] if n_length > 1 else ['<s>']][0]

        new_sentence = ''
        while True:
            nth_word = self.generate_token(tuple(current_sentence))
            current_sentence = current_sentence[1:] + [nth_word]  # Current sent is changed to accept the new word
            new_sentence = new_sentence + ' ' + nth_word
            if nth_word == self.end_token:
                break

        return '<s>' + new_sentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate

    Returns:
      list: a list containing strings, one per generated sentence
    """
        result = []
        for _ in range(n):
            result.append(self.generate_sentence())
        return result

    def perplexity(self, test_sequence):
        """
            Measures the perplexity for the given test sequence with this trained model.
            As described in the text, you may assume that this sequence may consist of many sentences "glued together".

        Parameters:
          test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
          float: the perplexity of the given sequence
        """
        return self.score(test_sequence) ** (1 / len(test_sequence))


def main():
    training_path = sys.argv[1]
    training_path1 = sys.argv[2]
    training_path2 = sys.argv[3]

    my_lang_model_bi = LanguageModel(2, is_laplace_smoothing=True)
    my_lang_model_uni = LanguageModel(1, is_laplace_smoothing=True)

    my_lang_model_bi.train(training_path)
    my_lang_model_uni.train(training_path)

    print("50 sentences of bi-gram")
    print(my_lang_model_bi.generate(50))

    print("50 sentences of uni-gram")
    print(my_lang_model_uni.generate(50))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print()
    print(".......................................................")
    print("Model: uni-gram, laplace smoothed")
    print("sentences: ")
    print("...")
    print("test corpus: ", training_path1)
    with open(training_path1, "r") as test_file:
        sentences_list = test_file.read().split("\n")
        print("Num of test sentences: ", len(sentences_list))
        prob_list = []
        for s in sentences_list:
            prob_list.append(my_lang_model_uni.score(s))
        print("Average probability: ", sum(prob_list) / len(prob_list))
        print("Standard deviation: ", np.std(prob_list))

    print("test corpus: ", training_path2)
    with open(training_path2, "r") as test_file:
        sentences_list = test_file.read().split("\n")
        print("Num of test sentences: ", len(sentences_list))
        prob_list = []
        for s in sentences_list:
            prob_list.append(my_lang_model_uni.score(s))
        print("Average probability: ", sum(prob_list) / len(prob_list))
        print("Standard deviation: ", np.std(prob_list))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print()
    print(".......................................................")
    print("Model: bi-gram, laplace smoothed")
    print("sentences: ")

    print("test corpus: ", training_path1)
    with open(training_path1, "r") as test_file:
        sentences_list = test_file.read().split("\n")
        print("Num of test sentences: ", len(sentences_list))
        prob_list = []
        for s in sentences_list:
            prob_list.append(my_lang_model_bi.score(s))
        print("Average probability: ", sum(prob_list) / len(prob_list))
        print("Standard deviation: ", np.std(prob_list))

    print("test corpus: ", training_path2)
    with open(training_path2, "r") as test_file:
        sentences_list = test_file.read().split("\n")
        print("Num of test sentences: ", len(sentences_list))
        prob_list = []
        for s in sentences_list:
            prob_list.append(my_lang_model_bi.score(s))
        print("Average probability: ", sum(prob_list) / len(prob_list))
        print("Standard deviation: ", np.std(prob_list))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print()
    print(".......................................................")
    print("# Perplexity for 1-grams: ")
    with open(training_path1, "r") as test_file:
        test_sequence = test_file.readlines()
        test_sequence = ' '.join(test_sequence[:10])
    print("hw2-test.txt: ", my_lang_model_uni.perplexity(test_sequence))
    with open(training_path2, "r") as test_file:
        test_sequence = test_file.readlines()
        test_sequence = ' '.join(test_sequence[:10])
    print("hw2-my-test.txt: ", my_lang_model_uni.perplexity(test_sequence))



if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw2_lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)

    main()
