
import re
import csv


def get_split(file_path):
    if 'training' in file_path:
        return 'training'
    else:
        return 'test'
    

def get_doc_info(header):
    doc_id = re.findall(r'id="(.*?)"', header)[0]
    genre = re.findall(r'genre="(.*?)"', header)[0]
    gender = re.findall(r'gender="(.*?)"', header)[0]
    return doc_id, genre, gender


def load_dataset(src_path):
    features_names = None
    documents_info = []
    features = []
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file, delimiter='\t')
        for row in csv_reader:
            if features_names is None: # la prima riga del csv contiene 'Filename' e tutti i nomi delle features
                features_names = row[1:]
            else:
                documents_info.append(row[0]) # il primo elemento di ogni riga contiene il nome del file, dove sono codificati lo split e le feature
                features.append(row[1:]) # dal secondo elemento in poi ci sono le features
    return features_names, documents_info, features


def create_label_list(documents_info):
    labels = []
    for doc_info in documents_info:

        doc_info = doc_info[0:-len('.conllu')]
        splitted_doc_info = doc_info.split('#')

        genre = splitted_doc_info[2]
        gender = splitted_doc_info[3]

        labels.append(gender)
    return labels


def train_test_split(documents_info, features, labels):
    train_features, test_features = [], []
    train_labels, test_labels = [], []

    for doc_info, doc_features, doc_label in zip(documents_info, features, labels): # for idx in range(len(documents_info)):
        if 'training' in doc_info:
            train_features.append(doc_features)                                     # train_features.append(documents_info[idx])
            train_labels.append(doc_label)
        else: # if 'test' in file_name
            test_features.append(doc_features)
            test_labels.append(doc_label)

    return train_features, train_labels, test_features, test_labels


class Document:

    def __init__(self, document_path):
        self.document_path = document_path
        self._parse_doc_info(document_path)
        self.sentences = []
        self.features = None

    def _parse_doc_info(self, document_path):
        document_path = document_path.split('/')[-1]
        document_info = document_path.split('.')[0]
        document_info = document_info.split('#')
        self.split = document_info[0]
        self.genre = document_info[2]
        self.gender = document_info[3]

    def add_sentence(self, sentences):
        self.sentences.append(sentences)

    # Per dopo

    def get_num_tokens(self):
        num_words = 0
        for sentence in self.sentences:
            num_words = num_words + sentence.get_num_tokens()
        return num_words

    def get_num_chars(self):
        num_chars = 0
        for sentence in self.sentences:
            sentence_char_len = sentence.get_num_chars()
            num_chars = num_chars + sentence_char_len
        return num_chars

class Sentence:

    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    # Per dopo

    def get_words(self):
        return [token.word for token in self.tokens]

    def get_lemmas(self):
        return [token.lemma for token in self.tokens]

    def get_pos(self):
        return [token.pos for token in self.tokens]

    def get_num_tokens(self):
        return len(self.tokens)

    def get_num_chars(self):
        num_chars = 0
        for token in self.tokens:
            num_chars = num_chars + token.get_num_chars()
        num_chars = num_chars + self.get_num_tokens() - 1 # contiamo anche gli spazi
        return num_chars

    def __str__(self):
        return ' '.join([token.word for token in self.tokens])

class Token:

    def __init__(self, word, lemma, pos):
        self.word = word
        self.lemma = lemma
        self.pos = pos


    # Per dopo

    def get_num_chars(self):
        return len(self.word)
    

def load_document_sentences(document):
    sentence = Sentence()
    for line in open(document.document_path, 'r'):
        if line[0].isdigit():  # se la riga inizia con un numero
            splitted_line = line.strip().split('\t')
            if '-' not in splitted_line[0]:  # se l'id della parola non contiene un trattino
                token = Token(splitted_line[1], splitted_line[2], splitted_line[3])
                sentence.add_token(token)
        if line == '\n':  # se la riga è vuota significa che la frase è finita
            document.add_sentence(sentence)
            sentence = Sentence()


def extract_word_ngrams_from_sentence(word_ngrams, sentence, el, n):
    # creiamo una lista con tutte le parole
    if el == 'word':
        all_words = sentence.get_words()
    elif el == 'lemma':
        all_words = sentence.get_lemmas()
    elif el == 'pos':
        all_words = sentence.get_pos()
    else:
        raise Exception(f'Invalid element {el}')
    

    # scorriamo la lista delle parole ed estraiamo gli n-grammi
    for i in range(0, len(all_words) - n + 1): # -n+1 serve per non uscire dal vettore
        ngram_words = all_words[i: i + n]
        ngram = f'{el.upper()}_{n}_' + '_'.join(ngram_words)
        # print(f'{i}: {ngram_words} -> {ngram}')
        if ngram not in word_ngrams:
            word_ngrams[ngram] = 1
        else:
            word_ngrams[ngram] += 1

    return word_ngrams


def extract_char_ngrams_from_sentence(char_ngrams, sentence, n):
    # creiamo una lista con tutte le parole
    all_words = sentence.get_words()

    # creiamo una stringa che contenga tutte le parole separate tra spazi perchè vogliamo scorrere i caratteri
    all_words = ' '.join(all_words)
    # print(all_words)
    # all_words = all_words.lower()

    # scorriamo la stringa ed estraiamo gli n-grammi di caratteri
    for i in range(0, len(all_words) - n + 1):
        ngram_chars = all_words[i:i + n]
        ngram = f'CHAR_{n}_' + ngram_chars
        # print(f'{i}: {ngram_chars} -> {ngram}')

        if ngram not in char_ngrams:
            char_ngrams[ngram] = 1
        else:
            char_ngrams[ngram] += 1

    return char_ngrams


def extract_documents_ngrams(all_documents):
    for document in all_documents:
        document_ngrams = dict()
        for sentence in document.sentences:
            extract_word_ngrams_from_sentence(document_ngrams, sentence, 'word', 1)
            extract_word_ngrams_from_sentence(document_ngrams, sentence, 'word', 2)
            extract_char_ngrams_from_sentence(document_ngrams, sentence, 1)
            extract_char_ngrams_from_sentence(document_ngrams, sentence, 2)

        document.features = document_ngrams


def normalize_ngrams(ngrams_dict, doc_len):
    for ngram in ngrams_dict:
        ngrams_dict[ngram] = ngrams_dict[ngram] / float(doc_len)
        
def extract_documents_ngrams_normalized(all_documents):
    for document in all_documents:
        word_unigrams = dict()
        word_bigrams = dict()
        char_unigrams = dict()
        char_bigrams = dict()
        for sentence in document.sentences:
            extract_word_ngrams_from_sentence(word_unigrams, sentence, 'word', 1)
            extract_word_ngrams_from_sentence(word_bigrams, sentence, 'word', 2)
            extract_char_ngrams_from_sentence(char_unigrams, sentence, 1)
            extract_char_ngrams_from_sentence(char_bigrams, sentence, 2)

        num_words = document.get_num_tokens()
        num_chars = document.get_num_chars()
        normalize_ngrams(word_unigrams, num_words)
        normalize_ngrams(word_bigrams, num_words)
        normalize_ngrams(char_unigrams, num_chars)
        normalize_ngrams(char_bigrams, num_chars)

        document_ngrams = word_unigrams | word_bigrams | char_unigrams | char_bigrams

        document.features = document_ngrams



def get_num_features(features_dict):
    all_features = set()
    for document_feats in features_dict:
        all_features.update(list(document_feats.keys()))
    return len(all_features)

def filter_features(train_features_dict, min_occurrences):
    # contiamo ogni feature in quanti user diversi compare
    features_counter = dict()
    for document_features_dict in train_features_dict:
        for feature in document_features_dict:
            if feature in features_counter:
                features_counter[feature] += 1
            else:
                features_counter[feature] = 1

    # per ogni user, togliamo le features che compaiono in meno di "min_occurrences" utenti
    for document_features_dict in train_features_dict:
        document_features = list(document_features_dict.keys())
        for feature in document_features:
            if features_counter[feature] < min_occurrences:
                document_features_dict.pop(feature)

    return train_features_dict
