
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