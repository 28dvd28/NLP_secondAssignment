import sys
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np 
import threading
from tqdm import tqdm



def rotaing_symbol():
    global terminate_rotating_symbol
    symbol = ['|', '/', '-', '\\']
    i = 0
    while (not terminate_rotating_symbol):
        sys.stdout.write('\rSlice_computing: ' + symbol[i%4])
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

def text_elaboration(text):

    punctuation = "!#$%&()*+,-./:;<=>?@[\]^_`{|}~"

    text = text.translate(str.maketrans("", "", punctuation))

    elenco_stopword = stopwords.words('english')
    tokenized_text = word_tokenize(text)
    clean_text = [word for word in tokenized_text if word.lower() not in elenco_stopword]
    lemmatized_text = [WordNetLemmatizer().lemmatize(word) for word in clean_text]

    return lemmatized_text

def vectorize_text(text : str, bow: dict ):

    lemmatized_text = text_elaboration(text.lower())

    vector = bow.copy()

    for word in lemmatized_text:
        if word in vector:
            vector[word] += 1

    for w in vector:
        vector[w] = vector[w]/len(lemmatized_text)

    return list(vector.values())


def cosine_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return vector1.dot(vector2) / ( np.linalg.norm(vector1) * np.linalg.norm(vector2) )
    

def generate_slice(text: str , bow: dict):

    text_slices = {}
    text_in_sentence = sent_tokenize(text)

    cosine_distance_minimum = 0.8

    newslice = ""
    newslice_size = 0

    while True:
        for i in range(len(text_in_sentence)):
            
            sentence = text_in_sentence[i]

            if newslice_size + len(word_tokenize(sentence)) <= 2000:
                newslice = newslice + " " + sentence
                newslice_size += len(word_tokenize(sentence))            
            else:
                i -= 1
                current_slice_vector = vectorize_text(newslice, bow)
                if len(text_slices) == 0:
                    text_slices[newslice] = current_slice_vector
                    newslice = newslice.replace(sent_tokenize(newslice)[0], '')
                    newslice_size = len(word_tokenize(newslice))
                else:
                    cosine_distances = []
                    for vector in list(text_slices.values()):
                        cosine_distances.append(cosine_distance(current_slice_vector, vector))
                    
                    if max(cosine_distances)>=cosine_distance_minimum:
                        newslice = newslice.replace(sent_tokenize(newslice)[0], '')
                        newslice_size = len(word_tokenize(newslice))
                    else:
                        text_slices[newslice] = current_slice_vector
                        newslice = newslice.replace(sent_tokenize(newslice)[0], '')
                        newslice_size = len(word_tokenize(newslice))

        if not (text_in_sentence[len(text_in_sentence)-1] in newslice):
            newslice = newslice + " " +text_in_sentence[len(text_in_sentence)-1]


        while True:
            current_slice_vector = vectorize_text(newslice, bow)
            cosine_distances = []
            for vector in list(text_slices.values()):
                cosine_distances.append(cosine_distance(current_slice_vector, vector))
            
            if max(cosine_distances)>=cosine_distance_minimum or len(word_tokenize(newslice))>2000 :
                newslice = newslice.replace(sent_tokenize(newslice)[0], '')
            else:
                text_slices[newslice] = current_slice_vector
                break

        # count how many tokens are in total for all the slice generated and in the orginal text
        # if the amount of tokens in the slice are more or equal than the original text i will terminate
        dimension = [0, len(word_tokenize(text))]
        for slice in list(text_slices.keys()):
            dimension[0] += len(word_tokenize(slice)) 
        
        if dimension[0] >= dimension[1]:
            break
        else:
            cosine_distance_minimum += 0.05

    returnList = list(text_slices.keys())
    for i in range(len(returnList)):
        if returnList[i].startswith(" "):
            returnList[i] =returnList[i][1:]

    return returnList, dimension


def overlapping_tokens(sliced_text : list):

    total_overlapping = {}

    for i in range(len(sliced_text)-1):

        current = sent_tokenize(sliced_text[i])
        next = sent_tokenize(sliced_text[i+1])

        for j in range(2,min([len(next), len(current)])):
            sub_current = current[len(current)-j+1:]
            sub_next = next[:j]

            if all(x == y for x, y in zip(sub_current, sub_next)):
                if not all(x in sub_next for x in current[0:len(current)-j+1]):
                    s = ""
                    for string in sub_next:
                        s = s + string + " "
                    total_overlapping[f"Slice {i+1} and slice {i+2} overlaps for "]= (200 * len(word_tokenize(s)) / (len(word_tokenize(sliced_text[i])) + len(word_tokenize(sliced_text[i+1]))))
                    break

    return total_overlapping






terminate_rotating_symbol = False

class sliceGenerator:

    def __init__(self, text):
        self.text = text

        self.bow = {}
        
        for word in text_elaboration(self.text):
            self.bow[word] = 0

        thread_rotating_symbol = threading.Thread(target=rotaing_symbol)
        thread_rotating_symbol.start()
        self.sliced_text, self.dimension = generate_slice(self.text, self.bow)
        global terminate_rotating_symbol
        terminate_rotating_symbol = True
        thread_rotating_symbol.join()

        print("\r                                            \r", end='')

        self.overlapping_percentage = overlapping_tokens(self.sliced_text)
