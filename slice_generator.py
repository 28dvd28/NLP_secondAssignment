from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np 

from sklearn.metrics.pairwise import cosine_similarity

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

    newslice = ""
    newslice_size = 0
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
                
                if max(cosine_distances)>=0.9:
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
        
        if max(cosine_distances)>=0.9 or len(word_tokenize(newslice))>2000 :
            newslice = newslice.replace(sent_tokenize(newslice)[0], '')
        else:
            text_slices[newslice] = current_slice_vector
            break     
    

    return text_slices


class sliceGenerator:

    def __init__(self, text):
        self.text = text

        self.bow = {}
        
        for word in text_elaboration(self.text):
            self.bow[word] = 0
        
        self.sliced_text = generate_slice(self.text, self.bow)

    def getSlices(self):
        return self.sliced_text
