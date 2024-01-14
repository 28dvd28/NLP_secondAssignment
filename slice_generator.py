import sys
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np 
import threading
from tqdm import tqdm


# a function used just to output on the terminal a rotating symbol to
# show that the software is still computing
def rotaing_symbol():
    global terminate_rotating_symbol
    symbol = ['|', '/', '-', '\\']
    i = 0
    while (not terminate_rotating_symbol):
        sys.stdout.write('\rSlice_computing: ' + symbol[i%4])
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

# function that extract from a inputn text the important word 
# eliminating all the symbol, the stopwords and making the tokenization and also
# the lemmatization
def text_elaboration(text):

    punctuation = "!#$%&()*+,-./:;<=>?@[\]^_`{|}~"

    text = text.translate(str.maketrans("", "", punctuation))

    elenco_stopword = stopwords.words('english')
    tokenized_text = word_tokenize(text)
    clean_text = [word for word in tokenized_text if word.lower() not in elenco_stopword]
    lemmatized_text = [WordNetLemmatizer().lemmatize(word) for word in clean_text]

    return lemmatized_text


# given a text and a bag of words it generate a vector where each position 
# correspond to a word inside the bag of words and contains it's frequency inside the
# given text. 
def vectorize_text(text : str, bow: dict ):

    lemmatized_text = text_elaboration(text.lower())

    vector = bow.copy()

    for word in lemmatized_text:
        if word in vector:
            vector[word] += 1

    for w in vector:
        vector[w] = vector[w]/len(lemmatized_text)

    return list(vector.values())

# function that compute the cosine distance with the 
# dot operation between the two vectors given in input
def cosine_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return vector1.dot(vector2) / ( np.linalg.norm(vector1) * np.linalg.norm(vector2) )
    
# function that given a text, it generate the slice in such a way that the total number of tokens
# of this slice are more or equals number of the tokens of the original text. The threshold is 0.8
# so each text which cosine distance is < than this value is considered different enough. This threshold 
# will be increased if at the end of the loop the tokens are still less than in the orginal text
def generate_slice(text: str , bow: dict):

    # the generation of a slice starts from the sentence tokenisation of the input text and then
    # each slice will be composed by adding the sentences. Each slice will be saved inside a dictionary 
    # where the key is the slice and the value is it correspondent vector
    text_slices = {}
    text_in_sentence = sent_tokenize(text)

    cosine_distance_minimum = 0.8

    newslice = ""
    newslice_size = 0

    while True:
        for i in range(len(text_in_sentence)):
            
            sentence = text_in_sentence[i]

            # if the slice is still too small i just add the sentence and update the newslice_size
            #variable containing the number of tokens inside the slice
            if newslice_size + len(word_tokenize(sentence)) <= 2000:
                newslice = newslice + " " + sentence
                newslice_size += len(word_tokenize(sentence)) 
            # otherwise i will look if the slice is different enough
            # from the previous and in case i will save it           
            else:
                i -= 1 # i will put i to the previous value so at the next iteration i will reconsider the current sentence that otherwise will be lost
                current_slice_vector = vectorize_text(newslice, bow) # get the vector of the current slice

                # if the dict is empty i just add the slice otherwise i do the computation
                if len(text_slices) == 0:
                    text_slices[newslice] = current_slice_vector
                    newslice = newslice.replace(sent_tokenize(newslice)[0], '')
                    newslice_size = len(word_tokenize(newslice))
                else:
                    cosine_distances = [] # vector of the distances from all the others slices
                    for vector in list(text_slices.values()):
                        cosine_distances.append(cosine_distance(current_slice_vector, vector)) # compute the cosine_distance
                    
                    # if there exist a text where the cosine distance is greather than the value
                    # decided, then i will just remove the first sentence and procede in the composition of my slice
                    if max(cosine_distances)>=cosine_distance_minimum:
                        newslice = newslice.replace(sent_tokenize(newslice)[0], '')
                        newslice_size = len(word_tokenize(newslice))
                    # otherwise i will save my slice with its correspondent vector and i will remove
                    # the first sentence because doing so i will get the next slice that maybe will overlap
                    # but it will not be fully contained inside the previous
                    else:
                        text_slices[newslice] = current_slice_vector
                        newslice = newslice.replace(sent_tokenize(newslice)[0], '')
                        newslice_size = len(word_tokenize(newslice))

        # just add the last sentence that maybe is not been added to the last slice
        if not (text_in_sentence[len(text_in_sentence)-1] in newslice):
            newslice = newslice + " " +text_in_sentence[len(text_in_sentence)-1]

        # in this loop the last slice is elaborated in order to make it satisfy
        # the constraint described at the beginning of the function, basically with a loop
        # similar to the one used for the other slices
        while True:
            current_slice_vector = vectorize_text(newslice, bow) # get the vector
            cosine_distances = []
            for vector in list(text_slices.values()): # compute all the cosine distances
                cosine_distances.append(cosine_distance(current_slice_vector, vector))
            
            #check if it is ok or not
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

    # then from the dict i just get the slices and return it
    returnList = list(text_slices.keys())
    for i in range(len(returnList)):
        if returnList[i].startswith(" "):
            returnList[i] =returnList[i].lstrip()

    return returnList, dimension


# function that for all slices computes how much does they overlap.
# This value is computed as a percentage of the total tokens of the two slice
# that actualli overlapp with the other text
def overlapping_tokens(sliced_text : list):

    total_overlapping = {}

    for i in range(len(sliced_text)-1):

        current = sent_tokenize(sliced_text[i]) # tokenization of the first text
        next = sent_tokenize(sliced_text[i+1]) # tokenization of the next text

        # to know when two slice are overlapping just i consider the begin of the next and
        # the end of the previous, increasing how many sentence of this consider until
        # i found that they are equals and that no other occurences are contained
        for j in range(2,min([len(next), len(current)])):
            sub_current = current[len(current)-j+1:]
            sub_next = next[:j]

            if all(x == y for x, y in zip(sub_current, sub_next)): # when all sentence are equals then i found the overlapping portion
                if not all(x in sub_next for x in current[0:len(current)-j+1]):
                    s = ""
                    for string in sub_next:
                        s = s + string + " "
                    total_overlapping[f"Slice {i+1} and slice {i+2} overlaps for "]= (200 * len(word_tokenize(s)) / (len(word_tokenize(sliced_text[i])) + len(word_tokenize(sliced_text[i+1]))))
                    break

    # the value returned is a dict where the key says between which slice is computed the correspondent value
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
