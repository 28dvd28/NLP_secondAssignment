# NLP second assignment 

So the deploy of this project consists in two file:
 - slice_generator.py implement a class that during its initialization compute the slice of the input text
 - main.py make an instance of the class described above and send the slices to the LLM, saving the slice and the response into a text file inside the folder _output_

Before getting into the details of the implementation, the following libraries must be installated in your environment:
  1. nltk
  2. tqdm
  3. sys
  4. time
  5. numpy
  6. threading
  7. llamaapi
  8. openai

## Generating the slice
After that it has been checked that the text given in input is bigger than the context window off the LLM, it is divided into the slice. The generation of the slice begin with the sentence tokenisation of the text, then the first slice is computed by adding the sentence from the beginning of the text. When it is reached the maximum length of the context window the slice is saved. Now the second slice, like also all the other ones, is computed taking the sentence of the previous slice except the first one and then do the iteration seen before except that when the max length is reched the slice is saved only if the cosine distance from all the others is under a specific value. 
So basically the idea is slide along the text and save all the slice that results to be different enough. 
All the details over the code implementation can be seen inside the code itself that has been commentend in detail. 

### LLama
The LLM used is LLama ...
In order to get better response is added...

### Vectorization
The vector of each text is computed from the bag of words of the text to be sliced. The bag of words is obtained from the words that still remains after the stopwords elimination and the lemmatization over the text tokenized in words. Then, for each slice genereted the vector is simply a list with the frequency of all the words in the bow. The words that doesn't appear inside the slice will have frequency 0.  

### Cosine distance threshold adaptation

## Input prompt

