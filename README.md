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
  8. openai
  9. dotenv

## Generating the slice
After that it has been checked that the text given in input is bigger than the context window off the LLM, it is divided into the slices. The generation of the slices begin with the sentence tokenisation of the text, then the first slice is computed by adding the sentences from the beginning of the text. When it is reached the maximum length of the context window the slice is saved. Now the second slice, like also all the other ones, is computed taking the sentence of the previous slice except the first one and then do the iteration seen before except that when the max length is reched the slice is saved only if the cosine distance from all the others is under a specific value. 
So basically the idea is slide along the text and save all the slice that results to be different enough. 
All the details over the code implementation can be seen inside the code itself that has been commentend in detail. 

### LLama
The LLM used is LLama 70B. It is good enough to generate a response coerent enough to the slice sent even without adding any question or other modification to it. It's context window is 2048 tokens, so this is the value used to compute the text's slices. 

### Vectorization
The vector of each text is computed from the bag of words of the text to be sliced. The bag of words is obtained from the words that still remains after the stopwords elimination and the lemmatization over the text tokenized in words. Then, for each slice genereted the vector is simply a list with the frequency of all the words in the bow. The words that doesn't appear inside the slice will have frequency 0.  

### Cosine distance threshold adaptation
Doing some tests it is noticed that using the same threshold will not guarantee that, for every input text, the slice generated will get a number of tokens grather or equals from the original text. To avoid this situation we start from a threshold of 0.8, that is that every slice that have cosine similarity lower than 0.8 will be considered different enough, will be increased by 0.05 each time. Increasing the cosine similarity consider means to increase also the number of slices that are considered different enough. This will be repeated unitil the slicer gat the expected number of tokens.

This process will also terminate in the case the threshold go over 0.95 because if we get closer to 1  means that even few sentence of difference between the slices are enough to make them diferent enough. Using out method that iterates all over the sentence of the text make means that with a threshold closer to one generate a lot of slice that are different from the previous one only for very few sentences.

## Execution of the program
To get the correct execution of the program and avoid errors:
  - save your llama api key inside the .env file, make sure to put it into the LLAMA-KEY field
  - the program must be executed inside the same folder where it is saved the main
  - before the execution insert inside the _input_ folder the .txt file containing the text to be sliced
  - at the beginning of the execution it will be asked to insert the name of the file. **Important**: insert only the name, not the path of the file, and don't forget the extension .txt
  - the output will be displayed inside the output folder. At the begin of the execution any file in this folder will be deleted, so save them before if you want to keep them.