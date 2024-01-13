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

As 
