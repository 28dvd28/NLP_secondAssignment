import os
import sys
import time
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from slice_generator import sliceGenerator
from openai import OpenAI
from dotenv import load_dotenv

# this function simply eliminate all the files previously obtained inside the output folder
def clear_output_folder():
    lista_file = os.listdir("output")
    for file in lista_file:
        percorso_file = os.path.join("output", file)
        try:
            if os.path.isfile(percorso_file):
                os.remove(percorso_file)
        except Exception as e:
            print(f"Errore durante l'eliminazione del file {file}: {e}")

# the function that get the slices saved inside a list and send one by one to the LLM 
# After getting the response he save in the correspondent file the slice and the correspondent response
# all inside the output folder of course
def send_to_LLM(slice_computed):

    print("Sending slice to the LLM")

    load_dotenv()

    client = OpenAI(
        api_key = os.getenv("LLAMA-KEY"),
        base_url = "https://api.llama-api.com"
        )

    j=0
    error = False
    while j < len(slice_computed):

        time.sleep(1) # waiting time to avoid problems with LLM

        if not error:
            print(f"    Sending slice {j+1}", end='\r')

        slice = slice_computed[j]

        new_file = "slice " + str(slice_computed.index(slice)+1) + ".txt"
        with open(os.path.join("output", new_file), 'w', encoding='utf-8') as file:
            file.write(slice)

            input_string = slice

            # Execute the Request and in case redo it until not getting the right answer
            try:
                response = client.chat.completions.create(
                    model="llama-70b-chat",
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": input_string}
                    ])
            except Exception as e:
                print(f"    Retrying slice {j} after error: {e}", end='\r')
                error = True
                continue

            file.write("\n\n\nRESPONSE:\n" + response.choices[0].message.content)    

        j += 1
        error = False




if __name__ == "__main__":

    while True:
        file_name = os.path.join("input", input(">>>Insert the name of the file to be sliced: "))
        if not os.path.exists(file_name):
            print(f">>>File not found")
        else:
            break

    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()

    clear_output_folder()

    #if the output is under the context window then it is send as it is
    if len(word_tokenize(text))<2048:
        slice_computed = [text]
        print("The slice fits the context window so it will be sent as it is to the LLM")
        send_to_LLM(slice_computed)
        print("Slice and the correspondent response can be found inside the output folder")
        exit()

    
    # initialization of the sliceGenerator and conseguent slice computing, then are also calculated 
    # all the data about the tokens and the overlapping and output of all the information
    slice_generator = sliceGenerator(text)
    slice_computed = slice_generator.sliced_text

    print("Computed slices:      ", len(slice_computed))
    print("Slice tokens total:   ", slice_generator.dimension[0])
    print("Original text tokens: ", slice_generator.dimension[1])

    print("OVERLAPPING PERCENTAGE:")
    for x in slice_generator.overlapping_percentage:
        print(f"    - {x}{round(slice_generator.overlapping_percentage[x], 2)}% of their tokens")
    print()

    send_to_LLM(slice_computed)
    
    sys.stdout.write("\x1b[1A")
    print("Request collected, slice and the correspondent response can be found inside the output folder")
    print("                                                                                             ")
