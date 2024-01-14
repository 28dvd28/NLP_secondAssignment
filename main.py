import os
import time
from llamaapi import LlamaAPI
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from slice_generator import sliceGenerator
from openai import OpenAI


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

    _k1 = "AbG".split("b")[1]
    _k2 = "EWtbxl8qK56y1VUlIGbgkK1inb"
    _k3 = "aTEyh6TuBsZgSdMR1znho1f9pcX22bnvMwJ"
    
    client = OpenAI(
        api_key = f"LL-{_k2}LZ{_k3}{_k1}",
        base_url = "https://api.llama-api.com"
        )

    barra = tqdm(total=len(slice_computed), desc='Invio slice a llama-13b-chat', position=0, leave=False) # just a progress bar used to show that the software is still running
    for j in range(len(slice_computed)):

        time.sleep(1)

        slice = slice_computed[j]

        new_file = "slice " + str(slice_computed.index(slice)+1) + ".txt"
        with open(os.path.join("output", new_file), 'w', encoding='utf-8') as file:
            file.write(slice)

            input_string = slice

            # Execute the Request and in case redo it until not getting the right answer
            try:
                response = client.chat.completions.create(
                    model="llama-13b-chat",
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": input_string}
                    ])
                file.write("\n\n\nRESPONSE:\n" + response.choices[0].message.content)
            except Exception as e:
                print(e)
                j -= 1

        barra.update(1)

    barra.close()

if __name__ == "__main__":

    with open("input_text_to_slice.txt", 'r', encoding='utf-8') as file:
        text = file.read()

    clear_output_folder()
    
    # initialization of the sliceGenrator and conseguent slice computing, then are also calculated 
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
    
    print("Slice and the correspondent response can be found inside the output folder")
