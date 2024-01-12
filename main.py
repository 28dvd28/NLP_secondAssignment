import os
from llamaapi import LlamaAPI
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from slice_generator import sliceGenerator

def clear_output_folder():
    lista_file = os.listdir("output")
    for file in lista_file:
        percorso_file = os.path.join("output", file)
        try:
            if os.path.isfile(percorso_file):
                os.remove(percorso_file)
        except Exception as e:
            print(f"Errore durante l'eliminazione del file {file}: {e}")

def send_to_LLM(slice_computed):
    llama = LlamaAPI("LL-EWtbxl8qK56y1VUlIGbgkK1inbLZaTEyh6TuBsZgSdMR1znho1f9pcX22bnvMwJG")


    barra = tqdm(total=len(slice_computed), desc='Invio slice a LLama', position=0, leave=False)
    for j in range(len(slice_computed)):

        slice = slice_computed[j]

        new_file = "slice " + str(slice_computed.index(slice)+1) + ".txt"
        with open(os.path.join("output", new_file), 'w', encoding='utf-8') as file:
            file.write(slice)

            input_string = slice
            # Build the API request
            api_request_json = {
                "messages": [
                    {"role": "user", "content": input_string},
                ],
                "stream": False,
            }

            # Execute the Request
            try:
                response = llama.run(api_request_json)
                response_string = "\n\n\nRESPONSE:\n" + response.json()["choices"][0]["message"]["content"]
                file.write(response_string)
            except LlamaAPI.JSONDecodeError:
                response_string = "\n\n\nRESPONSE:\n" + "error"
                file.write(response_string)

        barra.update(1)

    barra.close()

if __name__ == "__main__":

    with open("input_text_to_slice.txt", 'r', encoding='utf-8') as file:
        text = file.read()

    clear_output_folder()
    
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
    
    print("Slice e rispettive risposte visualizzabili all'interno della cartella output")
