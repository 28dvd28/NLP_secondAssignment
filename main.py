import json
from llamaapi import LlamaAPI
from nltk.tokenize import word_tokenize
from slice_generator import sliceGenerator

if __name__ == "__main__":

    with open("input_text_to_slice.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    
    slice_generator = sliceGenerator(text)
    slice_computed = list(slice_generator.getSlices().keys())

    dimension = [0, len(word_tokenize(text))]

    for slice in slice_computed:
        dimension[0] += len(word_tokenize(slice))

    print("Dimensione: " + str(dimension))

    # Initialize the SDK
    llama = LlamaAPI("LL-EWtbxl8qK56y1VUlIGbgkK1inbLZaTEyh6TuBsZgSdMR1znho1f9pcX22bnvMwJG")

    for slice in slice_computed:

        input_string = "Make a summary of the following text:\n" + slice
        # Build the API request
        api_request_json = {
            "messages": [
                {"role": "user", "content": input_string},
            ],
            "stream": False,
        }

        # Execute the Request
        response = llama.run(api_request_json)
        print("RESPONSE: " + response.json()["choices"][0]["message"]["content"])
        print("\n\n\n")
