import gensim.downloader as api
import re

def load_model(model_name='glove-twitter-25'):
    # Load the pre-trained Word2Vec model
    model = api.load(model_name)
    return model

def get_word_vector(model, word):
    try:
        # Get the word vector
        vector = model[word]
        return vector
    except KeyError:
        # Handle the case where the word is not in the vocabulary
        return None

def process_line(line, model):
    # Extract word1, word2, and word3 from the line
    match = re.match(r"Like (.+?) is to (.+?), (.+?) is to\?", line.strip())
    if match:
        word1, word2, word3 = match.groups()

        # Get vectors for the three words
        vector1 = get_word_vector(model, word1)
        vector2 = get_word_vector(model, word2)
        vec_target = get_word_vector(model, word3)

        if vector1 is not None and vector2 is not None and vec_target is not None:
            # Calculate analogy vector
            analogy_vector = vec_target + (vector2 - vector1)

            # Find the most similar word to the analogy vector
            result = model.similar_by_vector(analogy_vector, topn=1)
            print(f"'{word1} is to {word2} as {word3} is to {result[0][0]}' with similarity {result[0][1]}")
            global flag
            flag += (f"{result[0][0]}")
        else:
            missing_words = [word for word, vec in zip([word1, word2, word3], [vector1, vector2, vec_target]) if vec is None]
            print(f"The following words were not found in the model: {', '.join(missing_words)}")
    else:
        print(f"Line format is incorrect: {line.strip()}")

def process_file(filename, model):
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            process_line(line, model)

def main():
    # Load the model
    model = load_model()

    # Path to your text file
    filename = 'chal.txt'  # Replace with your file path

    # Process the file and calculate analogy vectors for each line
    process_file(filename, model)

if __name__ == "__main__":
    flag = ""
    main()
    print(flag)
