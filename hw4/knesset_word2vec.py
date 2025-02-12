import json
import os
import sys
from typing import List, Any
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl_corpus(jsonl_filepath: str) -> List[dict]:
    """
    Loads lines from a JSONL file into a list of dictionaries.
    """
    all_data = []
    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                line = line.strip()
                if not line:
                    continue
                all_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file '{jsonl_filepath}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Unable to decode JSON. Invalid format? Details: {e}")
        sys.exit(1)
    except Exception as ex:
        print(f"An unexpected error occurred while reading the file: {ex}")
        sys.exit(1)

    return all_data


def tokenize_hebrew_sentence(sentence: str) -> list:
    """
      Splits a Hebrew sentence on whitespace to produce tokens.
    """
    # Strip leading/trailing whitespace
    sentence = sentence.strip()

    # Split on whitespace
    tokens = sentence.split()

    return tokens


def preprocess_sentences(corpus_records: list) -> tuple[list[list], list[list[Any]]]:
    """
        Takes a list of corpus records (dicts), extracts the 'sentence_text' key,
        tokenizes, and filters short/ numeric tokens.
    """
    all_tokens_list = []
    cleaned_tokens_list = []
    for record in corpus_records:
        sentence = record.get("sentence_text", "")
        tokenized_sentence = tokenize_hebrew_sentence(sentence)

        # Filter out very short tokens or numbers if desired:
        cleaned_tokens = [t for t in tokenized_sentence if not t.isdigit() and len(t) > 1]

        if cleaned_tokens:
            all_tokens_list.append(tokenized_sentence)
            cleaned_tokens_list.append(cleaned_tokens)
    return all_tokens_list, cleaned_tokens_list


def train_word2vec_model(sentence_list: list,
                         model_save_path: str,
                         vector_size: int = 100,
                         window_size: int = 5,
                         min_count: int = 5,
                         ):
    """
       Trains a Word2Vec model on the given list of sentences and saves it to model_save_path.
    """
    try:
        model = Word2Vec(
            sentences=sentence_list,
            vector_size=vector_size,
            window=window_size,
            min_count=min_count
            # seed=42,
            # workers=1
        )
        model.save(model_save_path)
    except Exception as ex:
        print(f"An error occurred while training/saving the model: {ex}")
        sys.exit(1)
    return model


def print_similar_words(model, words_list, output_file):
    """
    For each word in words_list, find the top 5 similar words and save to output_file.
    """
    word_vectors = model.wv
    vocab_terms = list(word_vectors.key_to_index.keys())  # All tokens in the vocabulary

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in words_list:     # Compute similarity of 'word' to all other vocab words
                if word not in word_vectors:
                    # If the word is not in vocabulary, just write "N/A"
                    line = f"{word}: N/A (not in vocabulary)\n"
                    f.write(line)
                    continue

                similarities = []
                for candidate in vocab_terms:
                    if candidate == word:
                        continue
                    sim = word_vectors.similarity(word, candidate)
                    similarities.append((candidate, sim))

                # Sort by similarity descending
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Take top 5
                top_5 = similarities[:5]

                # Prepare the output line
                line = f"{word}: " + ", ".join([f"({w}, {sim:.4f})" for w, sim in top_5])
                f.write(line + "\n")
    except Exception as ex:
        print(f"Error writing to '{output_file}': {ex}")


def compute_sentence_embedding(sentence_tokens, w2v_model):
    """
    Return the average of the word vectors for the given tokens.
    """
    valid_vectors = []
    for token in sentence_tokens:
        if token in w2v_model:
            valid_vectors.append(w2v_model[token])
    if not valid_vectors:
        return None
    return np.mean(valid_vectors, axis=0)


def create_sentences_embeddings(cleaned_tokenized_sentences, all_tokens_list, w2v_model):
    """
    Build embeddings for each sentence in the corpus.
    Returns a list of (embedding, original_tokens).
    """
    embeddings = []
    for (filtered_sentence_tokens, all_sentence_tokens) in zip(cleaned_tokenized_sentences, all_tokens_list):
        emb = compute_sentence_embedding(filtered_sentence_tokens, w2v_model)
        if emb is not None:
            embeddings.append((emb, all_sentence_tokens))

    return embeddings


def find_most_similar_sentence(query_emb, corpus_embeddings, query_tokens):
    """
    Given a query embedding, find the sentence in corpus_embeddings with highest cosine similarity.
    Returns the tokens of the best match (excluding the exact same sentence).
    """
    if query_emb is None:
        return None

    # Prepare a matrix of embeddings
    X = np.array([emb for emb, _ in corpus_embeddings])
    query_emb_reshaped = query_emb.reshape(1, -1)

    sims = cosine_similarity(query_emb_reshaped, X)[0]
    # Sort similarity in descending order
    sorted_indices = np.argsort(-sims)

    # Check each candidate in order of descending similarity
    for idx in sorted_indices:
        candidate_tokens = corpus_embeddings[idx][1]
        # Compare with query_tokens
        if candidate_tokens != query_tokens:
            return candidate_tokens

    return None


def print_similar_sentences(corpus_embeddings, chosen_indices, output_file="knesset_similar_sentences.txt"):
    """
    For the chosen sentences, find their most similar match in the corpus and save the results.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx in chosen_indices:
                # Original tokens with punctuation retained
                original_all_tokens = corpus_embeddings[idx][1]
                # embedding after punctuation removed
                cur_sent_emb = corpus_embeddings[idx][0]
                best_match_tokens = find_most_similar_sentence(cur_sent_emb, corpus_embeddings,
                                                               original_all_tokens)

                # Convert tokens back to a 'sentence-like' string
                original_sentence_str = " ".join(original_all_tokens)
                best_match_str = " ".join(best_match_tokens) if best_match_tokens else "N/A"

                line = f"{original_sentence_str}: most similar sentence: {best_match_str}"
                f.write(line + "\n")
    except Exception as ex:
        print(f"Error writing to '{output_file}': {ex}")


def get_chosen_sentences_indexes(corpus_embeddings):
    """
    Searches for predefined target sentences in the corpus embeddings.
    """
    target_sentences = [
        "אני מודה לך מאוד .",
        "חבר הכנסת זבולון המר , בבקשה .",
        "אחריו - חבר הכנסת שלמה בניזרי .",
        "תודה לחבר הכנסת זאב בנימין בגין .",
        "אני מזמין את חבר הכנסת יורם לס .",
        "חברי הכנסת , אני פותח את ישיבת הכנסת , יום שני , כ\"ו באייר התשנ\"ג , 17 במאי 1993 .",
        "אם הוא לא יהיה באולם - חבר הכנסת אברהם פורז .",
        "חבר הכנסת אופיר פינס - פז לא נמצא באולם ולכן לא נצביע על הסתייגותו ולא על החלופין .",
        "אני רוצה לסכם , אדוני היושב - ראש .",
        "אנו מצביעים על ההסתייגות של קבוצת גשר .",
    ]
    found_indexes = {}  # Dictionary: sentence -> index (or list of indices)

    for sentence in target_sentences:
        found_indexes[sentence] = []

    for i, (_, tokens_list) in enumerate(corpus_embeddings):
        joined_tokens = " ".join(tokens_list)

        # Check if joined_tokens is in our target sentences
        if joined_tokens in found_indexes:
            # If yes, record this index
            found_indexes[joined_tokens].append(i)

    index_list = []
    for text, cur_index_list in found_indexes.items():
        index_list.append(cur_index_list[0])

    return index_list


def replace_words_with_similar(model, original_sentence, red_words_cfg, topn=3):
    """
       Given a sentence and a configuration of target words with optional positive/negative
       lists, replaces each target word with the best candidate from the model.
    """
    replaced_details = []
    my_wv = model.wv
    new_sentence = original_sentence

    for (target_word, pos_list, neg_list, selected_pos) in red_words_cfg:
        # Gather candidates
        if target_word in my_wv.key_to_index:
            # If target_word is in vocabulary
            similar_candidates = my_wv.most_similar(
                positive=([target_word] + (pos_list if pos_list else [])),
                negative=(neg_list if neg_list else []),
                topn=topn
            )
        else:
            # If target_word not in vocabulary, try using only positive/negative
            if pos_list or neg_list:
                similar_candidates = my_wv.most_similar(
                    positive=(pos_list if pos_list else []),
                    negative=(neg_list if neg_list else []),
                    topn=topn
                )
            else:
                replaced_details.append((target_word, None))
                continue

        cur_pos = 0
        for cand, _ in similar_candidates:
            if cur_pos == selected_pos:
                # Replace only the first occurrence of target_word
                new_sentence = new_sentence.replace(target_word, cand, 1)
                replaced_word = cand
                replaced_details.append((target_word, replaced_word))
                break
            cur_pos += 1

    return new_sentence, replaced_details


def show_replace_red_words(model, original_sentences, red_words_config_per_sentence,
                           output_file_path):
    """
    For each sentence in original_sentences, replaces certain 'red words' based on
    the configuration in red_words_config_per_sentence, writes the results to a file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, (original_sentence, red_word_config) in enumerate(
                    zip(original_sentences, red_words_config_per_sentence),
                    start=1):
                new_sentence, replaced_info = replace_words_with_similar(
                    model,
                    original_sentence,
                    red_word_config,
                    topn=3
                )

                # Format the replaced words info
                replaced_words_str = ",".join([f"({tw}:{rw if rw else 'None'})" for tw, rw in replaced_info])

                line = (
                    f"{i}: {original_sentence}: {new_sentence}\n"
                    f"replaced words: {replaced_words_str}"
                )
                f.write(line + "\n")
    except Exception as ex:
        print(f"Error writing to '{output_file_path}': {ex}")


def main():
    """
        Main function for training a Word2Vec model on a Hebrew corpus,
        finding similar words & sentences, and doing example "red word" replacements.

        Usage:
          python knesset_word2vec.py <path/to/corpus_file.jsonl> <path/to/output_dir>
    """
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python knesset_word2vec.py "
            "<path/to/corpus_file.jsonl> <path/to/output_dir>\n"
        )
        sys.exit(1)

    corpus_path = sys.argv[1]
    out_dir_path = sys.argv[2]

    if not os.path.isdir(out_dir_path):
        # print(f"Output directory '{out_dir_path}' does not exist. Creating...")
        try:
            os.makedirs(out_dir_path, exist_ok=True)
        except Exception as ex:
            print(f"Error creating directory: {ex}")
            sys.exit(1)

    # Load corpus
    corpus_records = load_jsonl_corpus(corpus_path)

    # Tokenize
    all_tokens_list, cleaned_tokens_list = preprocess_sentences(corpus_records)

    saved_model_file_path = os.path.join(out_dir_path, "knesset_word2vec.model")

    if not os.path.isfile(saved_model_file_path):
        # Train model
        model = train_word2vec_model(
            cleaned_tokens_list,
            saved_model_file_path,
            vector_size=100,
            window_size=5,
            min_count=5
        )
    else:
        # Load the pre-trained Word2Vec model
        try:
            model = Word2Vec.load(saved_model_file_path)
        except Exception as ex:
            print(f"Error loading model file: {ex}")
            sys.exit(1)


    # sim1 = model.wv.similarity("כבד", "קל")
    # sim2 = model.wv.similarity("רע", "טוב")
    # sim3 = model.wv.similarity("סוגר", "פותח")
    # sim4 = model.wv.similarity("נגד", "בעד")

    knesset_similar_words_file_path = os.path.join(out_dir_path, "knesset_similar_words.txt")
    # Print similar words
    words_to_inspect = ["ישראל", "גברת", "ממשלה", "חבר", "בוקר", "מים", "אסור", "רשות", "זכויות"]
    print_similar_words(model, words_to_inspect, knesset_similar_words_file_path)

    # Build sentence embeddings
    corpus_embeddings = create_sentences_embeddings(cleaned_tokens_list, all_tokens_list, model.wv)

    knesset_similar_sentences_file_path = os.path.join(out_dir_path, "knesset_similar_sentences.txt")

    chosen_indices = get_chosen_sentences_indexes(corpus_embeddings)
    print_similar_sentences(corpus_embeddings, chosen_indices,
                            knesset_similar_sentences_file_path)

    # red-word replacements
    original_red_sentences = [
        "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים .",
        "בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים .",
        "בוקר טוב , אני פותח את הישיבה .",
        "שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום .",
        "אין מניעה להמשיך לעסוק בנושא ."
    ]

    red_config_per_sentence = [
        # Sentence #1: "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים ."
        [
            ("דקות", ["ספורות"], None, 2),
            ("הדיון", ["הישיבה"], None, 0),
        ],
        # Sentence #2: "בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים ."
        [
            ("הוועדה", ["המליאה"], None, 2),
            ("אני", None, None, 2),
            ("ההסכם", None, None, 0),
        ],
        # Sentence #3: "בוקר טוב , אני פותח את הישיבה ."
        [
            ("בוקר", ["ערב"], None, 0),
            ("פותח", ["מתחיל"], None, 0),
        ],
        # Sentence #4: "שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום ."
        [
            ("שלום", None, None, 1),
            ("שמחים", ["רגועים", "מרוצים"], None, 1),
            ("היקר", ["החשוב","החיובי"], None, 0),
            ("קידום", ["תפקיד", "מפעל"], None, 1),
        ],
        # Sentence #5: "אין מניעה להמשיך לעסוק בנושא ."
        [
            ("מניעה", ["בעיה"], None, 1),
        ],
    ]

    red_words_sentences_file_path = os.path.join(out_dir_path, "red_words_sentences.txt")
    show_replace_red_words(
        model,
        original_red_sentences,
        red_config_per_sentence,
        red_words_sentences_file_path
    )


if __name__ == "__main__":
    main()
