import random
import numpy as np
import json
import pandas as pd
import sys
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# ----------------------------
# Global random seeds
# ----------------------------
random.seed(42)
np.random.seed(42)


def tokenize_speaker_name(name_str: str) -> set:
    """
        Splits the speaker name on spaces/apostrophes and returns a set of tokens.
        Example: "בנימין גנץ" -> {"בנימין", "גנץ"}.
    """
    tokens = name_str.split()
    # Remove trailing apostrophes if any
    tokens = [t.rstrip("'") for t in tokens]
    return set(tokens)


def is_unified_speaker(base_name: str, candidate_name: str) -> bool:
    """
     Checks if candidate_name is effectively the same speaker as base_name.
     1. If exactly the same string -> return True.
     2. Otherwise, tokenizes both names and looks for matching or similar tokens.
     3. A token is considered 'equal' if exactly the same, or 'similar' if one is
        a substring of the other. If found sufficient matches, returns True.
     This function helps merge variations of a speaker's name into a single label.
     """
    if base_name == candidate_name:
        return True

    base_tokens = tokenize_speaker_name(base_name)
    cand_tokens = tokenize_speaker_name(candidate_name)

    if not base_tokens or not cand_tokens:
        return False

    def token_similarity_check(base_tok, cand_tok_set, tokens_to_exclude, similar_tokens_set):
        """
           checks a single token (base_tok) against all tokens
           in cand_tok_set. Returns:
             - ('equal', cand_token) if there's an exact match
             - ('similar', cand_token) if there's a substring match
             - ('different', None) otherwise
           tokens_to_exclude: tokens we already matched exactly
           similar_tokens_set: tokens marked as 'similar' previously
        """
        match_status = "different"
        similar_token = None
        for cand in cand_tok_set:
            if cand in tokens_to_exclude:
                continue
            if cand == base_tok:
                return "equal", cand
            if cand not in similar_tokens_set and (cand.startswith(base_tok) or base_tok.startswith(cand)):
                match_status = "similar"
                similar_token = cand
        return match_status, similar_token

    count_similarities = 0
    found_exact = False
    match_tokens = set()
    similar_tokens = set()
    for i in range(2):
        for b_tok in base_tokens:
            if i == 0 or b_tok not in match_tokens:
                result, token_to_exclude = token_similarity_check(b_tok, cand_tokens, match_tokens, similar_tokens)
                if result == "equal":
                    found_exact = True
                    count_similarities += 1
                    match_tokens.add(token_to_exclude)
                elif result == "similar":
                    count_similarities += 1
                    similar_tokens.add(token_to_exclude)

        # Check if there's overlap between exact matches and similar tokens
        common_strings = match_tokens & similar_tokens
        if common_strings:
            # If there's overlap, consider them matched
            count_similarities = len(match_tokens)
            similar_tokens.clear()
        else:
            break

    # Simple heuristic: at least one 'equal' match + multiple similarities
    if found_exact and count_similarities > 1:
        return True
    return False


def prepare_data_for_binary_classification(df_corpus: pd.DataFrame, label_column: str):
    """
       Identifies the top 2 speakers by frequency, merges any variations for them,
       and down-samples so both classes have equal size.
    """
    # Count how many sentences per speaker
    speaker_counts = df_corpus["speaker"].value_counts()
    if len(speaker_counts) < 2:
        print("Fewer than 2 unique speakers. Exiting...")
        sys.exit(0)

    main_speaker_1, main_speaker_2 = speaker_counts.index[0], speaker_counts.index[1]

    # print(f"Top speakers:\n1) {freq_speaker_1}\n2) {freq_speaker_2}\n")
    df_multi_local = df_corpus.copy()

    def label_multi(spk: str):
        """
          For each speaker name 'spk', check if it unifies with freq_speaker_1 or freq_speaker_2.
          Otherwise, label as 'other'.
        """
        if is_unified_speaker(spk, main_speaker_1):
            return "first"
        elif is_unified_speaker(spk, main_speaker_2):
            return "second"
        return "other"

    df_multi_local[label_column] = df_multi_local["speaker"].apply(label_multi)

    df_first = df_multi_local[df_multi_local[label_column] == "first"]
    df_second = df_multi_local[df_multi_local[label_column] == "second"]

    # print("Binary classification class sizes (before down-sampling):")
    # print(f"  {freq_speaker_1}: {len(df_first)}")
    # print(f"  {freq_speaker_2}: {len(df_second)}")

    # Down-sample so both classes have the same number of rows
    min_count = min(len(df_first), len(df_second))
    df_first_down_sample = df_first.sample(n=min_count, random_state=42)
    df_second_down_sample = df_second.sample(n=min_count, random_state=42)

    df_binary_balanced = pd.concat([df_first_down_sample, df_second_down_sample]).sample(frac=1, random_state=42)

    # print("After down-sampling:")
    # print(f"  {freq_speaker_1}: {len(df_first_down_sample)}")
    # print(f"  {freq_speaker_2}: {len(df_second_down_sample)}")

    return df_binary_balanced


def evaluate_models(
        knn_classifier,
        word2vec_model,
        df_for_eval: pd.DataFrame,
        label_col,
        sentences_col,
):
    """
    Runs a 5-fold cross-validation KNN classification using the average of
    Word2Vec embeddings as features.
    """
    labels = df_for_eval[label_col].values
    sentences = df_for_eval[sentences_col].values

    # We use StratifiedKFold to preserve class distribution in each fold
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Compute embeddings
    embeddings = []
    for sentence in sentences:
        emb = sentence_embedding(sentence, word2vec_model.wv)
        embeddings.append(emb)

    features = np.array(embeddings)
    # print("=== Sentence Embeddings Features ===")
    try:
        pred_knn_word2vec = cross_val_predict(knn_classifier, features, labels, cv=skfold)
        # print("KNN Results:")
        print(classification_report(labels, pred_knn_word2vec, digits=3))
    except Exception as e:
        print(f"Error during TF-IDF evaluation: {e}")


def tokenize_hebrew_sentence(sentence: str) -> list:
    """
    Splits a Hebrew sentence on whitespace to produce tokens.
    """
    # Strip leading/trailing whitespace
    sentence = sentence.strip()

    # Split on whitespace
    tokens = sentence.split()

    return tokens


def sentence_embedding(sentence, wv):
    """
    Return the average of the word vectors for the given tokens.
    """
    sentence_tokens = tokenize_hebrew_sentence(sentence)
    cleaned_tokens = [t for t in sentence_tokens if not t.isdigit() and len(t) > 1]
    valid_vectors = []
    for token in cleaned_tokens:
        if token in wv:
            valid_vectors.append(wv[token])
    if not valid_vectors:
        return np.zeros(wv.vectors.shape[1])
    return np.mean(valid_vectors, axis=0)


def main():
    """
       Main function for loading a corpus, preparing data for binary classification among
       two most frequent speakers, loading a Word2Vec model, and evaluating KNN in a 5-fold CV.
    """
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python knesset_word2vec_classification.py "
            "<path/to/corpus_file.jsonl> <path/to/knesset_word2vec.model>\n"
        )
        sys.exit(1)

    corpus_path = sys.argv[1]
    model_path = sys.argv[2]

    label_col_name = 'label'
    data_col_name = 'sentences'

    # Load the corpus
    try:
        corpus_data = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record_json = json.loads(line)
                corpus_data.append({
                    "speaker": record_json.get("speaker_name", ""),
                    data_col_name: record_json.get("sentence_text", "")
                })
        df_corpus = pd.DataFrame(corpus_data)
    except FileNotFoundError:
        print(f"Error: Corpus file '{corpus_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as jex:
        print(f"Error decoding JSON from corpus file: {jex}")
        sys.exit(1)
    except Exception as ex:
        print(f"Error loading corpus file: {ex}")
        sys.exit(1)

    # Prepare data for binary classification (two top speakers)
    df_binary_balanced = prepare_data_for_binary_classification(df_corpus,
                                                                label_col_name)
    # Load the pre-trained Word2Vec model
    try:
        word2vec_model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"Error: model file '{model_path}' not found.")
        sys.exit(1)
    except Exception as ex:
        print(f"Error loading model file: {ex}")
        sys.exit(1)

    # Instantiate a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=13, weights='distance')
    # Evaluate - Binary
    # print("\n\n================ BINARY CLASSIFICATION EVALUATION ================")
    evaluate_models(
        knn_classifier,
        word2vec_model,
        df_for_eval=df_binary_balanced,
        label_col=label_col_name,
        sentences_col=data_col_name
    )


if __name__ == "__main__":
    main()
