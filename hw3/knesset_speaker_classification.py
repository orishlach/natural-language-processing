import random
import numpy as np
import json
import pandas as pd
import sys
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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


def build_custom_feature_array(df: pd.DataFrame, sentences_col) -> np.ndarray:
    """
    Iterates over each row in 'df' and builds a custom numeric feature array.
    """
    feature_list = []
    for _, row in df.iterrows():
        feats = extract_custom_features(row, sentences_col)
        feature_list.append(feats)
    return np.array(feature_list)


def extract_custom_features(row: pd.Series, sentences_col) -> list:
    """
    Extracts a small set of custom features:
      1) # of words in the sentence
      2) # of question marks
      3) # of numeric substrings
      4) knesset_number (integer from the corpus)
      5) protocol_number (integer from the corpus)
    """
    sentence_str = row[sentences_col]
    words = sentence_str.split()
    length_words = len(words)
    question_count = sentence_str.count('?')
    # Using regex to find numeric substrings
    numbers = re.findall(r"\d+", sentence_str)
    numbers_count = len(numbers)

    knesset_num = row.get("knesset_number", 0)
    protocol_num = row.get("protocol_number", 0)

    return [length_words, question_count, numbers_count, knesset_num, protocol_num]


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

    freq_speaker_1, freq_speaker_2 = speaker_counts.index[0], speaker_counts.index[1]

    # print(f"Top speakers:\n1) {freq_speaker_1}\n2) {freq_speaker_2}\n")
    df_multi_local = df_corpus.copy()

    def label_multi(spk: str):
        """
          For each speaker name 'spk', check if it unifies with freq_speaker_1 or freq_speaker_2.
          Otherwise, label as 'other'.
        """
        if is_unified_speaker(spk, freq_speaker_1):
            return "first"
        elif is_unified_speaker(spk, freq_speaker_2):
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

    return freq_speaker_1, freq_speaker_2, df_binary_balanced, df_multi_local


def prepare_data_for_multi_class_classification(df_corpus: pd.DataFrame, main_speaker_1: str, main_speaker_2: str,
                                                label_col_name: str):
    """
    Takes the partially-labeled DataFrame (with 'first', 'second', and 'other'),
    then down-samples so that 'first', 'second', and 'other' all have equal size.
    Returns the balanced DataFrame for multi-class classification.
    """
    first_df = df_corpus[df_corpus[label_col_name] == "first"]
    second_df = df_corpus[df_corpus[label_col_name] == "second"]
    other_df = df_corpus[df_corpus[label_col_name] == "other"]

    # print("\nMulti-class classification class sizes (before down-sampling):")
    # print(f"  {main_speaker_1}: {len(first_df)}")
    # print(f"  {main_speaker_2}: {len(second_df)}")
    # print(f"  other: {len(other_df)}")

    min_count = min(len(first_df), len(second_df), len(other_df))
    df_first_down_sample = first_df.sample(n=min_count, random_state=42)
    df_second_down_sample = second_df.sample(n=min_count, random_state=42)
    df_other_down_sample = other_df.sample(n=min_count, random_state=42)

    # Combine and shuffle
    df_balanced = pd.concat([df_first_down_sample, df_second_down_sample, df_other_down_sample])
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    # print("After down-sampling:")
    # print(f"  {main_speaker_1}: {len(df_first_down_sample)}")
    # print(f"  {main_speaker_2}: {len(df_second_down_sample)}")
    # print(f"  other: {len(df_other_down_sample)}")

    return df_balanced


def evaluate_models(
        knn_classifier,
        lr_classifier,
        df_for_eval: pd.DataFrame,
        tfidf_feature_vect,
        label_col,
        sentences_col,
):
    """
    Runs a 5-fold cross-validation using two feature approaches on the given DataFrame:
      1) TF-IDF Features (KNN & LogisticRegression)
      2) Custom Features (KNN & LogisticRegression)
    """
    labels = df_for_eval[label_col].values
    sentences = df_for_eval[sentences_col].values

    # We use StratifiedKFold to preserve class distribution in each fold
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ----------------
    # TF-IDF Features
    # ----------------
    features_tfidf = tfidf_feature_vect.transform(sentences)
    # print("=== TF-IDF Features ===")
    try:
        pred_knn_tfidf = cross_val_predict(knn_classifier, features_tfidf, labels, cv=skfold)
        # print("KNN Results:")
        # print(classification_report(labels, pred_knn_tfidf, digits=3))

        pred_lr_tfidf = cross_val_predict(lr_classifier, features_tfidf, labels, cv=skfold)
        # print("LogisticRegression Results:")
        # print(classification_report(labels, pred_lr_tfidf, digits=3))
    except Exception as e:
        print(f"Error during TF-IDF evaluation: {e}")

    # ----------------
    # Custom Features
    # ----------------
    custom_feature = build_custom_feature_array(df_for_eval, sentences_col)
    # print("=== Custom Features ===")
    try:
        pred_knn_custom = cross_val_predict(knn_classifier, custom_feature, labels, cv=skfold)
        # print("KNN Results:")
        # print(classification_report(labels, pred_knn_custom, digits=3))

        pred_lr_custom = cross_val_predict(lr_classifier, custom_feature, labels, cv=skfold)
        # print("LogisticRegression Results:")
        # print(classification_report(labels, pred_lr_custom, digits=3))
    except Exception as e:
        print(f"Error during custom feature evaluation: {e}")


def classify_new_sentences(
        df_multi: pd.DataFrame,
        tfidf_vect_multi,
        lr_classifier,
        input_sentences_file_path: str,
        output_file_path: str,
        label_col_name: str,
        data_col_name: str
):
    """
       Performs final multi-class classification on new sentences:
       1) Trains a LogisticRegression on the multi-class TF-IDF feature set.
       2) Reads each line from 'input_sentences_file_path' (one sentence per line),
          transforms it into TF-IDF features, and predicts the label.
       3) Writes the predictions to 'output_file_path', one label per line.
    """
    features_train = tfidf_vect_multi.transform(df_multi[data_col_name])
    labels_train = df_multi[label_col_name].values

    try:
        lr_classifier.fit(features_train, labels_train)
    except Exception as ex:
        print(f"Error training final multi-class model: {ex}")
        sys.exit(1)

    # Classify new sentences from text file
    try:
        with open(input_sentences_file_path, "r", encoding="utf-8") as fin, \
                open(output_file_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line_clean = line.strip()
                if not line_clean:
                    fout.write("other\n")
                    continue
                feature_test_line = tfidf_vect_multi.transform([line_clean])
                predicted_label = lr_classifier.predict(feature_test_line)[0]
                fout.write(predicted_label + "\n")

    except FileNotFoundError:
        print(f"Error: The file '{input_sentences_file_path}' does not exist.")
        sys.exit(1)
    except Exception as ex:
        print(f"Error during final classification: {ex}")
        sys.exit(1)


def main():
    """
    The main entry point for the script. Loads the corpus from JSONL. Prepares binary data (two speakers) and
    multi-class data (including 'other'). Fits two TF-IDF vectorizers, one for binary data and one for multi-class.
    Evaluates both scenarios with KNN & LogisticRegression (TF-IDF & custom features). Classifies new sentences in a
    file and writes results.
    """
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python knesset_speaker_classification.py "
            "<path/to/corpus_file.jsonl> <path/to/sentences_file.txt> <path/to/output_dir>\n"
        )
        sys.exit(1)

    corpus_path = sys.argv[1]
    sentences_path = sys.argv[2]
    out_dir_path = sys.argv[3]

    if not os.path.isdir(out_dir_path):
        # print(f"Output directory '{out_dir_path}' does not exist. Creating...")
        try:
            os.makedirs(out_dir_path, exist_ok=True)
        except Exception as ex:
            print(f"Error creating directory: {ex}")
            sys.exit(1)

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
                    "protocol_name": record_json.get("protocol_name", ""),
                    "knesset_number": record_json.get("knesset_number", 0),
                    "protocol_type": record_json.get("protocol_type", ""),
                    "protocol_number": record_json.get("protocol_number", 0),
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

    # Prepare binary & multi data
    main_speaker_1, main_speaker_2, df_binary_balanced, df_multi = prepare_data_for_binary_classification(df_corpus,
                                                                                                          label_col_name)
    df_multi_balanced = prepare_data_for_multi_class_classification(df_multi, main_speaker_1, main_speaker_2,
                                                                    label_col_name)

    # Fit separate vectorizers for binary vs. multi
    tfidf_vect_bin = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, min_df=2)
    try:
        tfidf_vect_bin.fit(df_binary_balanced[data_col_name])
    except Exception as ex:
        print(f"Error fitting TF-IDF on binary data: {ex}")
        sys.exit(1)

    tfidf_vect_multi = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, min_df=2)
    try:
        tfidf_vect_multi.fit(df_multi_balanced[data_col_name])
    except Exception as ex:
        print(f"Error fitting TF-IDF on multi data: {ex}")
        sys.exit(1)

    # Instantiate default classifiers
    knn_classifier = KNeighborsClassifier(n_neighbors=13, weights='distance')
    lr_classifier = LogisticRegression(max_iter=4000)
    # Evaluate - Binary
    # print("\n\n================ BINARY CLASSIFICATION EVALUATION ================")
    evaluate_models(
        knn_classifier,
        lr_classifier,
        df_for_eval=df_binary_balanced,
        tfidf_feature_vect=tfidf_vect_bin,
        label_col=label_col_name,
        sentences_col=data_col_name
    )

    # Evaluate - Multi-class
    # print("\n\n================ MULTI-CLASS CLASSIFICATION EVALUATION ================")
    evaluate_models(
        knn_classifier,
        lr_classifier,
        df_for_eval=df_multi_balanced,
        tfidf_feature_vect=tfidf_vect_multi,
        label_col=label_col_name,
        sentences_col=data_col_name
    )

    # Final classification of new sentences (multi)
    output_file = os.path.join(out_dir_path, "classification_results.txt")
    classify_new_sentences(df_multi_balanced, tfidf_vect_multi, lr_classifier, sentences_path, output_file, label_col_name,
                           data_col_name)


if __name__ == "__main__":
    main()
