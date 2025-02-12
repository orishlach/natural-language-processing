import math
import os
import random
import sys
import pandas as pd
from collections import Counter

# Configurations
COLLOCATIONS_RESULTS_FILE = "knesset_collocations.txt"
SRC_SENTENCES_FILE = "original_sampled_sents.txt"
MASKED_SENTENCES_FILE = "masked_sampled_sents.txt"
SAMPLED_RESULTS_FILE = "sampled_sents_results.txt"
PERPLEXITY_RESULT_FILE = "perplexity_result.txt"

NUM_SENTENCES_TO_SAMPLE = 10
MASK_PERCENT = 0.1
FREQ_THRESH = 5
TOP_COLL = 10

INIT_TOKENS = ["s_0", "s_1"]


#####################################
# Utility Functions

def tokenize(sentence):
    return sentence.split()


def detokenize(tok_list):
    return " ".join(tok_list)


def get_all_ngrams(tok_seq, n):
    return [tuple(tok_seq[i:i + n]) for i in range(len(tok_seq) - n + 1)]


#####################################


# Trigram Language Model Class
class Trigram_LM:
    """
      A simple trigram language model that uses Laplace smoothing and
      fixed interpolation with bigram and unigram probabilities.
    """

    def __init__(self, sents):
        # sentences: list of strings
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        # Interpolation lambdas
        self.lambda1 = 0.4
        self.lambda2 = 0.3
        self.lambda3 = 0.3

        tmp_voc = set()
        # Build counts
        for sent in sents:
            tokens = INIT_TOKENS + tokenize(sent)
            for t in tokens:
                self.unigram_counts[t] += 1
                tmp_voc.add(t)
            for i in range(len(tokens) - 1):
                self.bigram_counts[(tokens[i], tokens[i + 1])] += 1
            for i in range(len(tokens) - 2):
                self.trigram_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

        self.total_unigrams = sum(self.unigram_counts.values())
        self.V = len(tmp_voc)
        # Create a separate vocabulary for generation that excludes the init tokens
        self.voc = tmp_voc.difference(INIT_TOKENS)

    def _prob_unigram(self, w):
        # Add-one smoothing for unigram
        return (self.unigram_counts.get(w, 0) + 1) / (self.total_unigrams + self.V)

    def _prob_bigram(self, w1, w2):
        # Add-one smoothing for bigram: P(w2|w1)
        uni_count = self.unigram_counts.get(w1, 0)
        bi_count = self.bigram_counts.get((w1, w2), 0)
        return (bi_count + 1) / (uni_count + self.V)

    def _prob_trigram(self, w1, w2, w3):
        # Add-one smoothing for trigram: P(w3|w1,w2)
        bi_count = self.bigram_counts.get((w1, w2), 0)
        tri_count = self.trigram_counts.get((w1, w2, w3), 0)
        return (tri_count + 1) / (bi_count + self.V)

    def _calc_prob_trigram(self, w1, w2, w3):
        """
            Compute the probability of w3 given w1,w2 using trigram, bigram, and unigram probabilities.
            Uses Laplace smoothing and interpolation with lambda1, lambda2, lambda3.
        """
        p_trigram = self._prob_trigram(w1, w2, w3)
        p_bigram = self._prob_bigram(w2, w3)
        p_unigram = self._prob_unigram(w3)
        p = self.lambda1 * p_trigram + self.lambda2 * p_bigram + self.lambda3 * p_unigram
        return p

    def calculate_prob_of_sentence(self, sentence):
        """
            Given a sentence (string), calculate its log probability under the language model.
            The sentence is first tokenized, and INIT_TOKENS are prepended.
        """
        tokens = INIT_TOKENS + tokenize(sentence)
        lp = 0.0
        for i in range(len(tokens)):
            if i >= 2:
                w1, w2, w3 = tokens[i - 2], tokens[i - 1], tokens[i]
                p = self._calc_prob_trigram(w1, w2, w3)
                if p <= 0:
                    # If probability is zero, handle gracefully
                    return float('-inf')
                lp += math.log(p)
            else:
                # Handle first two tokens similarly
                # i=0: P(w0|s_0,s_1)
                # i=1: P(w1|s_1,w0)
                if i == 0:
                    w3 = tokens[0]
                    w1, w2 = INIT_TOKENS[0], INIT_TOKENS[1]
                    p = self._calc_prob_trigram(w1, w2, w3)
                    if p <= 0:
                        return float('-inf')
                    lp += math.log(p)
                elif i == 1:
                    w3 = tokens[1]
                    w1, w2 = INIT_TOKENS[1], tokens[0]
                    p = self._calc_prob_trigram(w1, w2, w3)
                    if p <= 0:
                        return float('-inf')
                    lp += math.log(p)
        return lp

    def generate_next_token(self, sentence):
        """
            Given a partial sentence, choose the next token that maximizes the trigram probability.
            Returns the chosen token and its log probability.
        """
        tokens = INIT_TOKENS + tokenize(sentence)

        w1 = tokens[-2]
        w2 = tokens[-1]

        max_p = -1.0
        max_token = None
        for cand in self.voc:
            cur_p = self._calc_prob_trigram(w1, w2, cand)
            if cur_p > max_p:
                max_p = cur_p
                max_token = cand
        if max_p > 0:
            return max_token, math.log(max_p)
        else:
            return None, float('-inf')


def get_k_n_t_collocations(k, n, t, corpus, type="frequency"):
    """
        Extract top-k n-gram collocations from df['sentence_text'] by grouping texts by 'protocol_name'.
        metric can be 'frequency' or 'tfidf'.
        n - n-gram size
        k - top k results
        t - frequency threshold

        Returns a list of top k n-grams.
    """
    # Group by protocol_name to consider each protocol_name as one document
    try:
        grouped = corpus.groupby('protocol_name')['sentence_text'].apply(lambda x: ' '.join(x)).reset_index()
    except Exception as e:
        print(f"Error grouping documents by 'protocol_name': {e}")
        return []
    # Now each row in 'grouped' is a single document.
    docs = grouped['sentence_text'].tolist()

    document_ngram_counts = []
    all_ngrams = Counter()
    doc_frequency = Counter()

    for doc in docs:
        tokens = tokenize(doc)
        ngrams = get_all_ngrams(tokens, n)
        doc_ngram_cnt = Counter(ngrams)
        document_ngram_counts.append(doc_ngram_cnt)

        # Update global counts
        for ng in doc_ngram_cnt:
            all_ngrams[ng] += doc_ngram_cnt[
                ng]  # Add the frequency of this ngram in this document to the global frequency
            doc_frequency[ng] += 1  # For doc frequency, each ngram appears at least once in this doc => increment by 1

    number_of_docs = len(docs)
    # Filter by threshold
    filtered_ngrams = [ngrams for ngrams, count in all_ngrams.items() if count >= t]
    if type == "frequency":
        # Sort by total frequency across all docs
        sorted_ngrams = sorted(filtered_ngrams, key=lambda x: all_ngrams[x], reverse=True)[:k]
        return sorted_ngrams
    elif type.lower() == "tfidf":
        # Compute TF-IDF scores
        scores = {}
        for ng in filtered_ngrams:
            try:
                idf = math.log(number_of_docs / doc_frequency[ng])
            except ValueError:
                # If doc_freq[ng] is 0 or leads to negative value in log
                continue
            # We'll collect tf-idf scores per document and then average them.
            tfidf_vals = []
            for doc_ngram_cnt in document_ngram_counts:
                if ng in doc_ngram_cnt:
                    freq = doc_ngram_cnt[ng]
                    total_in_d = sum(doc_ngram_cnt.values())
                    tf = freq / total_in_d
                    tfidf_vals.append(tf * idf)
            if tfidf_vals:
                # Use the average TF-IDF score instead of the sum
                scores[ng] = sum(tfidf_vals) / number_of_docs
        if scores:
            sorted_ngrams = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
            return sorted_ngrams
        else:
            print("No valid TF-IDF scores computed.")
            return []
    else:
        print(f"Unknown metric: {type}")
        return []


def mask_tokens_in_sentences(sentences, x):
    """
       Mask a percentage x of tokens in each sentence with the placeholder '[*]'.
    """
    masked = []
    for sent in sentences:
        tokens = tokenize(sent)
        if not tokens:
            masked.append(sent)
            continue
        num_to_mask = max(1, round(len(tokens) * x))
        if num_to_mask > len(tokens):
            num_to_mask = len(tokens)
        try:
            indices_to_mask = random.sample(range(len(tokens)), num_to_mask)
        except ValueError:
            # If we cannot sample due to some error, just skip masking
            indices_to_mask = []
        for ix in indices_to_mask:
            tokens[ix] = "[*]"
        masked.append(detokenize(tokens))
    return masked


def fill_masked_tokens(sentence, lm):
    """
      Given a sentence with masked tokens '[*]' and a language model,
      fill each masked token by calling the LM's generate_next_token function.
    """
    tokens = tokenize(sentence)
    original_toks = tokens[:]
    for i, tok in enumerate(tokens):
        if tok == "[*]":
            # Construct a partial sentence up to the token before the masked one
            partial_sentence = ' '.join(original_toks[:i])
            # Use the language model's generate_next_token function
            best_t, _ = lm.generate_next_token(partial_sentence)
            if best_t is None:
                # If no token found, use a placeholder token
                best_t = "<UNK>"
            original_toks[i] = best_t
    return original_toks


def compute_sentence_perplexity_for_masked(filled_sentence, masked_sentence, lm):
    """
        Compute the perplexity for the masked tokens in a sentence.
        Only considers masked tokens, and returns None if no masked tokens present.
    """
    # Add init tokens internally for probability calculations
    filled_tokens = INIT_TOKENS + tokenize(filled_sentence)
    masked_tokens = tokenize(masked_sentence)

    log_probs = []
    # Find masked token positions
    for i, tok in enumerate(masked_tokens):
        if tok == "[*]":
            idx = i + 2  # offset by 2 due to INIT_TOKENS
            w1 = filled_tokens[idx - 2]
            w2 = filled_tokens[idx - 1]
            w3 = filled_tokens[idx]
            p = lm._calc_prob_trigram(w1, w2, w3)
            if p > 0:
                log_probs.append(math.log(p))

    if len(log_probs) == 0:
        return None  # No masked tokens in this sentence

    # Perplexity for this sentence considering only masked tokens
    # Perplexity = exp(- (1/M) * sum(log p))
    M = len(log_probs)
    perplexity = math.exp(-sum(log_probs) / M)
    return perplexity


def print_coll_to_file(committee_df, plenary_df, output_dir):
    """
     Compute collocations for committee and plenary dataframes and write results to a file.
    """
    output_file = os.path.join(output_dir, COLLOCATIONS_RESULTS_FILE)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for n, label in zip([2, 3, 4], ["Two", "Three", "Four"]):
                f.write(f"{label}-gram collocations:\n")
                # Frequency
                f.write("Frequency:\n")
                f.write("Committee corpus:\n")
                freq_committee_coll = get_k_n_t_collocations(TOP_COLL, n, FREQ_THRESH, committee_df, "frequency")
                for coll in freq_committee_coll:
                    f.write(" ".join(coll) + "\n")
                f.write("\nPlenary corpus:\n")
                freq_plenary_coll = get_k_n_t_collocations(TOP_COLL, n, FREQ_THRESH, plenary_df, "frequency")
                for coll in freq_plenary_coll:
                    f.write(" ".join(coll) + "\n")
                f.write("\n")

                # TF-IDF
                f.write("TF-IDF:\n")
                f.write("Committee corpus:\n")
                tfidf_committee_coll = get_k_n_t_collocations(TOP_COLL, n, FREQ_THRESH, committee_df, "tfidf")
                for coll in tfidf_committee_coll:
                    f.write(" ".join(coll) + "\n")
                f.write("\nPlenary corpus:\n")
                tfidf_plenary_coll = get_k_n_t_collocations(TOP_COLL, n, FREQ_THRESH, plenary_df, "tfidf")
                for coll in tfidf_plenary_coll:
                    f.write(" ".join(coll) + "\n")
                f.write("\n")

    except Exception as e:
        print(f"Error writing collocations to file: {e}")


def mask_predict_write_to_files(committee_df, committee_model, plenary_model, output_dir):
    """
       Sample sentences from the committee_df, mask tokens, and fill them using the plenary_model.
       Write the results (original, masked, filled) to the appropriate output files.
    """
    # Filter to committee sentences that have at least 5 tokens
    committee_sentences = committee_df['sentence_text'].tolist()
    committee_sentences = [s for s in committee_sentences if len(tokenize(s)) >= 5]
    if len(committee_sentences) < NUM_SENTENCES_TO_SAMPLE:
        print("Not enough committee sentences to sample from.")
        return [], []

    try:
        sampled_sentences = random.sample(committee_sentences, NUM_SENTENCES_TO_SAMPLE)
    except ValueError as e:
        print(f"Error sampling sentences: {e}")
        return [], []

    masked_sentences = mask_tokens_in_sentences(sampled_sentences, MASK_PERCENT)

    output_file = os.path.join(output_dir, SRC_SENTENCES_FILE)
    # Write original and masked sentences to files
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for s in sampled_sentences:
                f.write(s + "\n")
    except Exception as e:
        print(f"Error writing original sampled sentences: {e}")

    output_file = os.path.join(output_dir, MASKED_SENTENCES_FILE)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for s in masked_sentences:
                f.write(s + "\n")
    except Exception as e:
        print(f"Error writing masked sampled sentences: {e}")

    # Fill masked tokens using plenary model
    plenary_completed_sentences = []
    plenary_completed_tokens = []
    for cur_ms in masked_sentences:
        filled_tokens = fill_masked_tokens(cur_ms, plenary_model)
        plenary_completed_sentences.append(detokenize(filled_tokens))
        orig_tokens = tokenize(cur_ms)
        replaced_tokens = [filled_tokens[i] for i, t in enumerate(orig_tokens) if t == "[*]"]
        plenary_completed_tokens.append(replaced_tokens)

    output_file = os.path.join(output_dir, SAMPLED_RESULTS_FILE)
    # Compute probabilities and write results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(NUM_SENTENCES_TO_SAMPLE):
                original = sampled_sentences[i]
                masked = masked_sentences[i]
                plen_filled_sen = plenary_completed_sentences[i]
                plen_tokens = plenary_completed_tokens[i]

                plen_logprob_plen = plenary_model.calculate_prob_of_sentence(plen_filled_sen)
                plen_logprob_comm = committee_model.calculate_prob_of_sentence(plen_filled_sen)

                f.write(f"original_sentence: {original}\n")
                f.write(f"masked_sentence: {masked}\n")
                f.write(f"plenary_sentence: {plen_filled_sen}\n")
                f.write(f"plenary_tokens: {','.join(plen_tokens)}\n")
                f.write(f"probability of plenary sentence in plenary corpus: {plen_logprob_plen:.2f}\n")
                f.write(f"probability of plenary sentence in committee corpus: {plen_logprob_comm:.2f}\n")
    except Exception as e:
        print(f"Error writing sampled sentences results: {e}")
    return masked_sentences, plenary_completed_sentences


def compute_avg_perplexity_and_write_to_file(masked_sentences, plenary_completed_sentences, plenary_model, output_dir):
    """
       Compute average perplexity over masked tokens of sampled sentences and write to file.
    """
    perplexities = []
    for i in range(NUM_SENTENCES_TO_SAMPLE):
        masked = masked_sentences[i]
        filled = plenary_completed_sentences[i]

        sent_pp = compute_sentence_perplexity_for_masked(filled, masked, plenary_model)
        if sent_pp is not None:
            perplexities.append(sent_pp)

    # Average perplexity across all sampled sentences
    if len(perplexities) > 0:
        avg_perplexity = sum(perplexities) / len(perplexities)
    else:
        avg_perplexity = float('nan')

    output_file = os.path.join(output_dir, PERPLEXITY_RESULT_FILE)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{avg_perplexity:.2f}\n")
    except Exception as e:
        print(f"Error writing perplexity result: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python knesset_language_models.py <path/to/corpus_file_name.jsonl> <path/to/output_dir>")
        sys.exit(1)

    input_data_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_json(input_data_file, lines=True)
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        sys.exit(1)

    # Filter by protocol_type
    committee_df = df[df['protocol_type'] == 'committee']
    plenary_df = df[df['protocol_type'] == 'plenary']

    if committee_df.empty:
        print("No committee data found. Exiting.")
        sys.exit(0)

    if plenary_df.empty:
        print("No plenary data found. Exiting.")
        sys.exit(0)

    # Build Models
    try:
        committee_sentences = committee_df['sentence_text'].tolist()
        plenary_sentences = plenary_df['sentence_text'].tolist()
        committee_model = Trigram_LM(committee_sentences)
        plenary_model = Trigram_LM(plenary_sentences)
    except Exception as e:
        print(f"Error building language models: {e}")
        sys.exit(1)

    # Print collocations
    print_coll_to_file(committee_df, plenary_df, output_dir)

    # Mask and predict
    masked_sentences, plenary_completed_sentences = mask_predict_write_to_files(committee_df, committee_model,
                                                                                plenary_model, output_dir)

    # Compute perplexity
    compute_avg_perplexity_and_write_to_file(masked_sentences, plenary_completed_sentences, plenary_model, output_dir)


if __name__ == "__main__":
    main()
