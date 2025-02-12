import os
import re
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

star_substitution_regex = re.compile(r"\[\*]")


def read_file_lines(file_path):
    """
    Reads lines from a text file, stripping newlines and ignoring empty lines.
    Returns a list of non-empty lines.
    """
    lines_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    lines_list.append(line)
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied while reading file: '{file_path}'")
        sys.exit(1)
    except Exception as ex:
        print(f"An unexpected error occurred reading '{file_path}': {ex}")
        sys.exit(1)

    return lines_list


def prepare_masked_sentence(sentence):
    """
    Replaces any occurrence of '[*]' with '[MASK]' (as required by BERT-based masked LMs).
    """
    try:
        masked_sentence = star_substitution_regex.sub("[MASK]", sentence)
        return masked_sentence
    except Exception as ex:
        print(f"Error converting asterisks in sentence: {sentence}. Details: {ex}")
        return sentence


def fill_all_masks_with_dictabert(sentences, model_name="dicta-il/dictabert"):
    """
    Uses a DictaBERT (or other BERT-based masked LM) to fill all [MASK] tokens in one pass per sentence.
    """
    try:
        # Load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
    except Exception as ex:
        print(f"Error loading model/tokenizer from '{model_name}': {ex}")
        sys.exit(1)

    results = []

    for sent in sentences:
        # Tokenize the entire sentence (including special tokens)
        encoded = tokenizer.encode_plus(
            sent,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encoded['input_ids']  # shape: [1, seq_len]
        attention_mask = encoded['attention_mask']  # shape: [1, seq_len]

        # Find all positions of the [MASK] token
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            # No [MASK] found, just return the sentence as is
            decoded_sent = tokenizer.decode(input_ids[0])
            results.append({
                'dictabert_sentence': decoded_sent.strip(),
                'dictabert_tokens': []
            })
            continue

        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        # We'll fill each mask position with the best (top-1) prediction
        top_tokens = []
        new_input_ids = input_ids.clone()  # copy

        for pos in mask_positions:
            # Retrieve logits for this [MASK] position: shape [vocab_size]
            mask_logits = logits[0, pos, :]

            # Get top-1 predicted token ID
            top_token_id = torch.argmax(mask_logits)
            new_input_ids[0, pos] = top_token_id

            # Convert ID to string token for logging
            predicted_token_str = tokenizer.convert_ids_to_tokens(int(top_token_id))
            top_tokens.append(predicted_token_str)

        # Decode back to a string (skip [CLS], [SEP], etc.)
        final_sentence = tokenizer.decode(
            new_input_ids[0]
        ).strip()

        results.append({
            'dictabert_sentence': final_sentence,
            'dictabert_tokens': top_tokens
        })

    return results


def write_dictabert_to_output_file(masked_sentences, dictabert_output, output_path):
    """
    Writes DictaBERT fill-mask results to a file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for masked, out in zip(masked_sentences, dictabert_output):
                # line1 = f"original_sentence: {orig}"
                line1 = f"masked_sentence: {masked}"
                line2 = f"dictaBERT_sentence: {out['dictabert_sentence']}"
                line3 = "dictaBERT tokens: " + ",".join(out['dictabert_tokens'])
                f.write(f"{line1}\n{line2}\n{line3}\n")
                # f.write(f"{line1}\n{line2}\n{line3}\n{line4}\n\n")
    except Exception as ex:
        print(f"Error writing DictaBERT results to '{output_path}': {ex}")


def main():
    """
       Main function to read masked sentences, replace '[*]' with '[MASK]', run them
       through DictaBERT, and save the results.
    """
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python knesset_dictabert.py "
            "<path/to/masked_sampled_sents.txt> <path/to/output_dir>\n"
        )
        sys.exit(1)

    masked_file_path = sys.argv[1]
    out_dir_path = sys.argv[2]

    if not os.path.isdir(out_dir_path):
        try:
            os.makedirs(out_dir_path, exist_ok=True)
        except Exception as ex:
            print(f"Error creating directory: {ex}")
            sys.exit(1)

    # Load masked lines
    masked_sentences_raw = read_file_lines(masked_file_path)

    # Replace asterisks (*) with [MASK]
    masked_sentences = [prepare_masked_sentence(ms) for ms in masked_sentences_raw]

    # Use DictaBERT to fill the mask
    model_name = "dicta-il/dictabert"  # model name from Hugging Face
    dictabert_output = fill_all_masks_with_dictabert(masked_sentences, model_name=model_name)

    # Produce the output file in the required format
    output_file_path = os.path.join(out_dir_path, "dictabert_results.txt")
    write_dictabert_to_output_file(masked_sentences, dictabert_output, output_file_path)


if __name__ == "__main__":
    main()
