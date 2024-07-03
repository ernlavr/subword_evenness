# Process all 19 languages
# Tokenize, apply BPE, clump single characters, count lengths, UI, variance
# Store the whole table in each language folder

import csv
import os
import re, regex
import sys
import unicodedata
import statistics
from subprocess import call
from polyglot.text import Text
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
import argparse
from pathlib import Path
import codecs
import transformers

import subword_nmt.learn_bpe as lb
import subword_nmt.apply_bpe as ab

# Consts
HF_TOKENIZER = transformers.AutoTokenizer.from_pretrained("res/mbart-cleaned-deploy")
USE_HF_TOKENIZER = True
RE_BAD_CHARS = regex.compile(r"\p{Cc}|\p{Cs}")

def remove_bad_chars(text):
    return RE_BAD_CHARS.sub("", text)

def tokenize(line):
    new_line = remove_bad_chars(line)
    text = Text(new_line)
    try:
        tokens = text.words
    except ValueError:
        tokens = []
    return tokens

def tokenize_hf(line):
    new_line = remove_bad_chars(line)
    tokens = HF_TOKENIZER.tokenize(new_line, add_special_tokens=False)
    return tokens
    

def get_this_dir():
    return os.path.dirname(os.path.abspath(__file__))


def remove_punctuation(text):
    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)).startswith('P'))
    return text.translate(tbl)


def clump_bpe(segmented, substituted):
    segments = []
    with open(segmented, 'r') as f_segments:
        for line in f_segments:
            segments.append(line.strip())

    result = []
    for word in tqdm(segments):
        new_word = word

        start = re.search('^[^@]@@( [^@]@@)+', new_word)
        if start:
            new_start = start.group(0).replace('@@ ', '')
            new_word = new_word.replace(start.group(0), new_start)

        middle = re.findall(' ([^@]@@)(( [^@]@@)+)( [^@]$)?', new_word)
        if len(middle) > 0:
            for item in middle:
                new_item = (item[0] + item[1] + item[3]).replace('@@ ', '')
                new_word = new_word.replace(item[0] + item[1] + item[3], new_item)

        end = re.search('(^| )[^@]@@ [^@]$', new_word)
        if end:
            new_end = end.group(0).replace('@@ ', '')
            new_word = new_word.replace(end.group(0), new_end)

        result.append(new_word)

    # Replace all @@ symbols
    result = [segment.replace('@@ ', '|') for segment in result]

    with open(substituted, 'w') as f:
        for word in result:
            f.write(word + '\n')


def count_lengths_ui(segments):
    segments_lengths = []

    max_seg = -1
    min_seg = 1000
    for seg in segments:
        segments_lengths.append(len(seg))

    for seg_len in segments_lengths:
        if int(seg_len) > max_seg:
            max_seg = int(seg_len)
        if int(seg_len) < min_seg:
            min_seg = int(seg_len)

    index = max_seg - min_seg
    return segments_lengths, index


def count_variance(segments):
    int_seq = []
    for s in segments:
        int_seq.append(int(s))

    if len(int_seq) > 1:
        variance = statistics.variance(int_seq)
    else:
        variance = 'NA'
    return variance


def process_tokens_recursive(tokens, current_word="", final_words=None):
    # Initialize final_words list on the first call
    if final_words is None:
        final_words = []

    # Base case: if tokens list is empty, finalize and return
    if not tokens:
        if current_word:
            final_words.append(current_word)
        return final_words

    # Pop the first token from the list
    token = tokens.pop(0)

    # Check if the token denotes the start of a new word
    if token.startswith('▁'):
        # If we are currently building a word, add it to final words list
        if current_word:
            final_words.append(current_word)
        # Start a new word
        current_word = token[1:]
    else:
        # Continue building the current word
        current_word += token

    # Recursive call with the remaining tokens
    return process_tokens_recursive(tokens, current_word, final_words)


def find_files(start, pattern):
    all_lang_results = []
    for root, dirs, files in os.walk(start):
        for file in files:
            if pattern in file:
                all_lang_results.append(os.path.join(root, file))
    return all_lang_results


def process_chunk(lines):
    processed_lines = []
    for text_line in lines:
        text_line = text_line.strip()
        removed_punct_line = remove_punctuation(text_line)

        # HF tokenizer
        if USE_HF_TOKENIZER:
            tokenized_line = tokenize_hf(removed_punct_line)
            if len(tokenized_line) > 0:
                recursive_tokens = process_tokens_recursive(tokenized_line)
                for word in recursive_tokens:
                    processed_lines.append(word.lower() + '\n')

        # Old BPE stuff
        else:
            tokenized_line = tokenize(removed_punct_line)
            if len(tokenized_line) > 0:
                for token in tokenized_line:
                    if token != '\n' and token != '\r\n' and token != '\r':
                        processed_lines.append(token.lower().strip() + '\n')

    return processed_lines


def preprocess(filename, output_dir, num_workers=4):
    with open(filename, 'r') as f:
        lines = f.readlines()

    chunk_size = len(lines) // num_workers
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks)))
    
    # filename retrieve file extension
    filename_path = Path(filename)
    result_fpath =  os.path.join(output_dir, filename_path.stem + '_tokenized' + filename_path.suffix)
    with open(result_fpath, 'w') as f_result:
        for result in results:
            f_result.writelines(result)
    
    return result_fpath


def learn_bpe(tokenized_file: str, outfile: str, num_symbols=200):
    """ Learn BPE from the tokenized file.
        Args:
            tokenized_file: file with tokenized text
            outfile: file to write the output to
            num_symbols: number of BPE symbols to learn
    """
    lbpe_input = codecs.open(tokenized_file, encoding='utf-8')
    lbpe_output = codecs.open(outfile, 'w', encoding='utf-8')
    lb.learn_bpe(infile=lbpe_input, outfile=lbpe_output, num_symbols=num_symbols)
    
    # Close cuz streamwriter
    lbpe_output.flush()
    lbpe_output.close()
    

def apply_bpe(tokenized_file, outfile, codes, dropout=0):
    """ Apply BPE to the tokenized file.
        Args:
            tokenized_file: file with tokenized text
            outfile: file to write the output to
            codes: file with BPE codes (output of learn_bpe)
            dropout: dropout rate
    """
    tokenized_r = codecs.open(tokenized_file, encoding='utf-8')
    codes_r = codecs.open(codes, encoding='utf-8')
    segmented_w = codecs.open(outfile, 'w', encoding='utf-8')
    bpe = ab.BPE(codes_r)

    # loop over the infile and apply the BPE
    for line in tokenized_r:
        segmented_w.write(bpe.process_line(line, dropout))
    
    # Close the files cuz streamwriter..
    segmented_w.flush()
    segmented_w.close()

def apply_hf_tokenizer(tokenized_file, outfile, model_name='bert-base-uncased'):
    """ Apply Hugging Face tokenizer to the tokenized file.
        Args:
            tokenized_file: file with tokenized text
            outfile: file to write the output to
            model_name: model name of the tokenizer
    """
    # Load the tokenizer

    # Open the input and output files
    with codecs.open(tokenized_file, encoding='utf-8') as tokenized_r, \
            codecs.open(outfile, 'w', encoding='utf-8') as segmented_w:
        
        # Loop over the infile and tokenize each line
        for line in tokenized_r:
            # Tokenize the line
            tokens = HF_TOKENIZER.tokenize(line)
            segment = None

            # if last token is only ▁, remove it
            if tokens[-1] == '▁':
                tokens = tokens[:-1]
            

            # Remove the special token prefix
            for i, token in enumerate(tokens):
                if token.startswith('▁'):
                    tokens[i] = token[1:]

            # If there are subwords, join them with a pipe
            if len(tokens) > 1:
                segment = "|".join(tokens)
            else:
                segment = tokens[0]

            # Convert tokens back to string and write to outfile
            segmented_w.write(f"{segment}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, default='english', help='Language to process')
    parser.add_argument('-f', '--full', action='store_true', help='Process the full file instead of the training split', default=False)
    parser.add_argument('-e', '--extension', type=str, default='full', help='Training split extension of the file. Either "full" or "train"')
    return parser.parse_args()


def main():
    # Setup
    print('BPE-MIN-R')
    
    args = get_args()
    
    input_files = None
    if args.full:
        files = os.listdir('res/')
        files = sorted([f for f in files if args.extension in f])
    else:
        files = ['res/' + args.language + "_" + args.extension + ".txt"]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    for input_file in tqdm(files):
        OUTPUT_DIR = os.path.join("output", input_file[:-4])
        os.makedirs(os.path.join(get_this_dir(), "..", "..", OUTPUT_DIR), exist_ok=True)
        print(f"Processing: {input_file}")
        file_path = Path(input_file)
        file_path_stem = file_path.stem
        file_path_extension = file_path.suffix

        # preprocess, tokenize
        tokenized_file = preprocess(os.path.join("res", input_file), OUTPUT_DIR, multiprocessing.cpu_count())
        
        # learn and apply BPE
        print("Tokenizing")
        file_codes = os.path.join(OUTPUT_DIR, file_path_stem + '_codes' + file_path_extension)
        file_segm = os.path.join(OUTPUT_DIR, file_path_stem + '_segmented' + file_path_extension)
        # learn_bpe(tokenized_file, file_codes, 200)
        if USE_HF_TOKENIZER:
            apply_hf_tokenizer(tokenized_file, file_segm, file_codes)
        else:
            learn_bpe(tokenized_file, file_codes, 200)
            apply_bpe(tokenized_file, file_segm, file_codes)
        
        
        mode = 'bpe-min-r'
        print("Computing results")
        print(input_file)
        segmented_file = file_segm
        print(segmented_file)

        substituted_file = os.path.join(OUTPUT_DIR, file_path_stem + '_substituted' + file_path_extension)
        results_dir = os.path.join("output", 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, file_path_stem + '_results.csv')

        if mode == 'bpe-min-r' and USE_HF_TOKENIZER == False:
            print("Clumping BPE")
            clump_bpe(file_segm, substituted_file)
        else:
            # copy segmented file to substituted file, use system command
            call(['cp', segmented_file, substituted_file])
            

        with open(results_file, 'w', newline='') as sequences_results:
            seq_writer = csv.writer(sequences_results, delimiter='\t')
            seq_writer.writerow(['word_split', 'segments_lengths', 'word_length',
                                'index', 'variance', 'word',
                                'file', 'genre', 'language'])
            lang = input_file.split("_")[0]
            genre = 'na'

            with open(substituted_file, 'r') as f:
                for line in tqdm(f):
                    word_split = line.strip()
                    segments = line.strip().split('|')
                    word = ''.join(segments)
                    segments_lengths, index = count_lengths_ui(segments)
                    variance = count_variance(segments_lengths)

                    seq_writer.writerow([word_split, segments_lengths, len(word),
                                        index, variance, word,
                                        input_file, genre, lang])


if __name__ == '__main__':
    main()
