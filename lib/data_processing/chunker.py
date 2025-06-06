#from taxonerd import TaxoNERD
import spacy
import re
import string
from fuzzywuzzy import fuzz

class TextChunker:

    def __init__(self, overlap: int = 100, max_chunk_size: int = 2000,
                 window_size: int = 10, threshold: int = 90):
        """
        Initializes the Chunker with specified overlap and maximum chunk size.
        
        Parameters:
            overlap (int): Number of overlapping characters between chunks.
            max_chunk_size (int): Maximum size of each chunk in characters.
        """
        self.overlap = overlap
        self.max_chunk_size = max_chunk_size
        self.window_size = window_size
        self.threshold = threshold


    def chunk_text_for_cleaning(self, text: str, add_overlap: bool = True) -> list[str]:

        chunks = []

        current_chunk = []

        chunk_size = 0

        paragraphs = re.split("\n\s*\n", text)
        for paragraph in paragraphs:
            para_length = len(paragraph)

            if chunk_size + para_length <= self.max_chunk_size:
                current_chunk.append(paragraph)
                chunk_size += para_length
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                chunk_size = 0

                lines = paragraph.split("\n")
                num_of_lines_not_taken = len(lines)

                while num_of_lines_not_taken > 0:

                    num_lines_to_take = min(int(float(self.max_chunk_size / para_length) * len(lines)), num_of_lines_not_taken)

                    lines_to_add = lines[:num_lines_to_take]
                    joined_lines = "\n".join(lines_to_add)

                    while len(joined_lines) > self.max_chunk_size and num_lines_to_take > 1:
                        num_lines_to_take -= 1
                        lines_to_add = lines[:num_lines_to_take]
                        joined_lines = "\n".join(lines_to_add)


                    if len(joined_lines) <= self.max_chunk_size:
                        current_chunk.append(joined_lines)
                        chunk_size += len(joined_lines)
                    
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    chunk_size = 0
                    num_of_lines_not_taken -= num_lines_to_take        

        if current_chunk:
            chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])

        if not add_overlap:
            return chunks
        
        for i in range(1, len(chunks)):
            chunks[i] = " ".join(chunks[i-1].split()[-self.overlap:]) + " " + chunks[i]

        return chunks
    
    def extract_tokens_and_spans(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """
        Extracts tokens and their character spans from a given text.
        This function identifies non-whitespace runs in the text, strips leading and trailing punctuation,
        and returns a list of tokens along with their character spans in the original text.
        It uses regular expressions to find non-whitespace sequences and captures their start and end positions.

        Args:
            text (str): The input text from which to extract tokens and spans.

        Returns:
            tuple[list[str], list[tuple[int, int]]]: A tuple containing: clean tokens (list of str) and their corresponding spans (list of tuples).
        """
        tokens, spans = [], []
        for match in re.finditer(r'\S+', text):                # find non-whitespace runs
            tok = match.group()
            clean = tok.strip(string.punctuation)          # strip leading/trailing punctuation
            if clean:
                tokens.append(clean)
                spans.append(match.span())                     # (start_char, end_char) of the original tok
        return tokens, spans


    def merge_chunks_fuzzy(self, chunk_a, chunk_b):
        """
        Fuzzy-merge two text chunks by detecting an overlap of up to `self.overlap`
        (scanning `chunk_b` in a sliding window of that many cleaned tokens + a little buffer),
        but splice them together on the ORIGINAL strings so all whitespace/newlines/punctuation
        outside the matched overlap are preserved.
        """

        tokens_a, _ = self.extract_tokens_and_spans(chunk_a)
        tokens_b, spans_b = self.extract_tokens_and_spans(chunk_b)



        """
        Iteratively

        Left chunk: tokens_a <- Start from double the overlap words to create a tail
        Right chunk: tokens_b <- Start from doubles the overlap words to create a head

        Find the ratio, and best index / ratio

        until index is 0 -> regenerate the tail and head with the new index such that the new tail is tail[new_idx:] and head is head[:new_idx + 1]
        If the best ratio is above the threshold, splice the two chunks together at the best index.
        If the best ratio is below the threshold, concatenate the two chunks.
        If the best index is 0, just concatenate the two chunks.
        
        """

        window = int((self.overlap * 1.2) + self.window_size)
        best_ratio, best_i = 0, 0
        while window > 0:
            tail = " ".join(tokens_a[-window:])
            head = " ".join(tokens_b[:window + 1])

            ratio = fuzz.partial_ratio(tail, head)

            if ratio > best_ratio:
                best_ratio = ratio
                best_i = window
            
            window -= 1
        # if we found a good overlap → compute the character-offset in chunk_b

        if best_ratio >= self.threshold:
            cut_pos = spans_b[best_i][0] if best_i < len(spans_b) else len(chunk_b)
            # splice: keep all of chunk_a, then everything in chunk_b from that char-offset onward
            return chunk_a + "\n\n" + chunk_b[cut_pos:]
        else:
            # no confident overlap → just concatenate in full
            return chunk_a + "\n\n" + chunk_b


    def merge_sentences(self, sents):

        assert len(sents) > 0, "No sentences to merge"


        if len(sents) == 1:
            return sents[0]

        merge_to = sents[0]

        for i in range(1, len(sents)):
            merge_to = self.merge_chunks_fuzzy(merge_to, sents[i])
        return merge_to



class SpeciesChunker:

    SPECIES_REGEX = r"([A-Z][a-z]+(?: [a-z]+)\s?(?:[a-zA-Z\[\]\(\)\.\s\,]+)?)"

    def __init__(self, threshold=70):

        self.threshold = threshold
        self.nlp = None
        
    def load(self):
        if self.nlp is not None:
            raise RuntimeError("Chunker is already loaded. Please create a new instance to load again.")
        
        # = TaxoNERD(prefer_gpu=False)
        self.nlp = spacy.load("en_ner_eco_md")#taxonerd.load("en_core_eco_md", exclude=[], threshold=self.threshold)

    def chunk_species(self, text: str) -> list[str]:
        """
        Chunk the text into species records using TaxoNERD.
        Returns a list of dictionaries with species and folders.
        """
        if self.nlp is None:
            print("Chunker is not loaded. Loading Chunker...")
            self.load()

        doc = self.nlp(text)
        species_names = doc.ents

        all_valid_species = "|".join(re.escape(i.text) for i in species_names if re.match(self.SPECIES_REGEX, i.text))

        split_regex = re.compile(rf"^(([0-9]+\.\s?)?(\s+|-)?({all_valid_species})\s*.*)")

        text_splits = text.split("\n")

        chunks = []

        current_chunk = ""

        for line in text_splits:
            if not(line.strip()):
                #print("Skipping empty line")
                continue
            if re.match(split_regex, line):
                #(current_chunk)
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    #print("Chunk added:\n", current_chunk.strip())
                    #print("=" * 50)

                current_chunk = line.strip()
                #print("Matched:", line)
                #print(current_chunk)
            else:
                current_chunk += "\n" + line.strip()
                #print("Not matched:", line)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def group_into_major_chunks(self, chunks: list[str], max_chunk_size: int = 2000) -> list[str]:

        major_chunks = []
        current_chunk = ""

        for chunk in chunks:
            if len(current_chunk) + len(chunk) > max_chunk_size:
                major_chunks.append(current_chunk.strip())
                current_chunk = chunk.strip()
            else:
                current_chunk += "\n\n" + chunk.strip()
        
        if current_chunk:
            major_chunks.append(current_chunk.strip())

        return major_chunks
