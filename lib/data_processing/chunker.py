# Python libraries
#from taxonerd import TaxoNERD
import spacy
import re
import string
from fuzzywuzzy import fuzz

# Get the logger for this file
from lib.utils import get_logger
logger = get_logger(__name__)

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
        """Chunks the input text into smaller segments for cleaning.

        Args:
            text (str): The input text to be chunked.
            add_overlap (bool, optional): Whether to add overlapping text between chunks. Defaults to True.

        Returns:
            list[str]: A list of text chunks.
        """

        chunks = []

        current_chunk = []

        chunk_size = 0

        paragraphs = re.split("\n\s*", text)
        for paragraph in paragraphs:
            para_length = len(paragraph)

            if chunk_size + para_length <= self.max_chunk_size:
                current_chunk.append(paragraph)
                chunk_size += para_length
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = []
                chunk_size = 0
    

        if current_chunk:
            chunks.append("\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])

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


    def merge_chunks_fuzzy(self, chunk_a: str, chunk_b: str) -> str:
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


    def merge_sentences(self, sents: list[str]) -> str:
        """
        Merges a list of sentences into a single chunk using fuzzy merging.

        Args:
            sents (list[str]): The list of sentences to merge.

        Returns:
            str: Merged sentence
        """

        assert len(sents) > 0, "No sentences to merge"


        if len(sents) == 1:
            return sents[0]

        merge_to = sents[0]

        for i in range(1, len(sents)):
            merge_to = self.merge_chunks_fuzzy(merge_to, sents[i])
        return merge_to



class SpeciesChunker:

    SPECIES_REGEX = r"\b([A-Z][a-z]+(?: [a-z]+)\s?(?:[a-zA-Z\[\]\(\)\.\s\,]+)?)\b"


    def __init__(self, threshold: int = 70):

        self.threshold = threshold
        self.nlp = None
        
    
    def load(self):
        """
        Load the TaxoNERD model for species chunking.

        Raises:
            RuntimeError: If the chunker is already loaded.
        """
        if self.nlp is not None:
            raise RuntimeError("Chunker is already loaded. Please create a new instance to load again.")
        
        self.nlp = spacy.load("en_ner_eco_md")


    def chunk_species(self, text: str) -> list[str]:
        """
         Chunk the text into species records using TaxoNERD.
        Returns a list of dictionaries with species and folders.

        Args:
            text (str): The text to chunk.

        Returns:
            list[str]: A list of chunked species records.
        """
        if self.nlp is None:
            print("Chunker is not loaded. Loading Chunker...")
            self.load()

        doc = self.nlp(text)
        species_names = doc.ents

        all_valid_species = "|".join(re.escape(i.text) for i in species_names if re.match(self.SPECIES_REGEX, i.text))

        split_regex = re.compile(rf"^(([0-9]+\.\s?)?(\s+|-)?({all_valid_species}))")

        text_splits = text.split("\n")

        chunks = []

        current_chunk = ""

        for line in text_splits:
            if not(line.strip()):

                continue
            if re.match(split_regex, line):

                if current_chunk:
                    chunks.append(current_chunk.strip())


                current_chunk = line.strip()

            else:
                current_chunk += "\n" + line.strip()

        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    

    def group_into_major_chunks(self, chunks: list[str], max_chunk_size: int = 1000) -> list[str]:
        """
        Groups smaller chunks into larger major chunks without exceeding the specified maximum chunk size.

        Args:
            chunks (list[str]): The list of chunks to group.
            max_chunk_size (int, optional): The maximum size of each major chunk. Defaults to 1000.

        Returns:
            list[str]: A list of major chunks.
        """
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
