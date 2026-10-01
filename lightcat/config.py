"""Configuration constants for LightfootCatalogue pipeline."""

# Model Configuration
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# Image Processing
IMAGE_EXT = ["jpeg", "png", "jpg"]
IGNORE_FILE = [".ipynb_checkpoints"]
CROPPED_DIR_NAME = "cropped_images"
EXTRACTED_TEXT="extracted_text.txt"
ALLOWED_EXT=["jpeg", "png", "jpg", "pdf"]
# Paths
DEFAULT_SAVE_PATH = "./parsed_json"

# Processing Constants
SAVE_JSON_INTERVAL = 10  # Save JSON after processing this many blocks
SAVE_TEXT_INTERVAL = 3   # Save extracted text after this many images
TRANSCRIPTION_BATCH_SIZE = 1  # Batch size for transcription operations

# Performance
DEFAULT_MAX_CHUNK_SIZE = 3000
DEFAULT_BATCH_SIZE = 1

# Retry Configuration
DEFAULT_TIMEOUT_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 2

# OCR Configuration
DEFAULT_TESSERACT_DPI = 600
DEFAULT_PADDING = 100.0
DEFAULT_RESIZE_FACTOR = 0.4
DEFAULT_REMOVE_AREA_PERC = 0.01
DEFAULT_MIDDLE_MARGIN_PERC = 0.20


# Legacy family names

LEGACY_FAMILY_NAMES = [
            "COMPOSITAE", "GRAMINEAE", "LEGUMINOSAE", "PALMAE",
            "UMBELLIFERAE", "CRUCIFERAE", "LABIATAE", "GUTTIFERAE",
            "PAPILIONACEAE", "MIMOSACEAE", "CAESALPINIACEAE"
        ]