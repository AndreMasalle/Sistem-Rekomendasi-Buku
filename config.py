import os
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("KEY")
DATASET = os.getenv("BOOK_CSV_PATH")
TAG = os.getenv("TAG")
NA = os.getenv("NA")
