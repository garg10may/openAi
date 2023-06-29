import openai
import os
from dotenv import find_dotenv, load_dotenv

key = os.getenv('key')

def load_open_ai_key():
  _ = load_dotenv(find_dotenv())
  openai.api_key = key