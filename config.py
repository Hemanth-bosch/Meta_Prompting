import dspy
from typing import List, Dict, Any
import json
import os

import dotenv

dotenv.load_dotenv()

# if model name is gpt-5, then needs  customized handling
if "gpt-5" in dotenv.get_key(dotenv.find_dotenv(), "MODEL_NAME"):
    lm = dspy.LM(dotenv.get_key(dotenv.find_dotenv(), "MODEL_NAME"),
                 api_key = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_KEY"),
                 api_base = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_ENDPOINT"),
                 api_version = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_VERSION"), temperature=1.0, max_tokens=20000)
else:
    lm = dspy.LM(dotenv.get_key(dotenv.find_dotenv(), "MODEL_NAME"),
                 api_key = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_KEY"),
                 api_base = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_ENDPOINT"),
                 api_version = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_VERSION"))

reflectin_lm = dspy.LM(dotenv.get_key(dotenv.find_dotenv(), "REFLECTION_LM_NAME"),
                    api_key = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_KEY"),
                    api_base = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_ENDPOINT"),
                    api_version = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_VERSION"), temperature=1.0, max_tokens=20000)

dspy.configure(lm=lm)