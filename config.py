import dspy
from typing import List, Dict, Any
import json
import os

import dotenv

dotenv.load_dotenv()


lm = dspy.LM(dotenv.get_key(dotenv.find_dotenv(), "MODEL_NAME"),
                 api_key = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_KEY"),
                 api_base = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_ENDPOINT"),
                 api_version = dotenv.get_key(dotenv.find_dotenv(), "AZURE_OPENAI_API_VERSION"))

dspy.configure(lm=lm)