
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_CANDIDATES = [
    "gemini-2.5-pro",
    "gemini-1.5-pro"
]

def build_sentence(isl_sentence: str) -> str:
    """
    Converts ISL sentence to grammatical English using Gemini.

    Input:  "I TODAY BLUE SHIRT SCHOOL"
    Output: "I am wearing a blue shirt to school today."
    """

    if not isl_sentence or not isl_sentence.strip():
        return ""

    prompt = f"""
Convert this Indian Sign Language (ISL) sentence into proper,
grammatical English. Do not explain. Only return the sentence.

ISL: {isl_sentence}

English:
""".strip()

    response = None

    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            break
        except Exception as e:
            print(f"[Gemini] Failed with {model_name}: {e}")

    if response and hasattr(response, "text"):
        return response.text.strip()

    return isl_sentence.capitalize()