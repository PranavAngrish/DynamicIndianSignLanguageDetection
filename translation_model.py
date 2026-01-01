"""
English to Indian Language Translation Module
Using Free Google Translate API - No authentication required!
"""

import requests
from typing import Dict

class IndianLanguageTranslator:
    """
    Free Google Translate API translator
    No API key needed - completely free!
    """
    
    LANGUAGE_CODES = {
        "hi": "Hindi (हिंदी)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "bn": "Bengali (বাংলা)",
        "gu": "Gujarati (ગુજરાતી)",
        "mr": "Marathi (मराठी)",
    }

    TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
    
    def __init__(self):
        """Initialize translator"""
        print("[Translator] ✓ Free Google Translate API initialized!")
        print(f"[Translator] ✓ Supported languages: {len(self.LANGUAGE_CODES)}")
        print("[Translator] ✓ No API key required - completely free!")
        print()
    
    def translate(self, text: str, target_language: str, **kwargs) -> str:
        """
        Translate English text to target Indian language using Google Translate API
        
        Args:
            text: English sentence to translate
            target_language: Language code (hi, pa, bn, gu, mr)
            **kwargs: Ignored (kept for compatibility)
            
        Returns:
            Translated text in target language
        """

        if not text or not text.strip():
            return ""

        if target_language not in self.LANGUAGE_CODES:
            target_language = "hi" 
        
        try:

            params = {
                'client': 'gtx',
                'sl': 'en', 
                'tl': target_language,
                'dt': 't', 
                'q': text
            }

            response = requests.get(
                self.TRANSLATE_URL,
                params=params,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()

                translated = result[0][0][0]
                return translated
            else:
                return f"Translation error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def translate_all_languages(self, text: str) -> Dict[str, str]:
        """
        Translate to all supported Indian languages
        
        Args:
            text: English sentence
            
        Returns:
            Dictionary with all translations
        """
        translations = {"English": text}
        
        for lang_code, lang_name in self.LANGUAGE_CODES.items():
            translation = self.translate(text, lang_code)
            translations[lang_name] = translation
        
        return translations
    
    def get_all_languages(self):
        """Get all supported languages"""
        return self.LANGUAGE_CODES
    
    def batch_translate(self, texts: list, target_language: str) -> list:
        """
        Translate multiple texts
        
        Args:
            texts: List of English sentences
            target_language: Language code
            
        Returns:
            List of translated texts
        """
        return [self.translate(text, target_language) for text in texts]

translator = IndianLanguageTranslator()

def translate_text(sentence: str, language: str) -> str:
    """
    Wrapper function used by Flask backend
    
    Args:
        sentence: English sentence to translate
        language: Target language code (hi, pa, bn, gu, mr)
        
    Returns:
        Translated text
    """
    return translator.translate(sentence, language)

if __name__ == '__main__':
    print("=" * 70)
    print("TESTING FREE GOOGLE TRANSLATE API")
    print("=" * 70)
    print()
    
    test_sentences = [
        "I go to school on Monday.",
        "I have a black shirt.",
        "Thank you.",
        "I take my dog to the park.",
        "Father goes to the hospital.",
    ]
 
    for sentence in test_sentences:
        print(f"English: {sentence}")
        for lang_code, lang_name in translator.LANGUAGE_CODES.items():
            translation = translate_text(sentence, lang_code)
            print(f"  {lang_name}: {translation}")
        print()
    
    print("=" * 70)
    print("Translation service ready!")
    print("=" * 70)
