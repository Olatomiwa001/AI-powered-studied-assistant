import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from transformers import pipeline
import openai
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class Summarizer:
    def __init__(self, model_type="local"):
        """
        Initialize the summarizer with either local models or API-based models
        
        Args:
            model_type (str): "local" for HuggingFace or "openai" for OpenAI API
        """
        self.model_type = model_type
        
        if model_type == "local":
            # Use HuggingFace transformers
            self.summarizer = pipeline("summarization")
        elif model_type == "openai":
            # Use OpenAI API (requires API key set in environment)
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("Model type must be 'local' or 'openai'")
    
    def extractive_summarize(self, text, ratio=0.3):
        """
        Create an extractive summary using frequency-based approach
        
        Args:
            text (str): The input text to summarize
            ratio (float): The proportion of sentences to keep
        
        Returns:
            str: The summarized text
        """
        # Tokenize the text into sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calculate word frequencies
        word_frequencies = FreqDist(filtered_words)
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = 0
                    sentence_scores[i] += word_frequencies[word]
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * ratio))
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)  # Sort by position in original text
        
        # Combine the selected sentences
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    
    def abstractive_summarize(self, text, max_length=150):
        """
        Create an abstractive summary using transformer models or OpenAI
        
        Args:
            text (str): The input text to summarize
            max_length (int): Maximum length of the summary
        
        Returns:
            str: The summarized text
        """
        if self.model_type == "local":
            # Use HuggingFace for summarization
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        
        elif self.model_type == "openai":
            # Use OpenAI for summarization
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": f"Please summarize the following text in about {max_length} words: {text}"}
                ]
            )
            return response.choices[0].message.content.strip()
    
    def summarize(self, text, method="abstractive", **kwargs):
        """
        Main summarization method that chooses between extractive and abstractive
        
        Args:
            text (str): The input text to summarize
            method (str): "extractive" or "abstractive"
            **kwargs: Additional parameters for the specific method
        
        Returns:
            str: The summarized text
        """
        if method == "extractive":
            return self.extractive_summarize(text, **kwargs)
        elif method == "abstractive":
            return self.abstractive_summarize(text, **kwargs)
        else:
            raise ValueError("Method must be 'extractive' or 'abstractive'")