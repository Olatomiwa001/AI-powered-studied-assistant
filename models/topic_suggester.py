# models/topic_suggester.py

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import openai
import os
from collections import Counter

class TopicSuggester:
    def __init__(self, model_type="local"):
        """
        Initialize the topic suggester
        
        Args:
            model_type (str): "local" for local processing or "openai" for OpenAI API
        """
        self.model_type = model_type
        
        # Download necessary NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        if model_type == "openai":
            # Use OpenAI API
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def extract_topics(self, text, num_topics=5):
        """
        Extract main topics from the text
        
        Args:
            text (str): The input text
            num_topics (int): Number of topics to extract
        
        Returns:
            list: List of topics
        """
        if self.model_type == "local":
            return self._extract_topics_local(text, num_topics)
        elif self.model_type == "openai":
            return self._extract_topics_openai(text, num_topics)
    
    def _extract_topics_local(self, text, num_topics=5):
        """Extract topics using local NLP techniques"""
        
        # Clean text and tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(text)
        
        # Remove stopwords and short words
        words = [word for word in words if word not in self.stop_words and len(word) > 3]
        
        # Approach 1: Frequency-based single words
        word_freq = FreqDist(words)
        common_words = [word for word, _ in word_freq.most_common(num_topics)]
        
        # Approach 2: Bigrams (word pairs)
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(words)
        finder.apply_freq_filter(3)  # Filter out less frequent bigrams
        bigrams = finder.nbest(bigram_measures.pmi, num_topics)
        bigram_topics = [' '.join(bigram) for bigram in bigrams]
        
        # Combine both approaches
        combined_topics = []
        for i in range(min(num_topics, len(common_words) + len(bigram_topics))):
            if i % 2 == 0 and common_words:
                combined_topics.append(common_words.pop(0))
            elif bigram_topics:
                combined_topics.append(bigram_topics.pop(0))
            elif common_words:
                combined_topics.append(common_words.pop(0))
        
        # If we still need more topics, use TF-IDF on sentences
        if len(combined_topics) < num_topics:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) >= 3:  # Need at least a few sentences for meaningful TF-IDF
                vectorizer = TfidfVectorizer(max_features=num_topics*2, 
                                          stop_words='english', 
                                          ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Sum TF-IDF scores across all sentences
                tfidf_sums = tfidf_matrix.sum(axis=0).A1
                
                # Get top features
                top_indices = tfidf_sums.argsort()[-num_topics*2:][::-1]
                tfidf_topics = [feature_names[i] for i in top_indices]
                
                # Add unique topics
                for topic in tfidf_topics:
                    if topic not in combined_topics and len(combined_topics) < num_topics:
                        combined_topics.append(topic)
        
        # Ensure we have exactly the requested number of topics
        combined_topics = combined_topics[:num_topics]
        
        # Capitalize for better presentation
        formatted_topics = [topic.title() for topic in combined_topics]
        
        return formatted_topics
    
    def _extract_topics_openai(self, text, num_topics=5):
        """Extract topics using OpenAI API"""
        
        # Create a prompt for OpenAI
        prompt = f"""
        Extract {num_topics} main topics or concepts from the following text. 
        For each topic, provide:
        1. The topic name (1-3 words)
        2. A brief explanation why this topic is important in the context of the text
        
        Text:
        {text[:4000]}  # Limit text length for API
        
        Format as JSON with 'topics' as a list of objects with 'name' and 'explanation' fields.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant that extracts key topics from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to local processing if API fails
            print(f"OpenAI API error: {e}")
            return self._extract_topics_local(text, num_topics)
    
    def suggest_review_topics(self, text, student_answers=None, num_suggestions=3):
        """
        Suggest topics for review based on text and optionally student answers
        
        Args:
            text (str): The original text
            student_answers (dict): Optional dict of questions and student answers
            num_suggestions (int): Number of suggestions to make
        
        Returns:
            list: List of suggested topics to review
        """
        # Extract all potential topics
        all_topics = self.extract_topics(text, num_topics=num_suggestions*2)
        
        if student_answers and self.model_type == "openai":
            # If we have student answers and can use OpenAI, prioritize topics based on errors
            return self._suggest_topics_with_answers_openai(text, student_answers, all_topics, num_suggestions)
        
        # Otherwise just return the top topics
        return all_topics[:num_suggestions]
    
    def _suggest_topics_with_answers_openai(self, text, student_answers, all_topics, num_suggestions):
        """Use OpenAI to analyze student answers and suggest topics for review"""
        
        # Format student answers for the prompt
        answers_text = ""
        for i, qa in enumerate(student_answers):
            answers_text += f"Question {i+1}: {qa['question']}\n"
            answers_text += f"Correct answer: {qa['correct_answer']}\n"
            answers_text += f"Student answer: {qa['student_answer']}\n\n"
        
        # Create a prompt for OpenAI
        prompt = f"""
        Based on the student's quiz answers and the original text, suggest {num_suggestions} topics 
        the student should review. Focus on areas where the student seems to have misconceptions 
        or knowledge gaps.
        
        Original text excerpt:
        {text[:1000]}...
        
        Student quiz results:
        {answers_text}
        
        Potential topics to choose from:
        {', '.join(all_topics)}
        
        For each suggested topic, provide:
        1. Topic name (choose from the list above or suggest a new one if necessary)
        2. Reason why the student should review this topic
        3. A specific section from the text that covers this topic
        
        Format as JSON.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to basic topics if API fails
            print(f"OpenAI API error: {e}")
            return all_topics[:num_suggestions]