# models/quiz_generator.py

import nltk
from nltk.tokenize import sent_tokenize
import random
import re
import openai
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

class QuizGenerator:
    def __init__(self, model_type="local"):
        """
        Initialize the quiz generator
        
        Args:
            model_type (str): "local" for HuggingFace or "openai" for OpenAI API
        """
        self.model_type = model_type
        
        # Download necessary NLTK data
        nltk.download('punkt')
        
        if model_type == "local":
            # Set up local model for question generation
            model_name = "t5-small"  # You can use larger models like "t5-base" for better results
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Set up a fill-mask model for distractors
            self.fill_mask = pipeline("fill-mask", model="distilroberta-base")
        
        elif model_type == "openai":
            # Use OpenAI API
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        else:
            raise ValueError("Model type must be 'local' or 'openai'")
    
    def generate_mcq_with_local_model(self, text, num_questions=5):
        """Generate multiple-choice questions using local models"""
        # Tokenize into sentences and select important ones
        sentences = sent_tokenize(text)
        
        # Select a subset of sentences for questions
        if len(sentences) <= num_questions:
            selected_sentences = sentences
        else:
            # Select sentences with more content (longer sentences are often more informative)
            sentence_lengths = [(i, len(s.split())) for i, s in enumerate(sentences)]
            sentence_lengths.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in sentence_lengths[:num_questions]]
            selected_indices.sort()  # Keep original order
            selected_sentences = [sentences[idx] for idx in selected_indices]
        
        questions = []
        
        for sentence in selected_sentences:
            if len(questions) >= num_questions:
                break
                
            # Identify potential answer candidates (nouns, numbers, named entities)
            answer = self._extract_answer_candidate(sentence)
            if not answer:
                continue
                
            # Create question by replacing the answer with a blank
            question_text = sentence.replace(answer, "_____")
            
            # Generate distractors (wrong options)
            distractors = self._generate_distractors(answer, text)
            
            # Create multiple choice question
            options = distractors + [answer]
            random.shuffle(options)
            
            questions.append({
                "question": question_text,
                "options": options,
                "answer": answer
            })
            
        return questions
    
    def _extract_answer_candidate(self, sentence):
        """Extract a suitable answer candidate from a sentence"""
        # Look for capitalized words (potential named entities)
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', sentence)
        if capitalized_words and random.random() < 0.7:
            return random.choice(capitalized_words)
            
        # Look for numbers
        numbers = re.findall(r'\b\d+\b', sentence)
        if numbers and random.random() < 0.7:
            return random.choice(numbers)
            
        # Otherwise pick a random noun-like word (longer words are more likely to be content words)
        words = [word for word in sentence.split() if len(word) > 4]
        if words:
            return random.choice(words)
            
        return None
    
    def _generate_distractors(self, answer, text, count=3):
        """Generate distractor options for multiple choice questions"""
        if self.model_type == "local":
            # Try to use mask filling for distractors
            try:
                # Create a masked sentence
                masked_text = f"The word {answer} can be replaced by [MASK]."
                
                # Get predictions
                results = self.fill_mask(masked_text)
                
                # Extract unique distractors
                distractors = []
                for result in results:
                    distractor = result['token_str'].strip()
                    if distractor.lower() != answer.lower() and distractor not in distractors:
                        distractors.append(distractor)
                
                # If we don't have enough distractors, fall back to random words from text
                if len(distractors) < count:
                    words = set(re.findall(r'\b[a-zA-Z]{4,}\b', text))
                    words = [w for w in words if w.lower() != answer.lower() and w not in distractors]
                    distractors.extend(random.sample(words, min(count - len(distractors), len(words))))
                
                return distractors[:count]
            
            except Exception:
                # Fall back to random words from the text
                words = set(re.findall(r'\b[a-zA-Z]{4,}\b', text))
                words = [w for w in words if w.lower() != answer.lower()]
                return random.sample(words, min(count, len(words)))
    
    def generate_with_openai(self, text, num_questions=5, question_type="multiple_choice"):
        """Generate questions using OpenAI API"""
        
        prompt = f"""
        Generate {num_questions} {question_type} questions based on this text:
        
        {text}
        
        For each question, provide:
        1. The question
        2. The correct answer
        """
        
        if question_type == "multiple_choice":
            prompt += "3. Three incorrect options (distractors)\n"
        
        prompt += "\nFormat as JSON."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant that creates quiz questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_questions(self, text, num_questions=5, question_type="multiple_choice"):
        """
        Main method to generate quiz questions
        
        Args:
            text (str): The input text to generate questions from
            num_questions (int): Number of questions to generate
            question_type (str): Type of questions - "multiple_choice" or "open_ended"
            
        Returns:
            list: List of question objects
        """
        if self.model_type == "openai":
            return self.generate_with_openai(text, num_questions, question_type)
        else:
            if question_type == "multiple_choice":
                return self.generate_mcq_with_local_model(text, num_questions)
            else:
                # For open-ended questions with local model
                # This is much more limited without GPT capabilities
                return self._generate_basic_open_questions(text, num_questions)
    
    def _generate_basic_open_questions(self, text, num_questions=3):
        """Generate simple open-ended questions with local processing"""
        sentences = sent_tokenize(text)
        
        # Select sentences with more content
        sentence_lengths = [(i, len(s.split())) for i, s in enumerate(sentences)]
        sentence_lengths.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in sentence_lengths[:num_questions]]
        selected_indices.sort()
        selected_sentences = [sentences[idx] for idx in selected_indices]
        
        questions = []
        
        # Generate basic "Explain about X" type questions
        for sentence in selected_sentences:
            # Extract a topic from the sentence (simple approach)
            words = sentence.split()
            if len(words) < 3:
                continue
                
            # Look for capitalized words as potential topics
            capitalized = [w for w in words if w[0].isupper() and len(w) > 1]
            
            if capitalized:
                topic = random.choice(capitalized)
            else:
                # Pick a random longer word
                candidates = [w for w in words if len(w) > 5]
                topic = random.choice(candidates) if candidates else random.choice(words)
            
            question = f"Explain what the text says about {topic}."
            questions.append({
                "question": question,
                "answer": sentence  # Reference answer
            })
            
            if len(questions) >= num_questions:
                break
                
        return questions