# app.py

import streamlit as st
import json
import os
from dotenv import load_dotenv
from models.summarizer import Summarizer
from models.quiz_generator import QuizGenerator
from models.topic_suggester import TopicSuggester

# Load environment variables from .env file (for API keys)
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions
def create_summarizer():
    """Create and return the appropriate summarizer based on settings"""
    model_type = st.session_state.get("model_type", "local")
    return Summarizer(model_type=model_type)

def create_quiz_generator():
    """Create and return the appropriate quiz generator based on settings"""
    model_type = st.session_state.get("model_type", "local")
    return QuizGenerator(model_type=model_type)

def create_topic_suggester():
    """Create and return the appropriate topic suggester based on settings"""
    model_type = st.session_state.get("model_type", "local")
    return TopicSuggester(model_type=model_type)

def display_quiz(questions):
    """Display quiz questions and handle user answers"""
    
    # Initialize session state for quiz if needed
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
    
    # Display each question
    if isinstance(questions, str):
        # Try to parse JSON string (from OpenAI)
        try:
            questions = json.loads(questions)
            # Handle different possible JSON structures
            if isinstance(questions, dict) and 'questions' in questions:
                questions = questions['questions']
        except json.JSONDecodeError:
            st.error("Failed to parse questions. Please try again.")
            return
    
    if not questions:
        st.warning("No questions were generated. Try with a longer text.")
        return
        
    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                # Handle multiple choice questions
                if 'question' in q and 'options' in q:
                    st.markdown(f"**Q{i+1}: {q['question']}**")
                    
                    # Display options as radio buttons
                    option_key = f"q_{i}"
                    options = q.get('options', [])
                    
                    if options:
                        st.session_state.quiz_answers[option_key] = st.radio(
                            "Select your answer:",
                            options,
                            key=option_key
                        )
                    else:
                        # Handle open-ended questions
                        st.session_state.quiz_answers[option_key] = st.text_area(
                            "Your answer:",
                            key=option_key
                        )
                    
                    st.markdown("---")
                else:
                    # Alternative format handling
                    st.markdown(f"**Q{i+1}: {q.get('question', str(q))}**")
                    option_key = f"q_{i}"
                    st.session_state.quiz_answers[option_key] = st.text_area(
                        "Your answer:",
                        key=option_key
                    )
                    st.markdown("---")
            else:
                # Handle plain text questions
                st.markdown(f"**Q{i+1}: {q}**")
                option_key = f"q_{i}"
                st.session_state.quiz_answers[option_key] = st.text_area(
                    "Your answer:",
                    key=option_key
                )
                st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button("Submit Quiz")
        if submitted:
            st.session_state.quiz_submitted = True
    
    # Show results after submission
    if st.session_state.quiz_submitted:
        score = 0
        total = len(questions)
        results = []
        
        for i, q in enumerate(questions):
            option_key = f"q_{i}"
            user_answer = st.session_state.quiz_answers.get(option_key, "")
            
            if isinstance(q, dict):
                correct_answer = q.get('answer', q.get('correct_answer', None))
                
                if correct_answer and user_answer:
                    is_correct = user_answer.lower() == correct_answer.lower()
                    if is_correct:
                        score += 1
                    
                    results.append({
                        'question': q.get('question', ''),
                        'correct_answer': correct_answer,
                        'student_answer': user_answer,
                        'is_correct': is_correct
                    })
        
        # Display score
        st.success(f"Your score: {score}/{total}")
        
        # Option to review
        with st.expander("Review Answers"):
            for res in results:
                st.markdown(f"**Question:** {res['question']}")
                st.markdown(f"**Your answer:** {res['student_answer']}")
                st.markdown(f"**Correct answer:** {res['correct_answer']}")
                if res['is_correct']:
                    st.markdown("✅ **Correct!**")
                else:
                    st.markdown("❌ **Incorrect**")
                st.markdown("---")
        
        # Generate topic suggestions based on quiz results
        if st.button("Suggest Topics to Review"):
            text = st.session_state.get("current_text", "")
            if text:
                topic_suggester = create_topic_suggester()
                suggestions = topic_suggester.suggest_review_topics(
                    text, 
                    results, 
                    num_suggestions=3
                )
                
                st.subheader("Suggested Topics to Review")
                
                if isinstance(suggestions, str):
                    # Try to parse JSON from OpenAI
                    try:
                        suggestions_data = json.loads(suggestions)
                        
                        if isinstance(suggestions_data, dict) and 'topics' in suggestions_data:
                            topics = suggestions_data['topics']
                            for topic in topics:
                                st.markdown(f"**{topic['name']}**")
                                st.markdown(f"{topic['explanation']}")
                                st.markdown("---")
                        else:
                            st.write(suggestions)
                    except json.JSONDecodeError:
                        st.write(suggestions)
                else:
                    # Handle list of topics
                    for topic in suggestions:
                        st.markdown(f"**{topic}**")
                        st.markdown("---")

# Main application
def main():
    # Sidebar for settings
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_options = ["local", "openai"]
    selected_model = st.sidebar.selectbox(
        "Select Model Type",
        model_options,
        index=model_options.index(st.session_state.get("model_type", "local"))
    )
    
    # Update session state
    st.session_state.model_type = selected_model
    
    # API key input for OpenAI
    if selected_model == "openai":
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password"
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This AI Study Assistant helps students summarize texts, "
        "generate quizzes, and suggest topics to review."
    )
    
    # Main content area
    st.title("📚 AI Study Assistant")
    
    # Initialize session state for tabs if needed
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Text Input"
        
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Text Input", "Summarizer", "Quiz Generator", "Topic Suggester"])
    
    with tab1:
        st.header("Enter Your Study Text")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Text", "Upload File"],
            horizontal=True
        )
        
        if input_method == "Text":
            text_input = st.text_area(
                "Paste your text here:",
                height=300
            )
            if text_input:
                st.session_state.current_text = text_input
                st.success("Text saved! You can now use the other tabs.")
        else:
            uploaded_file = st.file_uploader("Upload a text file:", type=["txt", "pdf", "docx"])
            if uploaded_file:
                try:
                    # Handle different file types
                    if uploaded_file.name.endswith('.txt'):
                        text_input = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.name.endswith('.pdf'):
                        st.warning("PDF support requires additional libraries. Using text extraction.")
                        text_input = "PDF content extraction placeholder. Install PyPDF2 for actual extraction."
                    elif uploaded_file.name.endswith('.docx'):
                        st.warning("DOCX support requires additional libraries. Using text extraction.")
                        text_input = "DOCX content extraction placeholder. Install python-docx for actual extraction."
                    else:
                        text_input = uploaded_file.read().decode("utf-8")
                    
                    st.session_state.current_text = text_input
                    st.success("File uploaded successfully! You can now use the other tabs.")
                    
                    # Preview
                    with st.expander("Preview Text"):
                        st.write(text_input[:500] + "..." if len(text_input) > 500 else text_input)
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    with tab2:
        st.header("Text Summarizer")
        
        if 'current_text' not in st.session_state or not st.session_state.current_text:
            st.info("Please enter some text in the 'Text Input' tab first.")
        else:
            # Summarization options
            st.subheader("Summarization Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                summary_method = st.radio(
                    "Summary Method:",
                    ["abstractive", "extractive"]
                )
            
            with col2:
                if summary_method == "extractive":
                    ratio = st.slider("Summary Ratio:", 0.1, 0.5, 0.3, 0.05)
                    params = {"ratio": ratio}
                else:
                    max_length = st.slider("Max Summary Length:", 50, 500, 150, 10)
                    params = {"max_length": max_length}
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        summarizer = create_summarizer()
                        summary = summarizer.summarize(
                            st.session_state.current_text,
                            method=summary_method,
                            **params
                        )
                        
                        st.session_state.current_summary = summary
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            
            # Display summary if available
            if 'current_summary' in st.session_state and st.session_state.current_summary:
                st.subheader("Summary")
                st.write(st.session_state.current_summary)
                
                # Copy button (using JavaScript)
                st.markdown("""
                <button onclick="navigator.clipboard.writeText(document.getElementById('summary-text').innerText)">
                    Copy Summary
                </button>
                <div id="summary-text" style="display:none">{}</div>
                """.format(st.session_state.current_summary), unsafe_allow_html=True)
    
    with tab3:
        st.header("Quiz Generator")
        
        if 'current_text' not in st.session_state or not st.session_state.current_text:
            st.info("Please enter some text in the 'Text Input' tab first.")
        else:
            # Quiz options
            st.subheader("Quiz Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                question_type = st.radio(
                    "Question Type:",
                    ["multiple_choice", "open_ended"]
                )
            
            with col2:
                num_questions = st.slider("Number of Questions:", 3, 10, 5)
            
            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz questions..."):
                    try:
                        quiz_generator = create_quiz_generator()
                        questions = quiz_generator.generate_questions(
                            st.session_state.current_text,
                            num_questions=num_questions,
                            question_type=question_type
                        )
                        
                        st.session_state.current_questions = questions
                        
                    except Exception as e:
                        st.error(f"Error generating quiz: {e}")
            
            # Display quiz if available
            if 'current_questions' in st.session_state and st.session_state.current_questions:
                st.subheader("Quiz")
                display_quiz(st.session_state.current_questions)
    
    with tab4:
        st.header("Topic Suggester")
        
        if 'current_text' not in st.session_state or not st.session_state.current_text:
            st.info("Please enter some text in the 'Text Input' tab first.")
        else:
            # Topic options
            num_topics = st.slider("Number of Topics:", 3, 10, 5)
            
            if st.button("Extract Key Topics"):
                with st.spinner("Extracting topics..."):
                    try:
                        topic_suggester = create_topic_suggester()
                        topics = topic_suggester.extract_topics(
                            st.session_state.current_text,
                            num_topics=num_topics
                        )
                        
                        st.session_state.current_topics = topics
                        
                    except Exception as e:
                        st.error(f"Error extracting topics: {e}")
            
            # Display topics if available
            if 'current_topics' in st.session_state and st.session_state.current_topics:
                st.subheader("Key Topics")
                
                if isinstance(st.session_state.current_topics, str):
                    # Try to parse JSON from OpenAI
                    try:
                        topics_data = json.loads(st.session_state.current_topics)
                        
                        if isinstance(topics_data, dict) and 'topics' in topics_data:
                            topics = topics_data['topics']
                            for topic in topics:
                                st.markdown(f"**{topic['name']}**")
                                st.markdown(f"{topic['explanation']}")
                                st.markdown("---")
                        else:
                            st.write(st.session_state.current_topics)
                    except json.JSONDecodeError:
                        st.write(st.session_state.current_topics)
                else:
                    # Handle list of topics
                    for i, topic in enumerate(st.session_state.current_topics):
                        st.markdown(f"**{i+1}. {topic}**")
                        st.markdown("---")


if __name__ == "__main__":
    main()