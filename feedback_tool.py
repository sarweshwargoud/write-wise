import streamlit as st
import textstat
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="WriteWise | AI Writing Assistant",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dependency Management (Safe Mode) ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    AI_ERROR = str(e)
except OSError as e:
    # Catching DLL load errors (WinError 1114, etc)
    AI_AVAILABLE = False
    AI_ERROR = str(e)

# --- Custom CSS & Design System ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;700&display=swap');

    /* General Reset & Variables */
    :root {
        --primary-color: #6C5DD3;
        --secondary-color: #FF754C;
        --bg-color: #0d0e12;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-color: #ffffff;
        --glass-border: 1px solid rgba(255, 255, 255, 0.1);
        --glass-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* Overall Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(108, 93, 211, 0.2) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(255, 117, 76, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(108, 93, 211, 0.2) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(255, 117, 76, 0.15) 0px, transparent 50%);
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    /* Titles & Headings */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h1 {
        background: linear-gradient(90deg, #FFFFFF 0%, #A0A0A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
    }

    /* Cards / Containers (Glassmorphism) */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: var(--glass-border);
        border-radius: 16px;
        box-shadow: var(--glass-shadow);
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #594BC2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(108, 93, 211, 0.4);
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(108, 93, 211, 0.6);
        background: linear-gradient(135deg, #7B6DE0 0%, #6C5DD3 100%);
    }

    /* Text Area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: var(--glass-border) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 1px var(--primary-color) !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        color: var(--secondary-color);
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }

</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    if not AI_AVAILABLE:
        return None, None
        
    # CoLA Model for Grammar usage
    # LABEL_0: Unacceptable, LABEL_1: Acceptable
    model_name_cola = "textattack/bert-base-uncased-CoLA"
    tokenizer_cola = AutoTokenizer.from_pretrained(model_name_cola)
    model_cola = AutoModelForSequenceClassification.from_pretrained(model_name_cola)
    # Defaulting to pytorch
    grammar_pipeline = pipeline('sentiment-analysis', model=model_cola, tokenizer=tokenizer_cola)

    # T5 Model for Suggestions/Corrections
    model_name_t5 = "t5-small"
    tokenizer_t5 = AutoTokenizer.from_pretrained(model_name_t5)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name_t5)
    t5_pipeline = pipeline('text2text-generation', model=model_t5, tokenizer=tokenizer_t5)

    return grammar_pipeline, t5_pipeline

if AI_AVAILABLE:
    with st.spinner("Initializing AI Core..."):
        try:
            grammar_pipeline, t5_pipeline = load_models()
        except Exception as e:
            st.error(f"Error loading models: {e}.")
            AI_AVAILABLE = False
else:
    # Display fallback message
    st.warning(f"⚠️ **Core AI modules could not be loaded.** Running in Reduced Functionality Mode. (Error: {AI_ERROR if 'AI_ERROR' in locals() else 'System Error'})")

# --- Logic ---

def analyze_grammar(text):
    if not AI_AVAILABLE:
        # Mock Logic for demo
        import random
        score = random.uniform(0.7, 0.99)
        label = "LABEL_1" if score > 0.5 else "LABEL_0"
        return label, score

    # CoLA model logic: Label 1 is acceptable, Label 0 is unacceptable.
    # Truncate to 512
    result = grammar_pipeline(text, truncation=True, max_length=512)[0]
    return result['label'], result['score']

def suggest_improvements(text):
    if not AI_AVAILABLE:
        return "Since AI modules are offline, here is a placeholder suggestion: Consider varying your sentence structure for better flow."

    # T5 Grammar Correction
    input_text = "grammar: " + text
    result = t5_pipeline(input_text, max_length=512, truncation=True)
    return result[0]['generated_text']

def calculate_metrics(text):
    readability = textstat.flesch_kincaid_grade(text)
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    sentences = textstat.sentence_count(text)
    words = textstat.lexicon_count(text)
    unique_words = len(set(text.lower().split()))
    return readability, paragraphs, sentences, words, unique_words

def calculate_grade(grammar_label, grammar_score, readability):
    # Heuristic Data Driven Grading
    score = 0
    # Grammar impact (30%)
    if grammar_label == "LABEL_1":
        score += 30 * grammar_score
    else:
        score += 30 * (1 - grammar_score) # Penalty

    # Readability impact (Aim for 8-12) (30%)
    dist = abs(readability - 10)
    if dist <= 2: score += 30
    elif dist <= 4: score += 20
    else: score += 10
    
    # Vocabulary/Length (40%) - Placeholder logic
    score += 40 # Giving full points for effort for now
    
    if score >= 90: return "A+"
    elif score >= 85: return "A"
    elif score >= 80: return "A-"
    elif score >= 75: return "B+"
    elif score >= 70: return "B"
    elif score >= 60: return "C"
    else: return "Needs Work"

# --- Main UI ---

col_main, col_padding = st.columns([2, 1])

with col_main:
    st.title("WriteWise")
    st.markdown("### Elevate your writing with AI-powered feedback.")
    
    st.markdown("")
    st.markdown("Enter your text below to get instant analysis on **Grammar**, **Readability**, and **Style Suggestions**.")

    input_text = st.text_area("Your Draft", height=250, placeholder="Start typing or paste your essay here...", label_visibility="collapsed")

    if st.button("Analyze Text"):
        if input_text.strip():
            # Analysis
            with st.spinner("Analyzing..."):
                time.sleep(1) # Fake delay for UX if rapid
                g_label, g_score = analyze_grammar(input_text)
                suggestion = suggest_improvements(input_text)
                readability, n_para, n_sent, n_words, n_unique = calculate_metrics(input_text)
                grade = calculate_grade(g_label, g_score, readability)

            # --- Results Area ---
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Row 1: High Level Stats
            cols = st.columns(4)
            with cols[0]:
                st.metric("Grade", grade)
            with cols[1]:
                st.metric("Readability (Grade)", f"{readability:.1f}")
            with cols[2]:
                st.metric("Word Count", n_words)
            with cols[3]:
                status = "Excellent" if g_label == "LABEL_1" else "Needs Review"
                st.metric("Grammar Status", status, delta=f"{g_score:.1%}" if g_label == "LABEL_1" else f"-{g_score:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Row 2: Detailed Breakdown
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🔍 Grammar & Structure")
                if g_label == "LABEL_1":
                    st.success("Your grammar looks solid! The sentence structure flows well.")
                else:
                    st.warning("We detected some potential grammatical awkwardness. Consider revising.")
                
                st.markdown(f"""
                - **Sentences:** {n_sent}
                - **Paragraphs:** {n_para}
                - **Vocabulary Richness:** {n_unique} unique words
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### ✨ Smart Suggestion")
                st.markdown("**Original:**")
                st.caption(f"_{input_text[:150]}..._" if len(input_text) > 150 else f"_{input_text}_")
                st.markdown("**AI Improved:**")
                st.info(suggestion)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("Please enter some text to analyze.")

# Sidebar
with st.sidebar:
    st.markdown("## WriteWise AI")
    if AI_AVAILABLE:
        st.success("AI Models Online")
    else:
        st.error("AI Models Offline")
        
    st.markdown("Powered by:")
    st.markdown("- **BERT (CoLA)** for Grammar")
    st.markdown("- **T5 Transformer** for Suggestions")
    st.markdown("---")
    st.markdown("Developed by **Sarweshwar**")
    st.image("https://img.icons8.com/clouds/200/000000/idea.png", width=150)
