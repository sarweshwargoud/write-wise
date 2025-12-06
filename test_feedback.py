import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Importing TensorFlow...")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
import textstat

print("Loading models...")
try:
    # Using TF-compatible models
    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    bert_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    bert_pipeline = pipeline('sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer, framework="tf")
    
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
    t5_model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    t5_pipeline = pipeline('text2text-generation', model=t5_model, tokenizer=t5_tokenizer, framework="tf")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

text = "Me go to school yesterday."

print(f"Testing with text: '{text}'")

# Test BERT
try:
    result = bert_pipeline(text)[0]
    print(f"BERT Result: {result}")
except Exception as e:
    print(f"BERT Error: {e}")

# Test T5
try:
    input_text = "grammar: " + text
    result = t5_pipeline(input_text, max_length=128)[0]['generated_text']
    print(f"T5 Suggestion: {result}")
except Exception as e:
    print(f"T5 Error: {e}")
