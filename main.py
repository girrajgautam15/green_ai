import yaml
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from openai import OpenAI

# --- Assume mock classification functions and model loading would be here ---
# In a real application, you would load your RoBERTa models and label encoders here.
# For this example, we will simulate the classification like the HTML file did.
import re
import json
import time
import sqlite3
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import openai  # Assuming openai is configured for the LLM API
from contextlib import asynccontextmanager


OPEN_ROUTER_KEY='add api key'
OPEN_ROUTER_BASE_URL="add api key"
GROQ_API_KEY='add api key'
GROQ_BASE_URL='add api key'
GEMINI_API_KEY='add api key'
GOOGLE_BASE_URL="add api key"


llm_prompt = r"""
You are an expert AI assistant that analyzes user requests and provides a structured JSON output. Your task is to receive a raw user query, analyze it according to the rules below, and return a single JSON object.

**--- INSTRUCTIONS ---**

**1. Task Classification:**
Analyze the user's intent and classify the query into ONE of the following predefined categories:

* `code_generation`: Writing code in Python, Java, etc. (excluding SQL).
* `sql_query`: Writing, debugging, or optimizing a SQL query.
* `data_analysis`: Analyzing or finding patterns in a dataset.
* `data_extraction`: Pulling specific information from text (PDFs, emails).
* `report_generation`: Creating a structured summary or document.
* `text_summarization`: Condensing a long piece of text into a short summary.
* `financial_analysis`: Evaluating financial statements, assets, or investments.
* `financial_modeling`: Building predictive financial models (e.g., DCF, LBO).
* `risk_assessment`: Identifying and quantifying financial or operational risks.
* `compliance_check`: Verifying adherence to policies or regulations (AML, KYC).
* `policy_explanation`: Explaining a company policy or regulation.
* `fraud_detection`: Analyzing transactions for potential fraudulent activity.
* `customer_support`: Responding to a customer-facing issue.
* `technical_documentation`: Creating documentation for software or processes.
* `process_automation`: Designing a workflow to automate a business process.
* `sentiment_analysis`: Determining the emotional tone of text.
* `translation`: Translating text from one language to another.
* `text_generation`: General text creation (email, memo, marketing copy).
* `natural_language_query`: A simple question that can be answered directly.
* `miscellaneous`: A task that is not falling into any of the above categorires.


**2. Complexity Assessment:**
Evaluate the complexity of the request from the perspective of the language model's effort and assign one label:

* **low**: A simple, single-step task requiring minimal reasoning (e.g., "Translate this sentence").
* **medium**: A task requiring multiple steps or synthesis of information from a single context (e.g., "Summarize this article").
* **high**: A complex task requiring chained reasoning, deep domain knowledge, and a large, structured output (e.g., "Generate an investment report from these financial statements").


**--- OUTPUT RULES ---**

1.  Your final output MUST be a single, valid JSON object.
2.  The JSON object must contain exactly these two keys: `category`, `complexity`.
3.  **IMPORTANT:** Do NOT write any code, explanations, or text outside of the final JSON object. Your only job is to analyze the user's query and return the JSON.

**--- EXAMPLES ---**

**Example 1:**
* **User Input:** "hey can you please help me write a python script? i need it to connect to our oracle db and pull all the transactions from last month for the credit card division."
* **Expected Output:**
    ```json
    {
      "category": "code_generation",
      "complexity": "medium",
    }
    ```

**Example 2:**
* **User Input:** "what does our WFH policy say about international travel?"
* **Expected Output:**
    ```json
    {
      "category": "policy_explanation",
      "complexity": "low",
    }
    ```

**--- TASK ---**

Analyze the following user query according to all the rules and examples provided above. Respond with ONLY the required JSON object.

"""

openai=OpenAI(api_key=GROQ_API_KEY,base_url=GROQ_BASE_URL)


# Load RoBERTa model, tokenizer, and label encoder
model = AutoModelForSequenceClassification.from_pretrained('zip_model')
tokenizer = AutoTokenizer.from_pretrained('zip_model')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classification.npy', allow_pickle=True)

model_complexity = AutoModelForSequenceClassification.from_pretrained('zip_model_complexity')
tokenizer_complexity= AutoTokenizer.from_pretrained('zip_model_complexity')
label_encoder_complexity = LabelEncoder()
label_encoder_complexity.classes_ = np.load('label_encoder_complexity.npy', allow_pickle=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup events. In this case, it initializes the database.
    """
    init_db()
    yield
    # You can add cleanup code here if needed for shutdown

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

def clean_json_response(response: str) -> dict:
    """
    Clean and parse LLM response to ensure valid JSON.
    """
    response = response.strip()
    response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse extracted JSON: {response}")
        raise ValueError(f"Invalid JSON response: {response}")

def classify_with_llm(message: str, model: str = "llama-3.3-70b-versatile", retries: int = 3) -> dict:
    """
    Classify a single message using the LLM.
    """
    prompt = llm_prompt + f'\n{message}'

    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            reply = response.choices[0].message.content
            return clean_json_response(reply)
        except Exception as e:
            print(f"Error in classification (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)
    return None

def classify_with_roberta(message: str, model, tokenizer, label_encoder) -> tuple:
    """Generic classification function for a given RoBERTa model."""
    model.eval()
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = probabilities[predicted_class_idx]
    return predicted_label, float(confidence)

def classify_message(message: str) -> dict:
    """
    Main function to classify a message: Starts with RoBERTa, falls back to LLM if confidence < 0.7.
    This function now only returns the classification details.
    """
    task_label, task_confidence = classify_with_roberta(message, model, tokenizer, label_encoder)
    complexity_label, complexity_confidence = classify_with_roberta(message, model_complexity, tokenizer_complexity, label_encoder_complexity)
    method = "roberta"
    print(f"RoBERTa Task: '{task_label}' (Conf: {task_confidence:.2f}), Complexity: '{complexity_label}' (Conf: {complexity_confidence:.2f})")

    if task_confidence < 0.7 or complexity_confidence < 0.7:
        print("Confidence below threshold. Falling back to LLM...")
        llm_result = classify_with_llm(message)
        print(f"LLM Result: {llm_result}")

        if llm_result and llm_result.get("category") and llm_result.get("complexity"):
            method = "llm"
            task_label = llm_result.get("category")
            task_confidence = None
            complexity_label = llm_result.get("complexity")
            complexity_confidence = None
        else:
            method = "roberta (llm_failed)"

    return {
        "method": method,
        "task_label": task_label,
        "task_confidence": task_confidence,
        "complexity_label": complexity_label,
        "complexity_confidence": complexity_confidence
    }
# --------------------------------------------------------------------

# --- Database and Configuration Setup ---
DB_FILE = 'classifications.db'
CONFIG_FILE = 'model_config.yaml'

def init_db():
    with sqlite3.connect(DB_FILE, timeout=20.0) as conn:
        c = conn.cursor()
        c.execute("PRAGMA foreign_keys = ON;")
        c.execute('''
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT NOT NULL, method TEXT NOT NULL,
                task_label TEXT, complexity_label TEXT, task_confidence REAL,
                complexity_confidence REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS model_rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, classification_id INTEGER NOT NULL, rank INTEGER NOT NULL,
                model_name TEXT NOT NULL, provider TEXT NOT NULL, region TEXT NOT NULL, score REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (classification_id) REFERENCES classifications (id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
    print("Database initialized.")

def load_model_profiles():
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)['model_profiles']

# --- Scoring and Ranking Logic (from previous steps) ---
def rank_models_for_task(model_profiles: list, task_label: str, complexity_label: str) -> pd.DataFrame:
    candidates = [
        model for model in model_profiles
        if complexity_label in model['complexity_label_mapping'] and
           ("all" in model['complexity_label_mapping'][complexity_label] or
            task_label in model['complexity_label_mapping'][complexity_label])
    ]
    if not candidates: return pd.DataFrame()

    processed = []
    for model in candidates:
        price_in = model['pricing']['input_per_million_tokens']
        price_out = model['pricing']['output_per_million_tokens']
        avg_cost = (price_in * 0.25) + (price_out * 0.75)
        processed.append({'model_name': model['model_name'], 'provider': model['provider'], 'region': model['region'], 'latency': model['latency_ms_ttft'], 'co2': model['co2_footprint_g_per_million_tokens'], 'avg_cost': avg_cost})

    df = pd.DataFrame(processed)
    for metric in ['latency', 'co2', 'avg_cost']:
        min_val, max_val = df[metric].min(), df[metric].max()
        df[f'norm_{metric}'] = 1 - (df[metric] - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 1.0

    weights = {'co2': 0.4, 'latency': 0.3, 'avg_cost': 0.3}
    df['final_score'] = (df['norm_co2'] * weights['co2'] + df['norm_latency'] * weights['latency'] + df['norm_avg_cost'] * weights['avg_cost'])
    df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


# Define the structure of the incoming request
class RankRequest(BaseModel):
    prompt: str
    complexity_override: str # "auto", "low", "medium", "high"

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/rank_models")
def rank_models_endpoint(request: RankRequest):
    """
    Main API endpoint to handle a user request.
    """
    start_time = time.time()

    # Step 1: Classify the prompt using the real classification logic
    classification_result = classify_message(request.prompt)
    method = classification_result['method']
    task_label = classification_result['task_label']
    task_confidence = classification_result['task_confidence']
    complexity_label = classification_result['complexity_label']
    complexity_confidence = classification_result['complexity_confidence']

    # Step 2: Apply user override if provided
    if request.complexity_override != "auto":
        complexity_label = request.complexity_override
        method += " (complexity_override)" # Note the override

    # Step 3: Store detailed classification in DB
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO classifications (message, method, task_label, complexity_label, task_confidence, complexity_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (request.prompt, method, task_label, complexity_label, task_confidence, complexity_confidence))
        classification_id = c.lastrowid
        conn.commit()

    # Step 4: Load model profiles and rank
    model_profiles = load_model_profiles()
    ranked_df = rank_models_for_task(model_profiles, task_label, complexity_label)

    # Step 5: Store top 5 ranks in DB
    top_5 = ranked_df.head(5)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        for _, row in top_5.iterrows():
            c.execute('INSERT INTO model_rankings (classification_id, rank, model_name, provider, region, score) VALUES (?, ?, ?, ?, ?, ?)',
                      (classification_id, int(row['rank']), row['model_name'], row['provider'], row['region'], float(row['final_score'])))
        conn.commit()

    # Step 6: Prepare and return the response
    # We add the raw values back for display on the frontend
    response_data = top_5[['rank', 'model_name', 'provider', 'region', 'latency', 'co2', 'avg_cost', 'final_score']].to_dict(orient='records')

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000

    return {
        "task_label": task_label,
        "complexity_label": complexity_label,
        "ranked_models": response_data,
        "processing_time_ms": processing_time
    }
# --------------------------------------------------------------------
