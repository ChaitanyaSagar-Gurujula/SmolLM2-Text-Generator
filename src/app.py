import torch
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from model import GPT, GPTConfig
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import tempfile
from transformers import AutoTokenizer
import uvicorn

# Get the absolute path to the templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

MODEL_ID = "sagargurujula/smollm2-text-generator"

# Initialize FastAPI
app = FastAPI(title="SMOLLM2 Text Generator")

# Templates with absolute path
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use system's temporary directory
cache_dir = Path(tempfile.gettempdir()) / "model_cache"
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_HOME'] = str(cache_dir)

# Load model from Hugging Face Hub
def load_model():
    try:
        # Download the model file from HF Hub with authentication
        model_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="best_model.pth",
            cache_dir=cache_dir,
            token=os.environ.get('HF_TOKEN')  # Get token from environment variable
        )
        
        # Initialize our custom GPT model
        model = GPT(GPTConfig())
        
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path ="HuggingFaceTB/cosmo2-tokenizer", cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the model
model, tokenizer = load_model()

# Define the request body
class TextInput(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(input: TextInput):
    # Prepare input tensor
    input_ids = tokenizer(input.text, return_tensors='pt').input_ids.to(device)
    
    # Generate multiple tokens
    generated_tokens = []
    num_tokens_to_generate = 50  # Generate 20 new tokens
    
    with torch.no_grad():
        generated_tokens = model.generate(input_ids, max_length=50, eos_token_id = tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Return both input and generated text
    return {
        "input_text": input.text,
        "generated_text": generated_text
    }

# Modify the root route to serve the template
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "GPT Text Generator"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

# To run the app, use the command: uvicorn app:app --reload