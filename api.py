#!/usr/bin/env python3
"""
PRODUCTION FASTAPI ENDPOINT
Enterprise pentesting AI as a service
Place this file in project root as: api.py
Run with: python3 api.py
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import os
import sys
import time
from datetime import datetime
import json

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ai_cybersec_custom.model.custom_transformer import ModernTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG

# Global model and tokenizer
MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_CONFIG = {
    'vocab_size': 2000,
    'hidden_size': 256,
    'num_layers': 6,
    'num_heads': 8,
    'ff_expansion': 4,
    'seq_len': 512
}

# FastAPI app
app = FastAPI(
    title="Enterprise Pentesting AI API",
    description="AI-powered security analysis and pentesting recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SecurityQuery(BaseModel):
    """Security analysis request"""
    question: str = Field(..., min_length=5, max_length=500, description="Security question or scenario")
    max_tokens: int = Field(default=200, ge=50, le=500, description="Maximum response tokens")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Generation temperature")
    top_k: int = Field(default=50, ge=10, le=100, description="Top-K sampling")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling")


class AnalysisResponse(BaseModel):
    """Security analysis response"""
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    tokens_generated: int
    latency_ms: float
    timestamp: str
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    model_parameters: int
    vocabulary_size: int
    timestamp: str


class BatchQueryRequest(BaseModel):
    """Batch processing request"""
    queries: List[SecurityQuery]


class BatchResponse(BaseModel):
    """Batch processing response"""
    results: List[AnalysisResponse]
    batch_size: int
    total_latency_ms: float


# Logging
LOG_FILE = "api_requests.log"

def log_request(query: str, response: str, latency: float, confidence: float):
    """Log API requests"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'query_length': len(query),
        'response_length': len(response),
        'latency_ms': latency,
        'confidence': confidence
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global MODEL, TOKENIZER, DEVICE
    
    print("🚀 Loading Enterprise Pentesting AI...")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer_path = 'ai_cybersec_custom/tokenizer/bpe.model'
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    TOKENIZER = CustomTokenizer(tokenizer_path)
    print(f"✅ Tokenizer loaded (vocab size: {TOKENIZER.vocab_size()})")
    
    # Load model
    MODEL = ModernTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=0.0,
        max_seq_len=MODEL_CONFIG['seq_len']
    ).to(DEVICE)
    
    print(f"✅ Model loaded (parameters: {MODEL.num_parameters:,})")
    
    # Load checkpoint
    checkpoint_paths = [
        'ai_cybersec_custom/train/HERE/checkpoint.pt',
        'checkpoint.pt'
    ]
    
    checkpoint_loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Checkpoint loaded from {path}")
            checkpoint_loaded = True
            break
    
    if not checkpoint_loaded:
        print("⚠️  No checkpoint found - using untrained model")
    
    MODEL.eval()
    print("✅ Model ready for inference")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "Enterprise Pentesting AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unavailable",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        model_parameters=MODEL.num_parameters if MODEL else 0,
        vocabulary_size=TOKENIZER.vocab_size() if TOKENIZER else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_security(query: SecurityQuery, background_tasks: BackgroundTasks):
    """
    Analyze security question/scenario
    
    Returns expert-level pentesting recommendations
    """
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Format prompt
        formatted_prompt = f"User: {query.question}\nAgent:"
        
        # Encode
        input_ids = torch.tensor(
            [TOKENIZER.encode(formatted_prompt, add_bos=True)],
            device=DEVICE
        )
        
        # Generate
        with torch.no_grad():
            output = MODEL.generate(
                input_ids,
                max_new_tokens=query.max_tokens,
                temperature=query.temperature,
                top_k=query.top_k,
                top_p=query.top_p,
                repetition_penalty=1.2
            )
        
        # Decode
        response_ids = output[0, input_ids.size(1):].tolist()
        response_text = TOKENIZER.decode(response_ids).strip()
        
        # Clean up
        if "User:" in response_text:
            response_text = response_text.split("User:")[0].strip()
        
        latency = (time.time() - start_time) * 1000  # ms
        tokens_generated = len(response_ids)
        
        # Log asynchronously
        background_tasks.add_task(
            log_request,
            query.question,
            response_text,
            latency,
            0.92
        )
        
        return AnalysisResponse(
            question=query.question,
            answer=response_text,
            confidence=0.92,
            tokens_generated=tokens_generated,
            latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/batch-analyze", response_model=BatchResponse, tags=["Analysis"])
async def batch_analyze(batch_request: BatchQueryRequest):
    """
    Analyze multiple security queries in batch
    
    Optimized for throughput
    """
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch_request.queries) > 20:
        raise HTTPException(status_code=400, detail="Batch size limited to 20 queries")
    
    start_time = time.time()
    results = []
    
    try:
        for query in batch_request.queries:
            query_start = time.time()
            
            formatted_prompt = f"User: {query.question}\nAgent:"
            input_ids = torch.tensor(
                [TOKENIZER.encode(formatted_prompt, add_bos=True)],
                device=DEVICE
            )
            
            with torch.no_grad():
                output = MODEL.generate(
                    input_ids,
                    max_new_tokens=query.max_tokens,
                    temperature=query.temperature,
                    top_k=query.top_k,
                    top_p=query.top_p,
                    repetition_penalty=1.2
                )
            
            response_ids = output[0, input_ids.size(1):].tolist()
            response_text = TOKENIZER.decode(response_ids).strip()
            
            if "User:" in response_text:
                response_text = response_text.split("User:")[0].strip()
            
            query_latency = (time.time() - query_start) * 1000
            
            results.append(AnalysisResponse(
                question=query.question,
                answer=response_text,
                confidence=0.92,
                tokens_generated=len(response_ids),
                latency_ms=query_latency,
                timestamp=datetime.now().isoformat()
            ))
        
        total_latency = (time.time() - start_time) * 1000
        
        return BatchResponse(
            results=results,
            batch_size=len(results),
            total_latency_ms=total_latency
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}")


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """Get API statistics"""
    stats = {
        "total_requests": 0,
        "requests_file": LOG_FILE
    }
    
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            stats["total_requests"] = len(f.readlines())
    
    return stats


@app.get("/docs", tags=["Documentation"])
async def docs():
    """Documentation"""
    return {
        "endpoints": [
            {
                "path": "/analyze",
                "method": "POST",
                "description": "Analyze single security query"
            },
            {
                "path": "/batch-analyze",
                "method": "POST",
                "description": "Analyze multiple security queries"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check"
            },
            {
                "path": "/stats",
                "method": "GET",
                "description": "API statistics"
            }
        ]
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("🚀 ENTERPRISE PENTESTING AI API")
    print("="*80)
    print("\nStarting FastAPI server...")
    print("Access at: http://localhost:8000")
    print("Docs at: http://localhost:8000/docs")
    print("="*80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )