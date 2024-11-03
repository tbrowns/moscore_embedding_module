import os

from supabase import create_client, Client

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from dotenv import load_dotenv

from embedding_module import TextEmbedder

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase:Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class EmbeddingsRequest(BaseModel):
    cluster_uuid: str       


@app.options("/api/embeddings/")
async def options_embeddings():
    return {}  # This handles the OPTIONS request

@app.post("/api/embeddings/")
async def generate_embeddings(request: EmbeddingsRequest):
    try:
        embeddings_module = TextEmbedder(SUPABASE_URL, SUPABASE_KEY)
        embeddings_module.generate_embeddings(request.cluster_uuid)

        return {
            "message": "Embeddings generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/api/health")
async def read_root():
    return {"message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

