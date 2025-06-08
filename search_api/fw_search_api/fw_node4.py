import logging
import socket
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import uvicorn
from diskannpy import StaticDiskIndex
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
loaded_index = None
embeddings = None
qid_mapping = dict()
index_dir = "/ssd2/fineweb_ann_index/R80L120-4"


def get_ip_address():
    """
    Get the machine's IP address that can be used to access the service
    """
    try:
        # Create a temporary socket to determine the IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external server (doesn't actually send anything)
        s.connect(("8.8.8.8", 80))
        # Get the IP address used for this connection
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        logger.warning(f"Could not determine IP address: {str(e)}")
        # Fall back to hostname
        return socket.gethostbyname(socket.gethostname())


def init_index():
    """
    Initialize the vector search index
    """
    global loaded_index
    global index_dir

    try:
        index = StaticDiskIndex(index_directory=index_dir,
                                num_threads=16,
                                num_nodes_to_cache=30000,
                                cache_mechanism=1,
                                distance_metric='mips',
                                vector_dtype=np.float32,
                                dimensions=1024,
                                index_prefix="fineweb")
        loaded_index = index
        logger.info("-----------------Index shard loaded-----------------")
    except Exception as e:
        logger.error(f"Error initializing index: {str(e)}")
        raise


def query_index(q_emb: np.ndarray, k: int = 10, complexity: int = 50):
    """
    Query the vector index with the given embedding vector

    Args:
        q_emb: The query embedding vector
        k: Number of nearest neighbors to retrieve
        complexity: Search complexity parameter

    Returns:
        Tuple of (raw_doc_ids, distances)
    """
    global loaded_index

    try:
        # Reshape if needed to ensure correct dimensions
        if len(q_emb.shape) > 1:
            q_emb = q_emb.squeeze()

        logger.debug(f"Embedding shape: {q_emb.shape}")

        # Search the index
        raw_doc_ids, distances = loaded_index.search(query=q_emb,
                                                     k_neighbors=k,
                                                     complexity=complexity,
                                                     beam_width=10)

        logger.debug(f"Labels: {raw_doc_ids.tolist()}")
        logger.debug(f"Distances: {distances.tolist()}")
        return raw_doc_ids.tolist(), distances.tolist()
    except Exception as e:
        logger.error(f"Error in query_index: {str(e)}")
        raise


# Request and response models
class SearchRequest(BaseModel):
    q_emb: List[float]
    k: int
    complexity: int


# Response model for type hints and documentation
class SearchResponse(BaseModel):
    indices: List[int]
    distances: List[float]


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Initialize the index on startup
    init_index()

    # Display service information with actual IP
    ip_address = get_ip_address()
    port = 51001
    logger.info(f"Service is accessible at: http://{ip_address}:{port}")
    logger.info(f"API documentation available at: http://{ip_address}:{port}/docs")

    yield
    # Clean up resources if needed on shutdown
    logger.info("Shutting down service...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Vector Search API",
    description="API for vector similarity search using diskannpy",
    version="1.0.0",
    lifespan=lifespan
)


# New POST endpoint that accepts embedding directly
@app.post("/search_fw", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """
    Search endpoint that accepts embedding vector directly in the request body

    Args:
        request: SearchRequest containing query embedding vector and search parameters

    Returns:
        JSON with raw document IDs and distances
    """
    try:
        # Convert embedding list to numpy array
        q_emb = np.array(request.q_emb, dtype=np.float32)

        raw_doc_ids, distances = query_index(q_emb, request.k, request.complexity)
        return {"indices": raw_doc_ids, "distances": distances}
    except Exception as e:
        logger.error(f"Error in search_post endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if loaded_index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {"status": "healthy"}


if __name__ == "__main__":
    # Get the machine's IP address
    ip_address = get_ip_address()
    port = 51011

    # Print the service information before starting
    print(f"\n======== Service Information ========")
    print(f"Starting search service on HPC cluster")
    print(f"The service will be accessible at: http://{ip_address}:{port}")
    print(f"API documentation will be available at: http://{ip_address}:{port}/docs")
    print(f"======================================\n")

    # Run the server with uvicorn using the current file
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, log_level="info")
