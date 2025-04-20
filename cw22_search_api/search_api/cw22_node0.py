from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from diskannpy import StaticDiskIndex
from contextlib import asynccontextmanager
import socket
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
loaded_index = None
embeddings = None
qid_mapping = dict()
index_dir = "/ssd2/cw22_ann_index/cw22_b_en/R80L120-0"
embedding_dir = "/bos/usr0/jening/search_service/query_embeddings/"


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
                                index_prefix="cw22_b_en")
        loaded_index = index
        logger.info("-----------------Index shard loaded-----------------")
    except Exception as e:
        logger.error(f"Error initializing index: {str(e)}")
        raise


def query_index(q_emb: str, k: int = 10) -> tuple:
    """
    Query the vector index with the given embedding file

    Args:
        q_emb: The file name of the encoded query embedding vector, like "q_001.npy"
        k: Number of nearest neighbors to retrieve

    Returns:
        Tuple of (raw_doc_ids, distances)
    """
    global loaded_index
    global embedding_dir

    try:
        # Load the embedding from file
        cur_embedding = np.load(embedding_dir + q_emb)
        logger.debug(f"Embedding shape: {cur_embedding.shape}")

        # Search the index
        raw_doc_ids, distances = loaded_index.search(query=cur_embedding,
                                                     k_neighbors=k,
                                                     complexity=k * 5,
                                                     beam_width=10)

        logger.debug(f"Labels: {raw_doc_ids.tolist()}")
        logger.debug(f"Distances: {distances.tolist()}")
        return raw_doc_ids.tolist(), distances.tolist()
    except Exception as e:
        logger.error(f"Error in query_index: {str(e)}")
        raise


# Response model for type hints and documentation
class SearchResponse(BaseModel):
    raw_doc_ids: List[int]
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


@app.get("/search", response_model=SearchResponse)
async def search(q_emb: str, k: int):
    """
    Search endpoint that returns nearest neighbors for a given query embedding

    Args:
        q_emb: The filename of the query embedding
        k: Number of results to return

    Returns:
        JSON with raw document IDs and distances
    """
    try:
        raw_doc_ids, distances = query_index(q_emb, k)
        return {"raw_doc_ids": raw_doc_ids, "distances": distances}
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
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
    port = 51001

    # Print the service information before starting
    print(f"\n======== Service Information ========")
    print(f"Starting search service on HPC cluster")
    print(f"The service will be accessible at: http://{ip_address}:{port}")
    print(f"API documentation will be available at: http://{ip_address}:{port}/docs")
    print(f"======================================\n")

    # Run the server with uvicorn using the current file
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, log_level="info")
