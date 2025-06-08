import base64
import logging
import random
import socket
import csv
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

import uvicorn
from fastapi import FastAPI, Query, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from tqdm import tqdm

from cw22_searcher import ClueWeb22Searcher
from fw_searcher import FineWebSearcher

from utils.cw22_api import ClueWeb22Api

from auth.auth_db import *
from sqlite_db.search_logger import init_search_logger, log_search_async

from fw_searcher import FineWebSearcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
cw22_searcher = None
fw_searcher = None

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


def get_base64(json_bytes):
    """
    Convert bytes to base64 encoded string
    """
    return base64.b64encode(json_bytes).decode()


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Initialize the cw22_searcher on startup
    global cw22_searcher, fw_searcher

    logger.info("Initializing auth db...")
    init_auth()
    
    logger.info("Initializing search logger...")
    init_search_logger()
    
    logger.info("Initializing ClueWeb22 searcher...")
    cw22_searcher = ClueWeb22Searcher(verbose=False)
    fw_searcher = FineWebSearcher(verbose=False)
    
    # Display service information with actual IP
    ip_address = get_ip_address()
    port = 51000
    logger.info(f"Search service is accessible at: http://{ip_address}:{port}")
    logger.info(f"API documentation available at: http://{ip_address}:{port}/docs")

    yield

    # Clean up resources if needed on shutdown
    logger.info("Shutting down search service...")


# Initialize FastAPI app
app = FastAPI(
    title="ClueWeb22 Search API",
    description="API for searching ClueWeb22 documents",
    version="1.0.0",
    lifespan=lifespan
)


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify the API key from the X-API-Key header"""
    if not verify_api_key_exists(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


@app.get("/search")
async def search(
        request: Request,
        query: str = Query(..., description="Search query text"),
        k: Optional[int] = Query(10, description="Number of results to return"),
        complexity: Optional[int] = Query(None, description="Searche complexity"),
        num_of_shards: Optional[int] = Query(4, description="Number of shards to use for search"),
        with_distance: Optional[bool] = Query(False, description="Whether with document distance"),
        with_outlink: Optional[bool] = Query(False, description="Whether with document outlink"),
        api_key_valid: bool = Depends(verify_api_key)
):
    """
    Search endpoint that returns document results for a given query

    Args:
        query: The search query text
        k: Number of results to return (default: 10)
        num_of_shards: Number of shards to use for search (default: 4)

    Returns:
        JSON with search results
    """
    global cw22_searcher

    if not query:
        raise HTTPException(status_code=400, detail="No query text in request")

    if isinstance(k, str):
        k = int(k)

    if isinstance(num_of_shards, str):
        num_of_shards = int(num_of_shards)

    # Validate num_of_shards is in valid range
    if num_of_shards is not None and (num_of_shards < 1 or num_of_shards > 4):
        logger.warning(f"Invalid num_of_shards: {num_of_shards}, must be between 1 and 4")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid num_of_shards: {num_of_shards}. Value must be between 1 and 4."
        )

    # Log search request asynchronously
    client_ip = request.client.host
    log_search_async(client_ip, query, k, complexity, num_of_shards, with_distance, with_outlink, 'clueweb22')

    logger.info(f"Search query: {query}, k={k}, complexity={complexity}, num_of_shards={num_of_shards}")

    try:
        json_bytes_list = list()

        # Perform the actual search with num_of_shards parameter
        docs = cw22_searcher.search(query, k=k, complexity=complexity, num_of_shards=num_of_shards,
                               with_distance=with_distance)

        distances = None
        if with_distance:
            docids = [docid for _, docid in docs]
            distances = [dist for dist, _ in docs]
        else:
            docids = docs

        doc_texts = cw22_searcher.get_doc_texts(docids)
        for doc_text in doc_texts:
            json_bytes_list.append(get_base64(doc_text))

        doc_outlinks = list()
        if with_outlink:
            raw_outlinks = cw22_searcher.get_outlinks(docids)
            for outlink_str in raw_outlinks:
                doc_outlinks.append(get_base64(outlink_str))

        return {"results": json_bytes_list, "distances": distances, "outlinks": doc_outlinks}

    except Exception as e:
        logger.error(f"Error in cw22_searcher: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fineweb/search")
async def fineweb_search(
        request: Request,
        query: str = Query(..., description="Search query text"),
        k: Optional[int] = Query(10, description="Number of results to return"),
        complexity: Optional[int] = Query(None, description="Searche complexity"),
        num_of_shards: Optional[int] = Query(4, description="Number of shards to use for search"),
        with_distance: Optional[bool] = Query(False, description="Whether with document distance")
):
    """
    Search endpoint that returns document results for a given query

    Args:
        query: The search query text
        k: Number of results to return (default: 10)
        num_of_shards: Number of shards to use for search (default: 4)

    Returns:
        JSON with search results
    """
    global fw_searcher

    if not query:
        raise HTTPException(status_code=400, detail="No query text in request")

    if isinstance(k, str):
        k = int(k)

    if isinstance(num_of_shards, str):
        num_of_shards = int(num_of_shards)

    # Validate num_of_shards is in valid range
    if num_of_shards is not None and (num_of_shards < 1 or num_of_shards > 4):
        logger.warning(f"Invalid num_of_shards: {num_of_shards}, must be between 1 and 4")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid num_of_shards: {num_of_shards}. Value must be between 1 and 4."
        )

    # Log search request asynchronously
    client_ip = request.client.host
    log_search_async(client_ip, query, k, complexity, num_of_shards, with_distance, False, 'fineweb')

    logger.info(f"Search query: {query}, k={k}, complexity={complexity}, num_of_shards={num_of_shards}")

    try:
        json_bytes_list = list()

        # Perform the actual search with num_of_shards parameter
        docs = fw_searcher.search(query, k=k, complexity=complexity, num_of_shards=num_of_shards,
                               with_distance=with_distance)

        distances = None
        if with_distance:
            docids = [docid for _, docid in docs]
            distances = [dist for dist, _ in docs]
        else:
            docids = docs

        doc_texts = fw_searcher.get_doc_texts(docids)

        for doc_text in doc_texts:
            json_bytes_list.append(get_base64(doc_text))

        return {"results": json_bytes_list, "distances": distances, "outlinks": None}

    except Exception as e:
        logger.error(f"Error in fw_searcher: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if cw22_searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    return {"status": "healthy"}


if __name__ == "__main__":
    # Get the machine's IP address
    ip_address = get_ip_address()
    port = 51000

    # Print the service information before starting
    print(f"\n======== Search Service Information ========")
    print(f"Starting ClueWeb22 search service")
    print(f"The service will be accessible at: http://{ip_address}:{port}")
    print(f"API documentation will be available at: http://{ip_address}:{port}/docs")
    print(f"============================================\n")

    # Run the server with uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, log_level="info")