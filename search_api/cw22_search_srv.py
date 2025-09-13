import base64
import csv
import logging
import os
import random
import socket
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

import uvicorn
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from tqdm import tqdm

from cw22_searcher import ClueWeb22Searcher
from utils.cw22_api import ClueWeb22Api

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
searcher = None
url_to_docid: Dict[str, str] = {}
MAP_FILE = '/bos/tmp6/jmcoelho/cweb22-b-en/map_id_url.csv'


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


def generate_random_clueweb_id():
    """
    Generate a random ClueWeb document ID for testing purposes
    """
    xx = f"{random.randint(0, 45):02d}"
    yy = f"{random.randint(0, 99):02d}"
    abcde = f"{random.randint(0, 10000):05d}"

    # Ensure a <= 1
    if abcde[0] > '1':
        abcde = '1' + abcde[1:] if random.random() > 0.5 else '0' + abcde[1:]

    return f"clueweb22-en00{xx}-{yy}-{abcde}"


def get_base64(json_bytes):
    """
    Convert bytes to base64 encoded string
    """
    return base64.b64encode(json_bytes).decode()


# Response model
class SearchResponse(BaseModel):
    results: List[str]
    distances: Optional[List[float]] = None


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Initialize the searcher on startup
    global searcher, url_to_docid
    logger.info("Initializing ClueWeb22 searcher...")
    searcher = ClueWeb22Searcher(verbose=False)

    # Load URL to document ID mapping
    logger.info("Loading URL to document ID mapping...")
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for doc_id, url in tqdm(reader):
                url_to_docid[url] = doc_id
        logger.info(f"Loaded {len(url_to_docid)} URL mappings")
    else:
        logger.error(f"Mapping file not found at {MAP_FILE}")
        raise FileNotFoundError(f"Mapping file not found at {MAP_FILE}")

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


@app.get("/search", response_model=SearchResponse)
async def search(
        query: str = Query(..., description="Search query text"),
        k: Optional[int] = Query(10, description="Number of results to return"),
        complexity: Optional[int] = Query(None, description="Searche complexity"),
        num_of_shards: Optional[int] = Query(4, description="Number of shards to use for search"),
        with_distance: Optional[bool] = Query(False, description="Whether with document distance"),
        cw22_a: Optional[bool] = Query(False, description="Whether to search all 40 shards (CW22-A)")
):
    """
    Search endpoint that returns document results for a given query

    Args:
        query: The search query text
        k: Number of results to return (default: 10)
        num_of_shards: Number of shards to use for search (default: 4)
        cw22_a: Whether to search all 40 shards instead of just 4 (default: False)

    Returns:
        JSON with search results
    """
    global searcher

    if not query:
        raise HTTPException(status_code=400, detail="No query text in request")

    if isinstance(k, str):
        k = int(k)

    if isinstance(num_of_shards, str):
        num_of_shards = int(num_of_shards)

    # Validate num_of_shards is in valid range
    max_shards = 40 if cw22_a else 4
    if num_of_shards is not None and (num_of_shards < 1 or num_of_shards > max_shards):
        logger.warning(f"Invalid num_of_shards: {num_of_shards}, must be between 1 and {max_shards}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid num_of_shards: {num_of_shards}. Value must be between 1 and {max_shards}."
        )

    logger.info(
        f"Search query: {query}, k={k}, complexity={complexity}, num_of_shards={num_of_shards}, cw22_a={cw22_a}")

    try:
        json_bytes_list = list()

        # Perform the actual search with num_of_shards parameter
        docs = searcher.search(query,
                               k=k,
                               complexity=complexity,
                               num_of_shards=num_of_shards,
                               with_distance=with_distance,
                               cw22_a=cw22_a)

        if with_distance:
            docids = [docid for _, docid in docs]
            distances = [dist for dist, _ in docs]
        else:
            docids = docs

        doc_texts = searcher.get_doc_texts(docids)

        for doc_text in doc_texts:
            json_bytes_list.append(get_base64(doc_text))

        if with_distance:
            return {"results": json_bytes_list, "distances": distances}
        else:
            return {"results": json_bytes_list}

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch_clean_text")
async def fetch_clean_text(
        url: str = Query(..., description="The URL to fetch the clean text for")
):
    """
    Fetch clean text for a given URL

    Args:
        url: The URL to fetch the clean text for

    Returns:
        JSON with clean text
    """
    doc_id = url_to_docid.get(url.rstrip("\n"))
    if doc_id is None:
        raise HTTPException(status_code=404, detail="URL not found.")

    try:
        clueweb_api = ClueWeb22Api(doc_id)
        clean_txt = eval(clueweb_api.get_clean_text())
    except Exception as e:
        logger.error(f"Error fetching clean text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch clean text: {str(e)}")

    return {"clean_text": clean_txt['Clean-Text']}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    return {"status": "healthy"}


# Mock search endpoint for testing without the actual searcher
@app.get("/mock-search", response_model=SearchResponse)
async def mock_search(
        query: str = Query(..., description="Mock search query text"),
        k: Optional[int] = Query(5, description="Number of mock results to return"),
        num_of_shards: Optional[int] = Query(4, description="Number of shards to use for search (mock)")
):
    """
    Mock search endpoint for testing
    """
    if not query:
        raise HTTPException(status_code=400, detail="No query text in request")

    if isinstance(k, str):
        k = int(k)

    if isinstance(num_of_shards, str):
        num_of_shards = int(num_of_shards)

    logger.info(f"Mock search query: {query}, k={k}, num_of_shards={num_of_shards}")

    try:
        json_bytes_list = list()

        # Generate mock results
        for i in range(k):
            doc_id = generate_random_clueweb_id()
            # In a real implementation, you would fetch actual documents
            json_bytes_list.append(f"Mock document {i + 1} with ID: {doc_id}")

        return {"results": json_bytes_list}

    except Exception as e:
        logger.error(f"Error in mock search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
