from cw22_searcher import ClueWeb22Searcher
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import random
import base64
import uvicorn
import socket
import logging
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
searcher = None


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


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Initialize the searcher on startup
    global searcher
    logger.info("Initializing ClueWeb22 searcher...")
    searcher = ClueWeb22Searcher(verbose=False)

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
        num_of_shards: Optional[int] = Query(4, description="Number of shards to use for search")
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
    global searcher

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

    logger.info(f"Search query: {query}, k={k}, num_of_shards={num_of_shards}")

    try:
        json_bytes_list = list()

        # Perform the actual search with num_of_shards parameter
        docids = searcher.search(query, k=k, num_of_shards=num_of_shards)
        doc_texts = searcher.get_doc_texts(docids)

        for doc_text in doc_texts:
            json_bytes_list.append(get_base64(doc_text))

        return {"results": json_bytes_list}

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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