import asyncio
import base64
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
import uvloop
from fastapi import FastAPI, Query, HTTPException, Header, Request
from fastapi.responses import JSONResponse

from auth.auth_db import *
from cw22_qwen_searcher import ClueWeb22Searcher
from sqlite_db.search_logger import init_search_logger, log_search_async

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set uvloop as event loop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global variables
cw22_engine = None

CPU_THREAD_POOL = None
IO_THREAD_POOL = None
ENCODE_SEMAPHORE = None

PORT = 51006


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
    # Initialize the cw22_engine on startup
    global cw22_engine, CPU_THREAD_POOL, IO_THREAD_POOL, ENCODE_SEMAPHORE

    logger.info("Initializing thread pools...")
    # CPU-intensive tasks: use most cores for vector search and document processing
    CPU_THREAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="CPU-Worker")
    # I/O intensive tasks: network requests, file reading
    IO_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="IO-Worker")
    # Limit GPU encoding to prevent resource contention
    ENCODE_SEMAPHORE = asyncio.Semaphore(2)

    # logger.info("Initializing auth db...")
    # init_auth()

    # logger.info("Initializing search logger...")
    # init_search_logger()

    logger.info("Initializing ClueWeb22 searcher...")
    cw22_engine = ClueWeb22Searcher()

    # Display service information with actual IP
    ip_address = get_ip_address()
    port = PORT
    logger.info(f"Search service is accessible at: http://{ip_address}:{port}")
    logger.info(f"API documentation available at: http://{ip_address}:{port}/docs")

    yield

    # Clean up resources if needed on shutdown
    logger.info("Shutting down search service...")
    CPU_THREAD_POOL.shutdown(wait=True)
    IO_THREAD_POOL.shutdown(wait=True)


# Initialize FastAPI app
app = FastAPI(
    title="ClueWeb22 Search API",
    description="API for searching ClueWeb22 documents",
    version="1.0.0",
    lifespan=lifespan
)


def search_with_pools(query, k, complexity, num_of_shards,
                      with_distance, with_outlink, cw22_a):
    """
    Execute search using optimized thread pools for different types of work
    """
    global cw22_engine, IO_THREAD_POOL

    # Perform search (this will use internal thread pools)
    docs = cw22_engine.search(
        query, k=k, complexity=complexity, num_of_shards=num_of_shards,
        with_distance=with_distance, cw22_a=cw22_a
    )

    # Process results
    if with_distance:
        docids = [docid for _, docid in docs]
        distances = [dist for dist, _ in docs]
    else:
        docids = docs
        distances = None

    # Use I/O thread pool for document retrieval
    doc_texts = cw22_engine.get_doc_texts_parallel(docids, IO_THREAD_POOL)
    json_bytes_list = [get_base64(doc_text) for doc_text in doc_texts]

    doc_outlinks = []
    if with_outlink:
        outlinks = cw22_engine.get_outlinks_parallel(docids, IO_THREAD_POOL)
        doc_outlinks = [get_base64(outlink) for outlink in outlinks]

    return {
        "results": json_bytes_list,
        "distances": distances,
        "outlinks": doc_outlinks
    }


@app.get("/cw22_qwen/search")
async def search(
        request: Request,
        query: str = Query(..., description="Search query text"),
        k: Optional[int] = Query(10, description="Number of results to return"),
        complexity: Optional[int] = Query(None, description="Searche complexity"),
        num_of_shards: Optional[int] = Query(4, description="Number of shards to use for search"),
        with_distance: Optional[bool] = Query(False, description="Whether with document distance"),
        with_outlink: Optional[bool] = Query(False, description="Whether with document outlink"),
        cw22_a: Optional[bool] = Query(False, description="Whether to search all 40 shards (CW22-A)"),
        x_api_key: Optional[str] = Header(None)
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
    global cw22_engine

    if not query:
        raise HTTPException(status_code=400, detail="No query text in request")

    if isinstance(k, str):
        k = int(k)

    if isinstance(num_of_shards, str):
        num_of_shards = int(num_of_shards)

    # Validate num_of_shards is in valid range
    max_shards = 4
    if num_of_shards is not None and (num_of_shards < 1 or num_of_shards > max_shards):
        logger.warning(f"Invalid num_of_shards: {num_of_shards}, must be between 1 and {max_shards}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid num_of_shards: {num_of_shards}. Value must be between 1 and {max_shards}."
        )

    # Log search request
    client_ip = request.client.host
    x_api_key = x_api_key or ''
    # verify_result = verify_api_key_exists(x_api_key)

    # # Log search request with actual verification result
    # log_search_async(ip_address=client_ip,
    #                  query_text=query,
    #                  k=k,
    #                  complexity=complexity,
    #                  num_of_shards=num_of_shards,
    #                  with_distance=with_distance,
    #                  with_outlink=with_outlink,
    #                  search_type='qwen_clueweb22',
    #                  api_key=x_api_key,
    #                  verify_result=verify_result,
    #                  cw22_a=cw22_a)

    # # Return 401 if verification failed
    # if not verify_result:
    #     raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    try:
        # Execute search asynchronously using thread pools
        loop = asyncio.get_event_loop()

        # Use semaphore to control GPU encoding concurrency
        async with ENCODE_SEMAPHORE:
            result = await loop.run_in_executor(
                CPU_THREAD_POOL,
                search_with_pools,
                query, k, complexity, num_of_shards,
                with_distance, with_outlink, cw22_a
            )

        return result
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def health_priority(request: Request, call_next):
    if request.url.path == "/health":
        return JSONResponse({"status": "healthy"})
    return await call_next(request)


if __name__ == "__main__":
    # Get the machine's IP address
    ip_address = get_ip_address()
    port = PORT

    # Print the service information before starting
    print(f"\n======== Search Service Information ========")
    print(f"Starting ClueWeb22 search service")
    print(f"The service will be accessible at: http://{ip_address}:{port}")
    print(f"API documentation will be available at: http://{ip_address}:{port}/docs")
    print(f"============================================\n")

    # Run the server with uvicorn
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop="uvloop",  # Use high-performance event loop
        http="httptools",  # Use fast HTTP parser
        workers=1  # Single process with high concurrency
    )
