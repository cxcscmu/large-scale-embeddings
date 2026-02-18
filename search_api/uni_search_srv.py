import asyncio
import base64
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import uvloop
from fastapi import FastAPI, Query, HTTPException, Header, Request
from fastapi.responses import JSONResponse

from auth.auth_db import *
from cw22_searcher import ClueWeb22Searcher
from fw_searcher import FineWebSearcher
from sqlite_db.search_logger import init_search_logger, log_search_async
from utils.doc_reranker import DocReranker
from utils.performance_monitor import PerformanceMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set uvloop as event loop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global variables
cw22_engine = None
fw_engine = None

doc_reranker = None

SEARCH_POOL = None
IO_POOL = None
ENCODE_SEMAPHORE = None

# Performance monitor
PERF_MONITOR = None


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
    global cw22_engine, fw_engine, doc_reranker, SEARCH_POOL, IO_POOL, ENCODE_SEMAPHORE, PERF_MONITOR

    PERF_MONITOR = PerformanceMonitor(service_name="DR API", report_interval=60, window_size=1000)
    PERF_MONITOR.start()

    logger.info("Initializing thread pools...")
    # CPU-intensive tasks: use most cores for vector search and document processing
    SEARCH_POOL = ThreadPoolExecutor(max_workers=12, thread_name_prefix="CPU-Worker")
    # I/O intensive tasks: network requests, file reading
    IO_POOL = ThreadPoolExecutor(max_workers=20, thread_name_prefix="IO-Worker")
    # Limit GPU encoding to prevent resource contention
    ENCODE_SEMAPHORE = asyncio.Semaphore(1000)

    logger.info("Initializing auth db...")
    init_auth()

    logger.info("Initializing search logger...")
    init_search_logger()

    logger.info("Initializing searcher and reranker...")
    cw22_engine = ClueWeb22Searcher(enable_monitoring=True)
    fw_engine = FineWebSearcher(enable_monitoring=False)

    doc_reranker = DocReranker()

    # Display service information with actual IP
    ip_address = get_ip_address()
    port = 51000
    logger.info(f"Search service is accessible at: http://{ip_address}:{port}")
    logger.info(f"API documentation available at: http://{ip_address}:{port}/docs")

    yield

    # Clean up resources if needed on shutdown
    logger.info("Shutting down search service...")

    PERF_MONITOR.stop()
    SEARCH_POOL.shutdown(wait=True)
    IO_POOL.shutdown(wait=True)


# Initialize FastAPI app
app = FastAPI(
    title="ClueWeb22 Search API",
    description="API for searching ClueWeb22 documents",
    version="1.0.0",
    lifespan=lifespan
)


def search_with_pools(query, k, complexity, shard_ids,
                      with_distance, with_outlink, cw22_a, rerank=False):
    """
    Execute search using optimized thread pools for different types of work
    """
    global cw22_engine, doc_reranker, IO_POOL, PERF_MONITOR

    # Perform search (this will use internal thread pools)
    with PERF_MONITOR.measure(f"cw22_engine_search[shards={len(shard_ids)},k={k}]"):
        docs = cw22_engine.search(
            query,
            k=k,
            complexity=complexity,
            shard_ids=shard_ids,
            with_distance=with_distance
        )
    # Process results
    if with_distance:
        docids = [docid for _, docid in docs]
        distances = [dist for dist, _ in docs]
    else:
        docids = docs
        distances = None

    # Use I/O thread pool for document retrieval
    with PERF_MONITOR.measure(f"cw22_doc_retrieval[count={len(docids)}]"):
        doc_texts = cw22_engine.get_doc_texts_parallel(docids, IO_POOL)

    if rerank:
        with PERF_MONITOR.measure(f"cw22_doc_reranking[count={len(docids)}]"):
            doc_texts = doc_reranker.rerank(doc_texts=doc_texts, query=query)

    # Base64 encoding
    with PERF_MONITOR.measure(f"base64_encoding[count={len(doc_texts)}]"):
        json_bytes_list = [get_base64(doc_text) for doc_text in doc_texts]

    doc_outlinks = []
    if with_outlink:
        with PERF_MONITOR.measure(f"outlink_retrieval[count={len(docids)}]"):
            outlinks = cw22_engine.get_outlinks_parallel(docids, IO_POOL)
        with PERF_MONITOR.measure(f"outlink_encoding[count={len(outlinks)}]"):
            doc_outlinks = [get_base64(outlink) for outlink in outlinks]

    return {
        "results": json_bytes_list,
        "distances": distances,
        "outlinks": doc_outlinks
    }


def parse_cw_shard_ids(shard_ids_str: Optional[str], max_shards: int) -> List[int]:
    """
    Parse shard IDs and return a validated list of shard indices.
    If shard_ids_str is None, return all available shards.
    """
    # If no shard_ids provided, use all available shards
    if shard_ids_str is None:
        return list(range(max_shards))

    try:
        shard_list = [int(x.strip()) for x in shard_ids_str.split(',')]
        # Remove duplicates while preserving order
        seen = set()
        unique_shards = []
        for shard in shard_list:
            if shard not in seen:
                seen.add(shard)
                unique_shards.append(shard)

        # Validate range [0, max_shards-1]
        invalid_shards = [s for s in unique_shards if s < 0 or s >= max_shards]
        if invalid_shards:
            raise ValueError(f"Invalid shard IDs: {invalid_shards}. Must be in range [0, {max_shards - 1}]")
        return unique_shards
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid shard_ids: {str(e)}")


@app.get("/search")
async def search(
        request: Request,
        query: str = Query(..., description="Search query text"),
        k: Optional[int] = Query(10, description="Number of results to return"),
        complexity: Optional[int] = Query(None, description="Search complexity parameter"),
        shard_ids: Optional[str] = Query(None,
                                         description="Comma-separated shard IDs (e.g., '0,2,5'). If not provided, searches all available shards"),
        with_distance: Optional[bool] = Query(False,
                                              description="Whether to include document similarity scores in response"),
        with_outlink: Optional[bool] = Query(False, description="Whether to include document outlinks in response"),
        cw22_a: Optional[bool] = Query(False,
                                       description="Whether to search all 40 shards (CW22-A) instead of 4 shards (CW22-B)"),
        original_query: Optional[str] = Query(None, description="Original search query text for tracking purposes"),
        iteration: Optional[int] = Query(None, description="DeepResearch iteration number for tracking purposes"),
        rerank: Optional[bool] = Query(False, description="Whether enable document reranking"),
        x_api_key: Optional[str] = Header(None)
):
    """
    Search endpoint that returns document results for a given query from ClueWeb22 dataset.
    """
    global cw22_engine, PERF_MONITOR

    # Track overall request time
    request_start_time = time.time()

    with PERF_MONITOR.measure("cw22_request_total"):
        if not query:
            raise HTTPException(status_code=400, detail="No query text in request")

        if isinstance(k, str):
            k = int(k)

        max_shards = 40 if cw22_a else 4

        with PERF_MONITOR.measure("parse_shard_ids"):
            target_shards = parse_cw_shard_ids(shard_ids, max_shards)
            num_of_shards = len(target_shards)

        # Log search request
        client_ip = request.client.host
        x_api_key = x_api_key or ''

        # API key verification
        with PERF_MONITOR.measure("verify_api_key"):
            verify_result = verify_api_key_exists(x_api_key, dataset="clueweb")

        # Async log search request
        with PERF_MONITOR.measure("log_search_async"):
            log_search_async(ip_address=client_ip,
                             query_text=query,
                             k=k,
                             complexity=complexity,
                             num_of_shards=num_of_shards,
                             with_distance=with_distance,
                             with_outlink=with_outlink,
                             search_type='clueweb22',
                             api_key=x_api_key,
                             verify_result=verify_result,
                             cw22_a=cw22_a,
                             shard_ids=shard_ids,
                             original_query=original_query,
                             iteration=iteration)

        # Return 401 if verification failed
        if not verify_result:
            raise HTTPException(status_code=401,
                                detail="Invalid or missing API Key, please contact deepresearchgym@cmu.edu to request a free key.")

        try:
            # Execute search asynchronously using thread pools
            loop = asyncio.get_event_loop()

            # Track semaphore acquisition time
            semaphore_start = time.time()
            async with ENCODE_SEMAPHORE:
                semaphore_wait_time = time.time() - semaphore_start

                # Log if semaphore wait time is significant
                if semaphore_wait_time > 0.1:
                    with PERF_MONITOR.measure(f"semaphore_wait[{semaphore_wait_time:.2f}s]"):
                        pass

                # Execute the actual search
                with PERF_MONITOR.measure("executor_run"):
                    result = await loop.run_in_executor(
                        SEARCH_POOL,
                        search_with_pools,
                        query, k, complexity, target_shards,
                        with_distance, with_outlink, cw22_a, rerank
                    )

            # Log slow requests
            total_time = time.time() - request_start_time
            if total_time > 60.0:
                logger.warning(f"Slow request detected: {total_time:.2f}s for query: {query[:50]}...")

            return result

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


def fineweb_search_with_pools(query, k, complexity, num_of_shards, with_distance):
    """
    Execute FineWeb search using optimized thread pools for different types of work
    """
    global fw_engine, IO_POOL, PERF_MONITOR

    # Perform search (this will use internal thread pools)
    with PERF_MONITOR.measure(f"fw_engine_search[shards={num_of_shards},k={k}]"):
        docs = fw_engine.search(
            query, k=k, complexity=complexity, num_of_shards=num_of_shards,
            with_distance=with_distance
        )

    # Process results
    if with_distance:
        docids = [docid for _, docid in docs]
        distances = [dist for dist, _ in docs]
    else:
        docids = docs
        distances = None

    # Use I/O thread pool for FineWeb document retrieval
    with PERF_MONITOR.measure(f"fw_doc_retrieval[count={len(docids)}]"):
        doc_texts = fw_engine.get_doc_texts_parallel(docids, IO_POOL)

    with PERF_MONITOR.measure(f"fw_base64_encoding[count={len(doc_texts)}]"):
        json_bytes_list = [get_base64(doc_text) for doc_text in doc_texts]

    return {
        "results": json_bytes_list,
        "distances": distances,
        "outlinks": None  # FineWeb doesn't have outlinks
    }


@app.get("/fineweb/search")
async def fineweb_search(
        request: Request,
        query: str = Query(..., description="Search query text"),
        k: Optional[int] = Query(10, description="Number of results to return"),
        complexity: Optional[int] = Query(None, description="Searche complexity"),
        num_of_shards: Optional[int] = Query(7, description="Number of shards to use for search"),
        with_distance: Optional[bool] = Query(False, description="Whether with document distance"),
        original_query: Optional[str] = Query(None, description="Original search query text for tracking purposes"),
        iteration: Optional[int] = Query(None, description="DeepResearch iteration number for tracking purposes"),
        x_api_key: Optional[str] = Header(None)
):
    """
    Search endpoint that returns document results for a given query
    """
    global fw_engine, PERF_MONITOR

    # Track overall request time
    request_start_time = time.time()

    with PERF_MONITOR.measure("fw_request_total"):
        if not query:
            raise HTTPException(status_code=400, detail="No query text in request")

        if isinstance(k, str):
            k = int(k)

        if isinstance(num_of_shards, str):
            num_of_shards = int(num_of_shards)

        # Validate num_of_shards is in valid range
        if num_of_shards is not None and (num_of_shards < 1 or num_of_shards > 7):
            logger.warning(f"Invalid num_of_shards: {num_of_shards}, must be between 1 and 7")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid num_of_shards: {num_of_shards}. Value must be between 1 and 7."
            )

        x_api_key = x_api_key or ''

        with PERF_MONITOR.measure("fw_verify_api_key"):
            verify_result = verify_api_key_exists(x_api_key, dataset="fineweb")

        # Log search request asynchronously
        client_ip = request.client.host
        with PERF_MONITOR.measure("fw_log_search_async"):
            log_search_async(ip_address=client_ip,
                             query_text=query,
                             k=k,
                             complexity=complexity,
                             num_of_shards=num_of_shards,
                             with_distance=with_distance,
                             with_outlink=False,
                             search_type='fineweb',
                             api_key=x_api_key,
                             verify_result=verify_result,
                             original_query=original_query,
                             iteration=iteration)

        # Return 401 if verification failed
        if not verify_result:
            raise HTTPException(status_code=401,
                                detail="Invalid or missing API Key, please contact deepresearchgym@cmu.edu to request a free key.")

        try:
            # Execute search asynchronously using thread pools
            loop = asyncio.get_event_loop()

            # Track semaphore acquisition time
            semaphore_start = time.time()
            async with ENCODE_SEMAPHORE:
                semaphore_wait_time = time.time() - semaphore_start

                # Log if semaphore wait time is significant
                if semaphore_wait_time > 0.1:
                    with PERF_MONITOR.measure(f"fw_semaphore_wait[{semaphore_wait_time:.2f}s]"):
                        pass

                with PERF_MONITOR.measure("fw_executor_run"):
                    result = await loop.run_in_executor(
                        SEARCH_POOL,
                        fineweb_search_with_pools,
                        query, k, complexity, num_of_shards, with_distance
                    )

            # Log slow requests
            total_time = time.time() - request_start_time
            if total_time > 60.0:
                logger.warning(f"Slow FineWeb request: {total_time:.2f}s for query: {query[:50]}...")

            return result

        except Exception as e:
            logger.error(f"Error in fw_engine: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def health_priority(request: Request, call_next):
    if request.url.path == "/health":
        return JSONResponse({"status": "healthy"})
    return await call_next(request)


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
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop="uvloop",  # Use high-performance event loop
        http="httptools",  # Use fast HTTP parser
        workers=1  # Single process with high concurrency
    )
