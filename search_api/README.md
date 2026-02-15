# Search API

A distributed semantic search service for large-scale web document retrieval, supporting ClueWeb22 and FineWeb datasets with neural embedding-based search capabilities.

## Overview

This search API provides a production-ready, distributed search infrastructure for querying billions of web documents. The system uses neural embeddings (MiniCPM-Embedding-Light) for semantic search and implements a shard-based architecture to enable efficient retrieval across massive document collections. The service is designed for high-throughput concurrent queries with intelligent batch processing and GPU acceleration.

### Key Features

- **Distributed Architecture**: Multi-node sharding system (4 nodes for ClueWeb22, 7 nodes for FineWeb)
- **Neural Semantic Search**: Embedding-based retrieval using state-of-the-art sentence transformers
- **Batch Query Processing**: Automatic batching of concurrent requests for optimal GPU utilization
- **Concurrent Search**: Parallel querying across shards with result aggregation and re-ranking
- **API Authentication**: File-based API key management with hot-reload capabilities
- **Performance Monitoring**: Built-in logging and benchmarking tools
- **HPC Integration**: SLURM job scripts for cluster deployment

## System Architecture

The system follows a three-tier architecture:

1. **API Layer**: FastAPI-based REST endpoints for search requests
2. **Search Coordination Layer**: Query encoding and distributed search orchestration
3. **Storage Layer**: Distributed document shards with FAISS vector indices

Query flow: `User Query → Query Encoder → Parallel Shard Search → Result Aggregation → Document Retrieval`

## Directory Structure

### Core Services

- **`search_service.py`**: Main FastAPI service providing the `/search` endpoint for ClueWeb22 queries
- **`uni_search_srv.py`**: Unified search service supporting multiple datasets
- **`cw22_search_srv.py`**: ClueWeb22-specific search service with enhanced logging
- **`cw22_qwen_search_srv.py`**: Alternative search service using Qwen-based encoders
- **`fetch_srv.py`**: Document fetching service for retrieving full document content

### Searcher Implementations

- **`cw22_searcher.py`**: Core searcher class for ClueWeb22 dataset
  - Distributed search coordination across 4 shards
  - Query encoding and embedding management
  - Document ID translation and text retrieval
  - Support for both sequential and parallel shard queries
  
- **`cw22_qwen_searcher.py`**: Qwen model-based variant of ClueWeb22 searcher
- **`fw_searcher.py`**: FineWeb dataset searcher with 7-shard architecture

### Distributed Search Nodes

#### `cw22_search_api/`
ClueWeb22 search nodes (4 shards):
- `cw22_node{0-3}.py`: Individual node services running on separate machines
- `cw22_node_generic.py`: Generic node template for easy deployment
- `start_node{0-3}.sbatch`: SLURM job scripts for cluster deployment
- Each node hosts a portion of the ClueWeb22 index and serves vector search requests

#### `fw_search_api/`
FineWeb search nodes (7 shards):
- `fw_node{0-6}.py`: Individual node services for FineWeb dataset
- `start_node{0-6}.sbatch`: SLURM deployment scripts
- Similar architecture to ClueWeb22 but with more granular sharding

### Utilities

#### `utils/`
Core utility modules:

- **`query_encoder.py`**: Thread-safe batch query encoder
  - Automatic batching of concurrent encoding requests
  - Configurable batch size and timeout
  - GPU acceleration with optional Flash Attention
  - Statistics tracking for performance monitoring

- **`cw22_files.py`**: ClueWeb22 document storage and retrieval
  - Efficient document lookup by ID
  - Compressed document storage handling

- **`fineweb_files.py`**: FineWeb dataset document management

- **`cw22_outlinks.py`**: ClueWeb22 outlink graph utilities

- **`cw22_api.py`**: Helper functions for ClueWeb22 API interactions

- **`benchmark_search_rps.py`**: Performance benchmarking tool for measuring requests-per-second

#### `auth/`
Authentication and access control:

- **`auth_db.py`**: File-based API key authentication
  - JSON file storage for API keys
  - Hot-reload with background monitoring (10-minute intervals)
  - Thread-safe key cache for fast verification
  - Key management: add, delete, toggle, list operations

- **`auth_key_manager.py`**: Command-line tool for API key administration

#### `relevance_judgement/`
Search quality evaluation:

- **`relevance_metric.py`**: Relevance metrics and evaluation tools for assessing search quality

#### `sqlite_db/`
Logging infrastructure:

- **`search_logger.py`**: SQLite-based query logging for analytics and debugging

### Deployment Scripts

#### SLURM Scripts (`slurm_scripts/`)
- `run_trec_eval.sbatch`: TREC evaluation runner
- `run_rps_benchmark.sbatch`: RPS benchmarking job
- `compute_gt_files.sbatch`: Ground truth file generation
- `playground.sbatch`: Testing and development environment

#### Startup Scripts
- **`start_all_nodes_cw22.sh`**: Launch all ClueWeb22 search nodes
- **`start_all_nodes_fw.sh`**: Launch all FineWeb search nodes
- **`auto_uni_search.sh`**: Automated unified search service startup
- **`launch_searcher.sbatch`**: Generic searcher launch script

## API Endpoints

### Main Search Endpoint
```
POST/GET /search
Parameters:
  - query (str): Search query text
  - k (int): Number of results to return (default: 10)
  - num_of_shards (int): Number of shards to query (1-4, default: 4)

Returns:
  - results (list): Base64-encoded document contents
```

### Health Check
```
GET /health
Returns: Service health status
```

### Mock Search (Testing)
```
GET /mock-search
Parameters: Same as /search
Returns: Mock results for testing without backend
```

## Technical Details

- **Embedding Model**: MiniCPM-Embedding-Light (SentenceTransformer)
- **Vector Search**: FAISS-based approximate nearest neighbor search
- **Concurrency**: ThreadPoolExecutor for parallel shard queries
- **Batch Processing**: Automatic query batching with configurable timeout (default: 50ms)
- **Document Format**: Base64-encoded JSON for efficient transmission

## Performance

- Supports concurrent queries with intelligent batching
- Configurable batch size (default: 32 queries)
- Parallel shard querying for reduced latency
- GPU acceleration for query encoding
- Built-in RPS benchmarking tools

---

For deployment and usage instructions, please refer to the individual service scripts and SLURM job configurations.
