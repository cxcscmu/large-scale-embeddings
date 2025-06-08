import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from tqdm import tqdm


def benchmark_search_rps(n=100, k=100, search_url=None, parallel=False, num_threads=10, query_length=50):
    """
    Benchmark search performance by running multiple queries and measuring detailed timing statistics.
    
    Args:
        n (int): Number of query texts to generate and test
        k (int): Number of results to retrieve for each query
        search_url (str): The URL endpoint for the search API
        parallel (bool): Whether to use parallel processing for search requests
        num_threads (int): Number of threads to use for parallel processing
        query_length (int): Approximate length of randomly generated queries
        
    Returns:
        dict: Dictionary containing timing statistics including percentiles
    """
    print("######################################################################")

    def generate_random_text(approx_length=50):
        """Generate a random text string of approximate length."""
        words = []
        current_length = 0

        while current_length < approx_length:
            word_length = random.randint(3, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
            current_length += word_length + 1  # +1 for space

        return ' '.join(words)

    def perform_search(query):
        """Perform a search with the given query and measure time."""
        start_time = time.time()

        # If search_url is provided, use HTTP request, otherwise use a mock search
        if search_url:
            params = {"query": query, "k": k}
            response = requests.get(search_url, params=params)
            # Assuming the API returns results directly
            results = response.json()
        else:
            # Mock search for testing without an actual API
            time.sleep(random.uniform(0.01, 0.1))  # Simulate search time
            results = [f"Result {i} for {query}" for i in range(k)]

        end_time = time.time()
        return end_time - start_time

    # Generate n random queries
    print(f"Generating {n} random queries...")
    queries = [generate_random_text(query_length) for _ in range(n)]

    # Perform searches and measure time
    print(f"Performing {n} searches (k={k})...")
    search_times = []

    if parallel:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all search tasks
            future_to_query = {executor.submit(perform_search, query): query for query in queries}

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_query), total=n):
                search_time = future.result()
                search_times.append(search_time)
    else:
        for query in tqdm(queries):
            search_time = perform_search(query)
            search_times.append(search_time)

    # Calculate statistics
    total_time = sum(search_times)
    avg_time = total_time / n
    min_time = min(search_times)
    max_time = max(search_times)

    # Calculate percentiles
    percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    percentile_values = np.percentile(search_times, percentiles)

    # Create percentile dictionary
    percentile_dict = {f"p{p}": value for p, value in zip(percentiles, percentile_values)}

    # Display results
    print(f"Search Performance Results for n={n} k={k}:")
    print(f"  Total time for {n} searches: {total_time:.4f} seconds")
    print(f"  Average search time per query: {avg_time:.4f} seconds")
    print(f"  Minimum search time: {min_time:.4f} seconds")
    print(f"  Maximum search time: {max_time:.4f} seconds")

    print("\nPercentile Breakdown (seconds):")
    for p in percentiles:
        print(f"P{p}\t{percentile_dict[f'p{p}']:.4f}")

    # Return comprehensive stats
    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "percentiles": percentile_dict,
        "all_times": search_times
    }


# Example usage
if __name__ == "__main__":
    # stats = benchmark_search_rps(n=100, k=10, search_url="https://your-search-api.com/search")

    search_url = "https://www.clueweb22.us/search"
    # search_url = "http://10.1.1.40:51000/search"
    # stats = benchmark_search_rps(n=2000, k=1, search_url=search_url)
    # stats = benchmark_search_rps(n=2000, k=5, search_url=search_url)
    stats = benchmark_search_rps(n=1000, k=10, search_url=search_url)
    # stats = benchmark_search_rps(n=2000, k=25, search_url=search_url)
    # stats = benchmark_search_rps(n=2000, k=50, search_url=search_url)
    # stats = benchmark_search_rps(n=2000, k=100, search_url=search_url)

    # import matplotlib.pyplot as plt
    # plt.hist(stats["all_times"], bins=20)
    # plt.title("Distribution of Search Times")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Frequency")
    # plt.show()
