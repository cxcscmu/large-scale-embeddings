import sqlite3
import os
import threading
import queue
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = "/bos/usr0/jening/PycharmProjects/DiskANN_Search/sqlite_db/search_logs.db"

class SearchLogger:
    def __init__(self, db_path=DB_PATH):
        """Initialize SQLite search logger with background thread"""
        self.db_path = db_path
        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.stop_flag = False
        
        # Create database directory if not exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Start background worker
        self._start_worker()
        
        logger.info(f"Search logger initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize database and create tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                query_text TEXT NOT NULL,
                k INTEGER,
                complexity INTEGER,
                num_of_shards INTEGER,
                with_distance BOOLEAN,
                with_outlink BOOLEAN,
                search_type TEXT DEFAULT 'clueweb22',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _worker_thread_func(self):
        """Background worker thread for database writes"""
        while not self.stop_flag:
            try:
                # Get log entry from queue (blocking with timeout)
                log_entry = self.log_queue.get(timeout=1)
                
                # Write to database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO search_logs 
                    (ip_address, query_text, k, complexity, num_of_shards, with_distance, with_outlink, search_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', log_entry)
                
                conn.commit()
                conn.close()
                
                # Mark task as done
                self.log_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check stop flag
            except Exception as e:
                logger.error(f"Database write error: {e}")
    
    def _start_worker(self):
        """Start background worker thread"""
        self.worker_thread = threading.Thread(target=self._worker_thread_func, daemon=True)
        self.worker_thread.start()
    
    def log_search(self, ip_address: str, query_text: str, k: int, 
                   complexity: Optional[int], num_of_shards: int, 
                   with_distance: bool, with_outlink: bool, 
                   search_type: str = 'clueweb22'):
        """Add search log entry to queue (non-blocking)"""
        log_entry = (
            ip_address, query_text, k, complexity, 
            num_of_shards, with_distance, with_outlink, search_type
        )
        self.log_queue.put(log_entry)
    
    def stop(self):
        """Stop the logger and wait for remaining logs to be written"""
        self.stop_flag = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info("Search logger stopped")

# Global search logger instance
search_logger = None

def init_search_logger(db_path=DB_PATH):
    """Initialize the global search logger"""
    global search_logger
    search_logger = SearchLogger(db_path)
    return search_logger

def log_search_async(ip_address: str, query_text: str, k: int, 
                     complexity: Optional[int], num_of_shards: int, 
                     with_distance: bool, with_outlink: bool, 
                     search_type: str = 'clueweb22'):
    """Log search entry asynchronously"""
    global search_logger

    if complexity is None:
        complexity = 5 * k

    if search_logger:
        search_logger.log_search(
            ip_address, query_text, k, complexity, 
            num_of_shards, with_distance, with_outlink, search_type
        )