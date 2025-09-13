import json
import logging
import os
import threading
import time
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

GOLBAL_AUTH_DB = "/bos/usr0/jening/PycharmProjects/DiskANN_Search/auth/api_keys.json"

class FileAuth:
    def __init__(self, auth_file=GOLBAL_AUTH_DB, check_interval=600):  # 10 minutes
        """Initialize file-based auth with periodic file monitoring"""
        self.auth_file = auth_file
        self.cache = {}
        self.lock = threading.RLock()
        self.check_interval = check_interval  # seconds
        self.last_mtime = 0
        self.monitor_thread = None
        self.stop_monitoring = False
        
        self._load_keys()
        self._start_monitor()
        logger.info(f"File auth initialized with {auth_file}, checking every {check_interval//60} minutes")
    
    def _load_keys(self):
        """Load API keys from file to memory cache"""
        try:
            if os.path.exists(self.auth_file):
                # Update file modification time
                self.last_mtime = os.path.getmtime(self.auth_file)
                
                with open(self.auth_file, 'r') as f:
                    data = json.load(f)
                    with self.lock:
                        self.cache = {k: v for k, v in data.items() if v.get('is_active', False)}
                        
                logger.info(f"Loaded {len(self.cache)} active API keys")
            else:
                with self.lock:
                    self.cache = {}
                self._save_keys({})
        except Exception as e:
            logger.error(f"Failed to load auth file: {e}")
            with self.lock:
                self.cache = {}
    
    def _check_file_changed(self):
        """Check if auth file has been modified and reload if needed"""
        try:
            if os.path.exists(self.auth_file):
                current_mtime = os.path.getmtime(self.auth_file)
                if current_mtime > self.last_mtime:
                    logger.info("Auth file changed, reloading keys")
                    self._load_keys()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to check file changes: {e}")
            return False
    
    def _monitor_worker(self):
        """Background thread worker for periodic file monitoring"""
        while not self.stop_monitoring:
            try:
                self._check_file_changed()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitor worker: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _start_monitor(self):
        """Start background monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_monitoring = False
            self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            self.monitor_thread.start()
            logger.info("Started background file monitor")
    
    def stop_monitor(self):
        """Stop background monitoring thread"""
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        logger.info("Stopped background file monitor")
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key from memory cache (fast lookup)"""
        if not api_key:
            return False
        
        with self.lock:
            return api_key in self.cache
    
    def _save_keys(self, all_keys):
        """Save all keys to file"""
        try:
            with open(self.auth_file, 'w') as f:
                json.dump(all_keys, f, indent=2)
            # Update our own mtime after saving
            if os.path.exists(self.auth_file):
                self.last_mtime = os.path.getmtime(self.auth_file)
        except Exception as e:
            logger.error(f"Failed to save auth file: {e}")
    
    def _get_all_keys(self):
        """Get all keys from file (including inactive)"""
        try:
            if os.path.exists(self.auth_file):
                with open(self.auth_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to read auth file: {e}")
            return {}
    
    def add_api_key(self, api_key: str, user_info: str) -> bool:
        """Add new API key"""
        try:
            with self.lock:
                all_keys = self._get_all_keys()
                
                if api_key in all_keys:
                    return False  # Key already exists
                
                key_data = {
                    'enabled_time': datetime.now().isoformat(),
                    'user_info': user_info,
                    'is_active': True
                }
                
                all_keys[api_key] = key_data
                self._save_keys(all_keys)
                
                # Update cache immediately
                self.cache[api_key] = key_data
                
                return True
        except Exception as e:
            logger.error(f"Failed to add API key: {e}")
            return False
    
    def delete_api_key(self, api_key: str) -> bool:
        """Delete API key"""
        try:
            with self.lock:
                all_keys = self._get_all_keys()
                
                if api_key not in all_keys:
                    return False
                
                del all_keys[api_key]
                self._save_keys(all_keys)
                
                # Update cache immediately
                if api_key in self.cache:
                    del self.cache[api_key]
                
                return True
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            return False
    
    def list_api_keys(self) -> List[Dict]:
        """List all API keys from file (real-time)"""
        try:
            all_keys = self._get_all_keys()
            keys = []
            for api_key, data in all_keys.items():
                keys.append({
                    'api_key': api_key,
                    'enabled_time': data.get('enabled_time'),
                    'user_info': data.get('user_info'),
                    'is_active': data.get('is_active', False)
                })
            return sorted(keys, key=lambda x: x['enabled_time'], reverse=True)
        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            return []
    
    def toggle_api_key(self, api_key: str, active: bool) -> bool:
        """Enable/disable API key"""
        try:
            with self.lock:
                all_keys = self._get_all_keys()
                
                if api_key not in all_keys:
                    return False
                
                all_keys[api_key]['is_active'] = active
                self._save_keys(all_keys)
                
                # Update cache immediately
                if active:
                    self.cache[api_key] = all_keys[api_key]
                else:
                    if api_key in self.cache:
                        del self.cache[api_key]
                
                return True
        except Exception as e:
            logger.error(f"Failed to toggle API key: {e}")
            return False
    
    def reload_keys(self):
        """Manually reload keys from file"""
        logger.info("Manual reload requested")
        self._load_keys()
    
    def get_cache_info(self):
        """Get cache status info"""
        with self.lock:
            return {
                'active_keys': len(self.cache),
                'last_check': time.ctime(self.last_mtime) if self.last_mtime else 'Never',
                'monitor_running': self.monitor_thread and self.monitor_thread.is_alive()
            }

# Global auth instance
auth_manager = None

def init_auth(auth_file=GOLBAL_AUTH_DB, check_interval=60):
    """Initialize the global auth manager"""
    global auth_manager
    auth_manager = FileAuth(auth_file, check_interval)
    print(len(auth_manager.cache))
    return auth_manager

def verify_api_key_exists(api_key: str) -> bool:
    """Verify API key - compatible with original function"""
    global auth_manager
    if auth_manager is None:
        raise RuntimeError("Auth manager not initialized. Call init_auth() first.")
    return auth_manager.verify_api_key(api_key)