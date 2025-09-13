#!/usr/bin/env python3
"""
File-based API Key Management Tool
"""

import sys
import secrets
from auth_db import init_auth

def generate_api_key():
    """Generate a secure random API key"""
    return secrets.token_urlsafe(32)

def main():
    try:
        # Initialize with short check interval for immediate testing (normally 600s)
        auth_manager = init_auth(check_interval=10)  # 10 seconds for demo
    except Exception as e:
        print(f"Failed to initialize auth manager: {e}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python file_key_manager.py [list|add|delete|generate|enable|disable|status]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "list":
        keys = auth_manager.list_api_keys()
        if not keys:
            print("No API keys found.")
        else:
            print(f"{'API Key':<45} {'Enabled Time':<20} {'User Info':<20} {'Active'}")
            print("-" * 90)
            for key in keys:
                status = "Yes" if key['is_active'] else "No"
                print(f"{key['api_key']:<45} {key['enabled_time']:<20} {key['user_info']:<20} {status}")
    
    elif command == "add":
        if len(sys.argv) < 3:
            print("Usage: python file_key_manager.py add <user_info> [custom_key]")
            sys.exit(1)
        
        user_info = sys.argv[2]
        api_key = sys.argv[3] if len(sys.argv) > 3 else generate_api_key()
        
        if auth_manager.add_api_key(api_key, user_info):
            print(f"API key added successfully:")
            print(f"Key: {api_key}")
            print(f"User: {user_info}")
            print("Key saved to api_keys.json")
        else:
            print("Failed to add API key (key may already exist)")
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python file_key_manager.py delete <api_key>")
            sys.exit(1)
        
        api_key = sys.argv[2]
        if auth_manager.delete_api_key(api_key):
            print(f"API key deleted: {api_key}")
        else:
            print("API key not found")
    
    elif command == "enable":
        if len(sys.argv) < 3:
            print("Usage: python file_key_manager.py enable <api_key>")
            sys.exit(1)
        
        api_key = sys.argv[2]
        if auth_manager.toggle_api_key(api_key, True):
            print(f"API key enabled: {api_key}")
        else:
            print("API key not found")
    
    elif command == "disable":
        if len(sys.argv) < 3:
            print("Usage: python file_key_manager.py disable <api_key>")
            sys.exit(1)
        
        api_key = sys.argv[2]
        if auth_manager.toggle_api_key(api_key, False):
            print(f"API key disabled: {api_key}")
        else:
            print("API key not found")
    
    elif command == "generate":
        print(f"Generated API key: {generate_api_key()}")
    
    elif command == "status":
        info = auth_manager.get_cache_info()
        print("=== Auth Manager Status ===")
        print(f"Active keys in cache: {info['active_keys']}")
        print(f"Last file check: {info['last_check']}")
        print(f"Background monitor: {'Running' if info['monitor_running'] else 'Stopped'}")
    
    else:
        print("Unknown command. Use: list, add, delete, enable, disable, generate, or status")
    
    # Stop monitor thread before exit
    auth_manager.stop_monitor()

if __name__ == "__main__":
    main()