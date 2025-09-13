#!/bin/bash
#SBATCH --job-name=uni_search
#SBATCH --nodelist=boston-1-7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --mem=144G
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=30
#SBATCH --time=infinite
#SBATCH --output=node_log/auto_searcher_%j.log
#SBATCH --error=node_log/auto_searcher_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="ethanning@cmu.edu"

echo "Starting Self-Healing Uni Search Service at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"

cd /bos/usr0/jening/PycharmProjects/DiskANN_Search
source ~/.bashrc
conda activate encode

# Configuration
LOCAL_HEALTH_URL="http://localhost:51000/health"
EXTERNAL_HEALTH_URL="https://clueweb22.us/health"
HEALTH_CHECK_INTERVAL=600
MAX_LOCAL_FAILURES=4   # Max consecutive local service failures before restart
MAX_TUNNEL_FAILURES=2  # Max consecutive tunnel failures before restart
RESTART_SCRIPT="/bos/usr0/jening/PycharmProjects/DiskANN_Search/auto_uni_search.sh"

# Global variables
SEARCH_SRV_PID=""
TUNNEL_PID=""
LOCAL_FAILURE_COUNT=0
TUNNEL_FAILURE_COUNT=0
LAST_HEALTH_CHECK=0

# Function to cleanup processes on exit
cleanup() {
    echo "$(date): Cleaning up processes..."
    kill $SEARCH_SRV_PID $TUNNEL_PID 2>/dev/null
    wait
}
trap cleanup EXIT

# Function to start search service
start_search_service() {
    echo "$(date): Starting search service..."
    nohup python3 -u uni_search_srv.py > node_log/search_srv_${SLURM_JOB_ID}.log 2>&1 &
    SEARCH_SRV_PID=$!
    echo "$(date): Search service PID: $SEARCH_SRV_PID"
    
    # Service takes up to 5 minutes to start, so we just log and continue
    echo "$(date): Search service will take approximately 5 minutes to fully initialize"
    return 0
}

# Function to start tunnel
start_tunnel() {
    echo "$(date): Starting cloudflared tunnel..."
    nohup ~/bin/cloudflared tunnel run clueweb22-tunnel > node_log/tunnel_${SLURM_JOB_ID}.log 2>&1 &
    TUNNEL_PID=$!
    echo "$(date): Tunnel PID: $TUNNEL_PID"
    
    # Give tunnel time to establish connection
    sleep 20
}

# Function to check health and take appropriate action
check_health() {
    echo "$(date): Starting health check cycle..."
    
    # First check local service health
    local local_response
    echo "$(date): Checking local service at $LOCAL_HEALTH_URL..."
    local_response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 180 --max-time 180 "$LOCAL_HEALTH_URL" 2>/dev/null)
    
    echo "$(date): Local service returned HTTP code: $local_response"
    
    if [ "$local_response" != "200" ]; then
        LOCAL_FAILURE_COUNT=$((LOCAL_FAILURE_COUNT + 1))
        echo "$(date): Local service health check failed ($LOCAL_FAILURE_COUNT/$MAX_LOCAL_FAILURES): HTTP $local_response"
        
        # If local service has failed too many times, restart entire job
        if [ $LOCAL_FAILURE_COUNT -ge $MAX_LOCAL_FAILURES ]; then
            echo "$(date): Local service has failed $MAX_LOCAL_FAILURES times, restarting entire job..."
            restart_job
            return 1
        fi
    else
        # echo "$(date): Local service health check passed"
        LOCAL_FAILURE_COUNT=0
        
        # Only check external tunnel if local service is healthy
        local external_response
        echo "$(date): Checking external tunnel at $EXTERNAL_HEALTH_URL..."
        external_response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 180 --max-time 180 "$EXTERNAL_HEALTH_URL" 2>/dev/null)
        
        echo "$(date): External tunnel returned HTTP code: $external_response"
        
        if [ "$external_response" = "200" ]; then
            # echo "$(date): External tunnel health check passed"
            TUNNEL_FAILURE_COUNT=0
            return 0
        else
            TUNNEL_FAILURE_COUNT=$((TUNNEL_FAILURE_COUNT + 1))
            echo "$(date): External tunnel health check failed ($TUNNEL_FAILURE_COUNT/$MAX_TUNNEL_FAILURES): HTTP $external_response"
            
            # If tunnel has failed too many times, restart it
            if [ $TUNNEL_FAILURE_COUNT -ge $MAX_TUNNEL_FAILURES ]; then
                echo "$(date): Tunnel has failed $MAX_TUNNEL_FAILURES times, restarting tunnel only..."
                
                # Kill old tunnel
                if is_process_running "$TUNNEL_PID"; then
                    kill $TUNNEL_PID 2>/dev/null
                    sleep 2
                fi
                
                # Start new tunnel
                start_tunnel
                TUNNEL_FAILURE_COUNT=0
            fi
        fi
    fi
    
    return 0
}

# Function to restart entire job
restart_job() {
    echo "$(date): Restarting entire job..."
    echo "$(date): Current job ID: $SLURM_JOB_ID"
    
    # Submit new job before cancelling current one
    NEW_JOB_ID=$(sbatch --parsable "$RESTART_SCRIPT")
    if [ $? -eq 0 ]; then
        echo "$(date): Submitted new job: $NEW_JOB_ID"
        sleep 5
        
        # Cancel current job
        scancel $SLURM_JOB_ID
        exit 0
    else
        echo "$(date): Failed to submit new job, will continue monitoring..."
        # Reset failure count to avoid immediate retry
        LOCAL_FAILURE_COUNT=0
        return 1
    fi
}

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Start initial services
start_search_service

# Wait before starting health checks
echo "$(date): Waiting 10 minutes for search service to initialize before starting health checks..."
sleep 600

# Wait a bit before starting tunnel
echo "Starting tunnel"
start_tunnel

echo "$(date): All services started. Beginning monitoring loop..."

# Main monitoring loop
while true; do
    # Check if search service process is still running
    # if ! is_process_running "$SEARCH_SRV_PID"; then
    #     echo "$(date): Search service process died unexpectedly (PID: $SEARCH_SRV_PID)"
    #     # Process death is critical, but we still apply the failure count logic
    #     LOCAL_FAILURE_COUNT=$((LOCAL_FAILURE_COUNT + 1))
        
    #     if [ $LOCAL_FAILURE_COUNT -ge $MAX_LOCAL_FAILURES ]; then
    #         echo "$(date): Search service process has died $LOCAL_FAILURE_COUNT times, restarting entire job..."
    #         restart_job
    #     else
    #         echo "$(date): Attempting to restart search service locally (attempt $LOCAL_FAILURE_COUNT/$MAX_LOCAL_FAILURES)..."
    #         start_search_service
    #         # Give it time to start
    #         echo "$(date): Waiting 10 minutes for service restart..."
    #         sleep 600
    #         echo "$(date): Service restart wait completed"
    #     fi
    # else
    #     echo "$(date): Search service process is running normally (PID: $SEARCH_SRV_PID)"
    # fi
    
    # Check if tunnel process is still running
    if ! is_process_running "$TUNNEL_PID"; then
        echo "$(date): Tunnel process died, restarting tunnel..."
        start_tunnel
    else
        echo "$(date): Tunnel process is running normally (PID: $TUNNEL_PID)"
    fi
    
    # Perform comprehensive health check every HEALTH_CHECK_INTERVAL seconds
    current_time=$(date +%s)
    if [ $((current_time - LAST_HEALTH_CHECK)) -ge $HEALTH_CHECK_INTERVAL ]; then
        echo "$(date): Time for scheduled health check (last check: $LAST_HEALTH_CHECK, current: $current_time)"
        LAST_HEALTH_CHECK=$current_time
        check_health
    else
        echo "$(date): Skipping health check - not yet time (will check in $((HEALTH_CHECK_INTERVAL - (current_time - LAST_HEALTH_CHECK))) seconds)"
    fi
    
    echo "$(date): Monitoring cycle complete - sleeping for 2 minutes..."
    # Sleep for 10 minute before next process check
    sleep 600
done