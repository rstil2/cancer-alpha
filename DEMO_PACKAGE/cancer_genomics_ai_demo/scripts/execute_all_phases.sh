#!/bin/bash

# Cancer Genomics AI Demo - Master Execution Script
# Runs Phase A (Real Data Integration) and Phase B (Production Deployment Scale-up)
# This script can execute phases sequentially or in parallel

set -euo pipefail

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
BATCH_MODE=false
TRACK_PROGRESS=false
EMAIL_NOTIFICATIONS=false
NOTIFICATION_EMAIL=""
PARALLEL_EXECUTION=false
PHASE_A_ONLY=false
PHASE_B_ONLY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)
            BATCH_MODE=true
            shift
            ;;
        --track-progress)
            TRACK_PROGRESS=true
            shift
            ;;
        --email)
            EMAIL_NOTIFICATIONS=true
            NOTIFICATION_EMAIL="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_EXECUTION=true
            shift
            ;;
        --phase-a-only)
            PHASE_A_ONLY=true
            shift
            ;;
        --phase-b-only)
            PHASE_B_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --batch              Run in batch mode (non-interactive)"
            echo "  --track-progress     Enable progress tracking with checkpoints"
            echo "  --email EMAIL        Send notifications to EMAIL address"
            echo "  --parallel           Run Phase A and B in parallel (faster but resource-intensive)"
            echo "  --phase-a-only       Run only Phase A: Real Data Integration"
            echo "  --phase-b-only       Run only Phase B: Production deployment Scale-up"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --batch --track-progress --email admin@hospital.com"
            echo "  $0 --parallel --batch"
            echo "  $0 --phase-a-only --track-progress"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate options
if [[ "$PHASE_A_ONLY" == "true" && "$PHASE_B_ONLY" == "true" ]]; then
    echo -e "${RED}Error: Cannot specify both --phase-a-only and --phase-b-only${NC}"
    exit 1
fi

if [[ "$PARALLEL_EXECUTION" == "true" && ("$PHASE_A_ONLY" == "true" || "$PHASE_B_ONLY" == "true") ]]; then
    echo -e "${RED}Error: Cannot use --parallel with --phase-a-only or --phase-b-only${NC}"
    exit 1
fi

# Initialize logging
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
MASTER_LOG="$LOG_DIR/master_execution_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$MASTER_LOG"
}

# Email notification function
send_master_notification() {
    if [[ "$EMAIL_NOTIFICATIONS" == "true" && -n "$NOTIFICATION_EMAIL" ]]; then
        local subject="$1"
        local message="$2"
        
        if command -v mail >/dev/null 2>&1; then
            echo "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL" || log "WARN" "Failed to send email notification"
        else
            log "WARN" "Mail command not available for notifications"
        fi
    fi
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites for Cancer Genomics AI Demo execution"
    
    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        log "ERROR" "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Docker (for containerized execution)
    if ! command -v docker >/dev/null 2>&1; then
        log "WARN" "Docker not found - some deployment features may not work"
    fi
    
    # Check Kubernetes CLI (for Phase B)
    if [[ "$PHASE_B_ONLY" == "true" || ("$PHASE_A_ONLY" != "true" && "$PHASE_B_ONLY" != "true") ]]; then
        if ! command -v kubectl >/dev/null 2>&1; then
            log "WARN" "kubectl not found - Phase B Kubernetes deployment may not work"
        fi
    fi
    
    # Check required directories exist
    if [[ ! -d "$PROJECT_ROOT/src" ]]; then
        log "ERROR" "Source directory not found: $PROJECT_ROOT/src"
        exit 1
    fi
    
    # Check script files exist
    if [[ "$PHASE_A_ONLY" == "true" || ("$PHASE_A_ONLY" != "true" && "$PHASE_B_ONLY" != "true") ]]; then
        if [[ ! -f "$SCRIPT_DIR/execute_phase_a.sh" ]]; then
            log "ERROR" "Phase A script not found: $SCRIPT_DIR/execute_phase_a.sh"
            exit 1
        fi
    fi
    
    if [[ "$PHASE_B_ONLY" == "true" || ("$PHASE_A_ONLY" != "true" && "$PHASE_B_ONLY" != "true") ]]; then
        if [[ ! -f "$SCRIPT_DIR/execute_phase_b.sh" ]]; then
            log "ERROR" "Phase B script not found: $SCRIPT_DIR/execute_phase_b.sh"
            exit 1
        fi
    fi
    
    log "INFO" "Prerequisites check completed"
}

# Execute Phase A
execute_phase_a() {
    log "INFO" "Starting Phase A: Real Data Integration"
    
    local phase_a_args="--batch"
    
    if [[ "$TRACK_PROGRESS" == "true" ]]; then
        phase_a_args="$phase_a_args --track-progress"
    fi
    
    if [[ "$EMAIL_NOTIFICATIONS" == "true" ]]; then
        phase_a_args="$phase_a_args --email '$NOTIFICATION_EMAIL'"
    fi
    
    if eval "bash '$SCRIPT_DIR/execute_phase_a.sh' $phase_a_args"; then
        log "INFO" "Phase A completed successfully"
        return 0
    else
        log "ERROR" "Phase A failed"
        return 1
    fi
}

# Execute Phase B
execute_phase_b() {
    log "INFO" "Starting Phase B: Production Deployment Scale-up"
    
    local phase_b_args="--batch"
    
    if [[ "$TRACK_PROGRESS" == "true" ]]; then
        phase_b_args="$phase_b_args --track-progress"
    fi
    
    if [[ "$EMAIL_NOTIFICATIONS" == "true" ]]; then
        phase_b_args="$phase_b_args --email '$NOTIFICATION_EMAIL'"
    fi
    
    if eval "bash '$SCRIPT_DIR/execute_phase_b.sh' $phase_b_args"; then
        log "INFO" "Phase B completed successfully"
        return 0
    else
        log "ERROR" "Phase B failed"
        return 1
    fi
}

# Monitor parallel execution
monitor_parallel_execution() {
    local phase_a_pid=$1
    local phase_b_pid=$2
    
    log "INFO" "Monitoring parallel execution - Phase A PID: $phase_a_pid, Phase B PID: $phase_b_pid"
    
    local phase_a_status=0
    local phase_b_status=0
    local phase_a_done=false
    local phase_b_done=false
    
    while [[ "$phase_a_done" == "false" || "$phase_b_done" == "false" ]]; do
        # Check Phase A
        if [[ "$phase_a_done" == "false" ]]; then
            if ! kill -0 $phase_a_pid 2>/dev/null; then
                wait $phase_a_pid
                phase_a_status=$?
                phase_a_done=true
                if [[ $phase_a_status -eq 0 ]]; then
                    log "INFO" "Phase A completed successfully"
                else
                    log "ERROR" "Phase A failed with exit code $phase_a_status"
                fi
            fi
        fi
        
        # Check Phase B
        if [[ "$phase_b_done" == "false" ]]; then
            if ! kill -0 $phase_b_pid 2>/dev/null; then
                wait $phase_b_pid
                phase_b_status=$?
                phase_b_done=true
                if [[ $phase_b_status -eq 0 ]]; then
                    log "INFO" "Phase B completed successfully"
                else
                    log "ERROR" "Phase B failed with exit code $phase_b_status"
                fi
            fi
        fi
        
        sleep 5
    done
    
    return $((phase_a_status + phase_b_status))
}

# Display execution summary
display_summary() {
    local start_time=$1
    local end_time=$2
    local phase_a_status=${3:-"not_run"}
    local phase_b_status=${4:-"not_run"}
    
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo ""
    echo -e "${CYAN}=============================================${NC}"
    echo -e "${CYAN}Cancer Genomics AI Demo - Execution Summary${NC}"
    echo -e "${CYAN}=============================================${NC}"
    echo ""
    
    echo -e "${BLUE}Execution Configuration:${NC}"
    echo "  Batch Mode: $BATCH_MODE"
    echo "  Progress Tracking: $TRACK_PROGRESS"
    echo "  Email Notifications: $EMAIL_NOTIFICATIONS"
    echo "  Parallel Execution: $PARALLEL_EXECUTION"
    echo ""
    
    echo -e "${BLUE}Phase Execution Status:${NC}"
    if [[ "$phase_a_status" == "success" ]]; then
        echo -e "  Phase A (Real Data Integration): ${GREEN}✓ SUCCESS${NC}"
    elif [[ "$phase_a_status" == "failed" ]]; then
        echo -e "  Phase A (Real Data Integration): ${RED}✗ FAILED${NC}"
    elif [[ "$phase_a_status" == "not_run" ]]; then
        echo -e "  Phase A (Real Data Integration): ${YELLOW}- NOT RUN${NC}"
    fi
    
    if [[ "$phase_b_status" == "success" ]]; then
        echo -e "  Phase B (Production Deployment): ${GREEN}✓ SUCCESS${NC}"
    elif [[ "$phase_b_status" == "failed" ]]; then
        echo -e "  Phase B (Production Deployment): ${RED}✗ FAILED${NC}"
    elif [[ "$phase_b_status" == "not_run" ]]; then
        echo -e "  Phase B (Production Deployment): ${YELLOW}- NOT RUN${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Execution Time:${NC}"
    printf "  Total Duration: %02d:%02d:%02d\n" $hours $minutes $seconds
    echo "  Start Time: $(date -r $start_time '+%Y-%m-%d %H:%M:%S')"
    echo "  End Time: $(date -r $end_time '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    echo -e "${BLUE}Log Files:${NC}"
    echo "  Master Log: $MASTER_LOG"
    if [[ -f "$LOG_DIR/phase_a/phase_a_execution_"*".log" ]]; then
        echo "  Phase A Log: $(ls -t $LOG_DIR/phase_a/phase_a_execution_*.log 2>/dev/null | head -1 || echo 'Not found')"
    fi
    if [[ -f "$LOG_DIR/phase_b/phase_b_execution_"*".log" ]]; then
        echo "  Phase B Log: $(ls -t $LOG_DIR/phase_b/phase_b_execution_*.log 2>/dev/null | head -1 || echo 'Not found')"
    fi
    
    if [[ "$TRACK_PROGRESS" == "true" ]]; then
        echo ""
        echo -e "${BLUE}Progress Files:${NC}"
        if [[ -f "$LOG_DIR/phase_a/phase_a_progress.json" ]]; then
            echo "  Phase A Progress: $LOG_DIR/phase_a/phase_a_progress.json"
        fi
        if [[ -f "$LOG_DIR/phase_b/phase_b_progress.json" ]]; then
            echo "  Phase B Progress: $LOG_DIR/phase_b/phase_b_progress.json"
        fi
    fi
    
    echo ""
    echo -e "${CYAN}=============================================${NC}"
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    echo -e "${MAGENTA}=============================================${NC}"
    echo -e "${MAGENTA}Cancer Genomics AI Demo - Master Execution${NC}"
    echo -e "${MAGENTA}=============================================${NC}"
    echo ""
    
    log "INFO" "Starting Cancer Genomics AI Demo execution"
    log "INFO" "Configuration: batch=$BATCH_MODE, progress=$TRACK_PROGRESS, email=$EMAIL_NOTIFICATIONS, parallel=$PARALLEL_EXECUTION"
    
    # Check prerequisites
    check_prerequisites
    
    # Send start notification
    send_master_notification "Cancer Alpha Demo: Execution Started" \
        "Cancer Genomics AI Demo execution has started.
        
Configuration:
- Batch Mode: $BATCH_MODE
- Progress Tracking: $TRACK_PROGRESS
- Parallel Execution: $PARALLEL_EXECUTION
- Phase A Only: $PHASE_A_ONLY
- Phase B Only: $PHASE_B_ONLY

Start Time: $(date)
Log File: $MASTER_LOG

The system will notify you upon completion or failure."
    
    local phase_a_status="not_run"
    local phase_b_status="not_run"
    local overall_success=true
    
    # Execute phases based on configuration
    if [[ "$PARALLEL_EXECUTION" == "true" ]]; then
        # Parallel execution
        log "INFO" "Starting parallel execution of Phase A and Phase B"
        
        execute_phase_a &
        local phase_a_pid=$!
        
        execute_phase_b &
        local phase_b_pid=$!
        
        if monitor_parallel_execution $phase_a_pid $phase_b_pid; then
            phase_a_status="success"
            phase_b_status="success"
            log "INFO" "Both phases completed successfully in parallel"
        else
            overall_success=false
            # Determine individual phase statuses
            wait $phase_a_pid
            if [[ $? -eq 0 ]]; then
                phase_a_status="success"
            else
                phase_a_status="failed"
            fi
            
            wait $phase_b_pid
            if [[ $? -eq 0 ]]; then
                phase_b_status="success"
            else
                phase_b_status="failed"
            fi
            
            log "ERROR" "One or both phases failed in parallel execution"
        fi
        
    else
        # Sequential execution
        if [[ "$PHASE_A_ONLY" == "true" ]]; then
            # Phase A only
            if execute_phase_a; then
                phase_a_status="success"
                log "INFO" "Phase A execution completed successfully"
            else
                phase_a_status="failed"
                overall_success=false
                log "ERROR" "Phase A execution failed"
            fi
            
        elif [[ "$PHASE_B_ONLY" == "true" ]]; then
            # Phase B only
            if execute_phase_b; then
                phase_b_status="success"
                log "INFO" "Phase B execution completed successfully"
            else
                phase_b_status="failed"
                overall_success=false
                log "ERROR" "Phase B execution failed"
            fi
            
        else
            # Both phases sequentially
            log "INFO" "Starting sequential execution: Phase A followed by Phase B"
            
            if execute_phase_a; then
                phase_a_status="success"
                log "INFO" "Phase A completed, starting Phase B"
                
                if execute_phase_b; then
                    phase_b_status="success"
                    log "INFO" "Both phases completed successfully"
                else
                    phase_b_status="failed"
                    overall_success=false
                    log "ERROR" "Phase B failed after Phase A success"
                fi
            else
                phase_a_status="failed"
                overall_success=false
                log "ERROR" "Phase A failed - skipping Phase B"
                
                if [[ "$BATCH_MODE" == "false" ]]; then
                    echo -e "${RED}Phase A failed. Continue with Phase B? (y/n)${NC}"
                    read -r response
                    if [[ "$response" == "y" || "$response" == "Y" ]]; then
                        if execute_phase_b; then
                            phase_b_status="success"
                            log "INFO" "Phase B completed successfully despite Phase A failure"
                        else
                            phase_b_status="failed"
                            log "ERROR" "Both phases failed"
                        fi
                    fi
                fi
            fi
        fi
    fi
    
    local end_time=$(date +%s)
    
    # Display summary
    display_summary $start_time $end_time $phase_a_status $phase_b_status
    
    # Send completion notification
    local completion_status="SUCCESS"
    if [[ "$overall_success" != "true" ]]; then
        completion_status="PARTIAL SUCCESS / FAILURE"
    fi
    
    send_master_notification "Cancer Alpha Demo: Execution $completion_status" \
        "Cancer Genomics AI Demo execution has completed.
        
Final Status: $completion_status
- Phase A (Real Data Integration): $phase_a_status
- Phase B (Production Deployment): $phase_b_status

Total Duration: $(printf '%02d:%02d:%02d' $((($((end_time - start_time)) / 3600))) $((($((end_time - start_time)) % 3600) / 60)) $((($((end_time - start_time)) % 60))))
End Time: $(date)

Log Files:
- Master Log: $MASTER_LOG
- Phase A Log: $(ls -t $LOG_DIR/phase_a/phase_a_execution_*.log 2>/dev/null | head -1 || echo 'Not found')
- Phase B Log: $(ls -t $LOG_DIR/phase_b/phase_b_execution_*.log 2>/dev/null | head -1 || echo 'Not found')

Please review the logs for detailed execution information."
    
    # Final status
    if [[ "$overall_success" == "true" ]]; then
        echo -e "${GREEN}🎉 Cancer Genomics AI Demo execution completed successfully!${NC}"
        log "INFO" "Master execution completed successfully"
        exit 0
    else
        echo -e "${RED}⚠️  Cancer Genomics AI Demo execution completed with errors${NC}"
        log "ERROR" "Master execution completed with errors"
        exit 1
    fi
}

# Handle script interruption
trap 'echo -e "\n${RED}Execution interrupted by user${NC}"; log "WARN" "Execution interrupted"; exit 130' INT TERM

# Run main function
main "$@"
