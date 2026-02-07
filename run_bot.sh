#!/bin/bash

# AI Trading Bot Runner Script
# Version: 1.0.0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BOT_SCRIPT="ai_trading_bot.py"
DASHBOARD_SCRIPT="dashboard.py"
CONFIG_FILE="config.json"
LOG_DIR="logs"
BACKUP_DIR="backups"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════╗"
    echo "║           AI TRADING BOT CONTROL PANEL             ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[i] $1${NC}"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | awk '{print $2}')
    print_success "Python $python_version detected"
    
    # Check virtual environment
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found"
        read -p "Create virtual environment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 -m venv venv
            print_success "Virtual environment created"
        fi
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check requirements
    if [ -f "requirements.txt" ]; then
        print_info "Checking Python packages..."
        pip install -r requirements.txt --quiet
        print_success "Dependencies checked"
    fi
}

check_config() {
    print_info "Checking configuration..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        echo "Please create config.json first"
        exit 1
    fi
    
    # Validate JSON
    if ! python3 -m json.tool "$CONFIG_FILE" > /dev/null 2>&1; then
        print_error "Invalid JSON in config file"
        exit 1
    fi
    
    print_success "Configuration validated"
}

backup_data() {
    print_info "Creating backup..."
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$BACKUP_DIR/backup_$timestamp.tar.gz"
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup important files
    tar -czf "$backup_file" \
        config.json \
        ml_models/ \
        logs/ \
        data/ \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Backup created: $backup_file"
    else
        print_warning "Backup creation failed"
    fi
}

start_bot() {
    print_info "Starting AI Trading Bot..."
    
    # Check if bot is already running
    if pgrep -f "$BOT_SCRIPT" > /dev/null; then
        print_warning "Bot is already running"
        return 1
    fi
    
    # Start bot in background
    nohup python3 "$BOT_SCRIPT" > "$LOG_DIR/bot_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    # Wait a bit and check if it's running
    sleep 3
    if pgrep -f "$BOT_SCRIPT" > /dev/null; then
        print_success "Bot started successfully (PID: $(pgrep -f "$BOT_SCRIPT"))"
        return 0
    else
        print_error "Failed to start bot"
        return 1
    fi
}

stop_bot() {
    print_info "Stopping AI Trading Bot..."
    
    # Find and kill bot process
    pkill -f "$BOT_SCRIPT"
    
    # Wait for process to stop
    sleep 2
    
    if pgrep -f "$BOT_SCRIPT" > /dev/null; then
        print_warning "Bot still running, forcing stop..."
        pkill -9 -f "$BOT_SCRIPT"
        sleep 1
    fi
    
    if ! pgrep -f "$BOT_SCRIPT" > /dev/null; then
        print_success "Bot stopped successfully"
        return 0
    else
        print_error "Failed to stop bot"
        return 1
    fi
}

start_dashboard() {
    print_info "Starting Dashboard..."
    
    # Check if dashboard is already running
    if pgrep -f "$DASHBOARD_SCRIPT" > /dev/null; then
        print_warning "Dashboard is already running"
        return 1
    fi
    
    # Get dashboard port from config
    port=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('monitoring', {}).get('dashboard_port', 8050))
")
    
    # Start dashboard in background
    nohup python3 "$DASHBOARD_SCRIPT" > "$LOG_DIR/dashboard_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    # Wait for dashboard to start
    sleep 5
    
    if pgrep -f "$DASHBOARD_SCRIPT" > /dev/null; then
        print_success "Dashboard started successfully"
        print_success "Dashboard URL: http://localhost:$port"
        return 0
    else
        print_error "Failed to start dashboard"
        return 1
    fi
}

stop_dashboard() {
    print_info "Stopping Dashboard..."
    
    # Find and kill dashboard process
    pkill -f "$DASHBOARD_SCRIPT"
    
    sleep 2
    
    if ! pgrep -f "$DASHBOARD_SCRIPT" > /dev/null; then
        print_success "Dashboard stopped successfully"
        return 0
    else
        print_error "Failed to stop dashboard"
        return 1
    fi
}

show_status() {
    print_info "System Status:"
    echo "----------------"
    
    # Bot status
    if pgrep -f "$BOT_SCRIPT" > /dev/null; then
        echo -e "Bot: ${GREEN}RUNNING${NC} (PID: $(pgrep -f "$BOT_SCRIPT"))"
    else
        echo -e "Bot: ${RED}STOPPED${NC}"
    fi
    
    # Dashboard status
    if pgrep -f "$DASHBOARD_SCRIPT" > /dev/null; then
        echo -e "Dashboard: ${GREEN}RUNNING${NC} (PID: $(pgrep -f "$DASHBOARD_SCRIPT"))"
    else
        echo -e "Dashboard: ${RED}STOPPED${NC}"
    fi
    
    # Disk usage
    echo -e "Disk Usage: $(df -h . | awk 'NR==2 {print $5}')"
    
    # Memory usage
    echo -e "Memory Usage: $(free -m | awk 'NR==2 {printf "%.1f%%", $3*100/$2}')"
    
    # Log files
    log_count=$(find "$LOG_DIR" -name "*.log" | wc -l)
    echo -e "Log Files: $log_count"
    
    # Last backup
    last_backup=$(ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | head -1)
    if [ -n "$last_backup" ]; then
        echo -e "Last Backup: $(basename "$last_backup")"
    else
        echo -e "Last Backup: ${YELLOW}None${NC}"
    fi
}

view_logs() {
    print_info "Recent Logs:"
    echo "------------"
    
    # Show last 5 log files
    recent_logs=$(find "$LOG_DIR" -name "*.log" -type f -exec ls -lt {} + | head -5)
    
    if [ -n "$recent_logs" ]; then
        echo "$recent_logs" | awk '{print $6" "$7" "$8" - "$9}'
        
        read -p "View latest log? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            latest_log=$(find "$LOG_DIR" -name "*.log" -type f -exec ls -t {} + | head -1)
            if [ -f "$latest_log" ]; then
                echo
                echo "=== Last 20 lines of $latest_log ==="
                tail -20 "$latest_log"
            fi
        fi
    else
        print_warning "No log files found"
    fi
}

run_backtest() {
    print_info "Starting Backtest..."
    
    read -p "Enter symbol (e.g., BTC/IDR): " symbol
    read -p "Enter strategy (ml/momentum/mean_reversion/breakout): " strategy
    read -p "Initial capital (IDR): " capital
    
    if [ -z "$capital" ]; then
        capital=1000000
    fi
    
    echo
    print_info "Running backtest for $symbol with $strategy strategy..."
    
    python3 -c "
from backtester import Backtester
backtester = Backtester()
result = backtester.run_backtest('$symbol', strategy='$strategy', initial_capital=$capital)
backtester.generate_report('$symbol')
"
}

emergency_stop() {
    print_warning "EMERGENCY STOP PROCEDURE"
    echo "This will immediately stop all trading activities!"
    
    read -p "Are you sure? (type 'YES' to confirm): " confirmation
    if [ "$confirmation" != "YES" ]; then
        print_info "Emergency stop cancelled"
        return
    fi
    
    # Stop everything
    stop_bot
    stop_dashboard
    
    # Send emergency notification
    if [ -f "notifications.py" ]; then
        python3 -c "
from notifications import NotificationSystem
notifier = NotificationSystem()
notifier.send_emergency_stop('Manual emergency stop initiated', 'All trading activities have been stopped.')
"
    fi
    
    print_success "Emergency stop completed"
}

show_menu() {
    while true; do
        echo
        echo -e "${BLUE}MAIN MENU${NC}"
        echo "1. Start Trading Bot"
        echo "2. Stop Trading Bot"
        echo "3. Start Dashboard"
        echo "4. Stop Dashboard"
        echo "5. Show Status"
        echo "6. View Logs"
        echo "7. Run Backtest"
        echo "8. Backup Data"
        echo "9. Emergency Stop"
        echo "0. Exit"
        echo
        
        read -p "Select option (0-9): " choice
        
        case $choice in
            1) start_bot ;;
            2) stop_bot ;;
            3) start_dashboard ;;
            4) stop_dashboard ;;
            5) show_status ;;
            6) view_logs ;;
            7) run_backtest ;;
            8) backup_data ;;
            9) emergency_stop ;;
            0) 
                print_info "Exiting..."
                exit 0
                ;;
            *) print_error "Invalid option" ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Main execution
main() {
    print_header
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then 
        print_warning "Warning: Running as root is not recommended"
    fi
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    
    # Check dependencies
    check_dependencies
    
    # Check configuration
    check_config
    
    # Show menu
    show_menu
}

# Handle script arguments
case "$1" in
    "start")
        check_dependencies
        check_config
        start_bot
        ;;
    "stop")
        stop_bot
        ;;
    "dashboard")
        check_dependencies
        start_dashboard
        ;;
    "status")
        show_status
        ;;
    "backtest")
        check_dependencies
        run_backtest
        ;;
    "backup")
        backup_data
        ;;
    *)
        main
        ;;
esac