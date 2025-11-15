#!/bin/bash

###############################################################################
# AQI Prediction System - Model Training Script
# 
# This script trains all models for all locations in all CSV files in data directory.
# It supports multiple data files and provides options for customization.
#
# Usage:
#   ./train_all_locations.sh                    # Train all CSV files in data/
#   ./train_all_locations.sh path/to/data.csv   # Train specific CSV file
#   ./train_all_locations.sh --all              # Train all CSV files in data/
#   ./train_all_locations.sh --help             # Show help message
###############################################################################

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="data/"
DATA_FILE=""
OUTPUT_DIR="models/"
LOG_DIR="logs/"
PYTHON_CMD="python"
TRAIN_ALL=false

# Function to print colored messages
print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to show help
show_help() {
    cat << EOF
AQI Prediction System - Model Training Script

Usage:
    ./train_all_locations.sh [OPTIONS] [DATA_FILE]

Arguments:
    DATA_FILE           Path to specific CSV file to train
                        If not provided, trains all CSV files in data/ directory

Options:
    --all               Train all CSV files in data/ directory (default behavior)
    -d, --data-dir DIR  Data directory containing CSV files
                        Default: data/
    -o, --output DIR    Output directory for trained models
                        Default: models/
    -h, --help          Show this help message
    --log-dir DIR       Directory for log files (default: logs/)
    --python CMD        Python command to use (default: python)

Examples:
    # Train all CSV files in data/ directory
    ./train_all_locations.sh
    ./train_all_locations.sh --all

    # Train specific CSV file
    ./train_all_locations.sh data/my_data.csv

    # Specify custom data directory
    ./train_all_locations.sh -d my_data_folder/

    # Specify output directory
    ./train_all_locations.sh -o my_models/

    # Use python3 explicitly
    ./train_all_locations.sh --python python3

Data File Format:
    Each CSV file should contain the following columns:
    - datetime: Timestamp (e.g., 2022-10-12 20:00:00)
    - lon: Longitude
    - lat: Latitude
    - co, no, no2, o3, so2, pm2_5, pm10, nh3: Pollutant levels
    - aqi: Air Quality Index (target variable)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --all)
            TRAIN_ALL=true
            shift
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -*)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            DATA_FILE="$1"
            shift
            ;;
    esac
done

# Function to train a single CSV file
train_single_file() {
    local csv_file="$1"
    local file_name=$(basename "$csv_file" .csv)
    local log_file="${LOG_DIR}training_${file_name}_$(date +%Y%m%d_%H%M%S).log"
    
    print_header "📊 TRAINING: $csv_file"
    echo ""
    
    print_info "Training will process all unique locations in the dataset..."
    print_info "This may take several minutes depending on data size and number of locations"
    print_info "Log file: $log_file"
    echo ""
    
    # Run the training script with both console output and logging
    $PYTHON_CMD train_models.py \
        --data "$csv_file" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "$log_file"
    
    # Check exit status
    local train_exit_code=${PIPESTATUS[0]}
    
    echo ""
    
    if [ $train_exit_code -eq 0 ]; then
        print_success "Training completed for: $csv_file"
        print_success "Log saved to: $log_file"
        return 0
    else
        print_error "Training failed for: $csv_file (exit code: $train_exit_code)"
        print_info "Check log file for details: $log_file"
        return 1
    fi
}

# Main training function
main() {
    print_header "🚀 AQI PREDICTION SYSTEM - MODEL TRAINING"
    
    echo ""
    print_info "Configuration:"
    echo "  Data directory: $DATA_DIR"
    echo "  Output dir:     $OUTPUT_DIR"
    echo "  Log dir:        $LOG_DIR"
    echo "  Python cmd:     $PYTHON_CMD"
    echo ""
    
    # Check if Python is available
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python command '$PYTHON_CMD' not found"
        print_info "Install Python or specify correct command with --python option"
        exit 1
    fi
    
    print_success "Python found: $($PYTHON_CMD --version)"
    
    # Check if required Python packages are installed
    print_info "Checking Python dependencies..."
    
    $PYTHON_CMD -c "
import sys
try:
    import pandas
    import numpy
    import sklearn
    import xgboost
    import tensorflow
    import prophet
    
    # Check NumPy version
    numpy_version = tuple(map(int, numpy.__version__.split('.')[:2]))
    if numpy_version >= (2, 0):
        print('❌ NumPy 2.0+ detected, which is not compatible with Prophet')
        print('Please downgrade NumPy: pip uninstall numpy && pip install \"numpy<2.0\"')
        print('Or reinstall all dependencies: pip install -r requirements.txt')
        sys.exit(1)
    
    print('✅ All required packages are installed')
    print(f'   NumPy version: {numpy.__version__} (compatible)')
except ImportError as e:
    print(f'❌ Missing package: {e.name}')
    print('Please install required packages: pip install -r requirements.txt')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error checking dependencies: {str(e)}')
    sys.exit(1)
" || exit 1
    
    # Create output directory if it doesn't exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        print_info "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
    
    # Create log directory if it doesn't exist
    if [ ! -d "$LOG_DIR" ]; then
        print_info "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
    
    print_success "Output directory ready: $OUTPUT_DIR"
    print_success "Log directory ready: $LOG_DIR"
    
    echo ""
    
    # Determine which files to train
    local csv_files=()
    
    if [ -n "$DATA_FILE" ]; then
        # Single file specified
        if [ ! -f "$DATA_FILE" ]; then
            print_error "Data file not found: $DATA_FILE"
            print_info "Please provide a valid CSV file path"
            exit 1
        fi
        csv_files=("$DATA_FILE")
        print_info "Training single file: $DATA_FILE"
    else
        # Train all CSV files in data directory
        if [ ! -d "$DATA_DIR" ]; then
            print_error "Data directory not found: $DATA_DIR"
            exit 1
        fi
        
        # Find all CSV files in data directory
        while IFS= read -r -d '' file; do
            csv_files+=("$file")
        done < <(find "$DATA_DIR" -maxdepth 1 -name "*.csv" -type f -print0)
        
        if [ ${#csv_files[@]} -eq 0 ]; then
            print_error "No CSV files found in: $DATA_DIR"
            print_info "Please add CSV files to the data directory"
            exit 1
        fi
        
        print_info "Found ${#csv_files[@]} CSV file(s) to process:"
        for csv_file in "${csv_files[@]}"; do
            echo "  - $(basename "$csv_file")"
        done
    fi
    
    echo ""
    
    # Train models for each CSV file
    local total_files=${#csv_files[@]}
    local successful=0
    local failed=0
    local current=0
    
    for csv_file in "${csv_files[@]}"; do
        current=$((current + 1))
        echo ""
        print_header "[$current/$total_files] Processing: $(basename "$csv_file")"
        echo ""
        
        if train_single_file "$csv_file"; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
        
        echo ""
    done
    
    # Final summary
    print_header "✅ TRAINING PIPELINE COMPLETED"
    echo ""
    print_info "Summary:"
    echo "  Total files:      $total_files"
    echo "  Successful:       $successful"
    echo "  Failed:           $failed"
    echo ""
    
    # Show summary of trained models
    print_info "Trained models in $OUTPUT_DIR:"
    if ls "$OUTPUT_DIR"/*_best.pkl 1> /dev/null 2>&1; then
        local model_count=0
        for model_file in "$OUTPUT_DIR"/*_best.pkl; do
            model_name=$(basename "$model_file")
            model_size=$(ls -lh "$model_file" | awk '{print $5}')
            echo "  - $model_name ($model_size)"
            model_count=$((model_count + 1))
        done
        echo ""
        print_success "Total models trained: $model_count"
    else
        print_warning "No model files found in $OUTPUT_DIR"
    fi
    
    echo ""
    print_info "Next steps:"
    echo "  1. Start the API: uvicorn api.main:app --reload"
    echo "  2. Test predictions: python test_system.py"
    echo "  3. View API docs: http://localhost:8000/docs"
    echo ""
    
    if [ $failed -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main
