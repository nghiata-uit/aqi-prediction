#!/bin/bash

###############################################################################
# AQI Prediction System - Model Training Script
# 
# This script trains all models for all locations in the provided dataset.
# It supports multiple data files and provides options for customization.
#
# Usage:
#   ./train_all_locations.sh                    # Train with default settings
#   ./train_all_locations.sh path/to/data.csv   # Train with custom data file
#   ./train_all_locations.sh --help             # Show help message
###############################################################################

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATA_FILE="data/sample_data.csv"
OUTPUT_DIR="models/"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
PYTHON_CMD="python"

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
    DATA_FILE           Path to CSV file containing training data
                        Default: data/sample_data.csv

Options:
    -o, --output DIR    Output directory for trained models
                        Default: models/
    -h, --help          Show this help message
    --log FILE          Log file name (default: training_YYYYMMDD_HHMMSS.log)
    --python CMD        Python command to use (default: python)

Examples:
    # Train with default settings
    ./train_all_locations.sh

    # Train with custom data file
    ./train_all_locations.sh data/my_data.csv

    # Specify output directory
    ./train_all_locations.sh -o my_models/ data/my_data.csv

    # Use python3 explicitly
    ./train_all_locations.sh --python python3

Data File Format:
    The CSV file should contain the following columns:
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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
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

# Main training function
main() {
    print_header "🚀 AQI PREDICTION SYSTEM - MODEL TRAINING"
    
    echo ""
    print_info "Configuration:"
    echo "  Data file:      $DATA_FILE"
    echo "  Output dir:     $OUTPUT_DIR"
    echo "  Log file:       $LOG_FILE"
    echo "  Python cmd:     $PYTHON_CMD"
    echo ""
    
    # Check if Python is available
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python command '$PYTHON_CMD' not found"
        print_info "Install Python or specify correct command with --python option"
        exit 1
    fi
    
    print_success "Python found: $($PYTHON_CMD --version)"
    
    # Check if data file exists
    if [ ! -f "$DATA_FILE" ]; then
        print_error "Data file not found: $DATA_FILE"
        print_info "Please provide a valid CSV file path"
        exit 1
    fi
    
    print_success "Data file found: $DATA_FILE"
    
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
    print('✅ All required packages are installed')
except ImportError as e:
    print(f'❌ Missing package: {e.name}')
    print('Please install required packages: pip install -r requirements.txt')
    sys.exit(1)
" || exit 1
    
    # Create output directory if it doesn't exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        print_info "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
    
    print_success "Output directory ready: $OUTPUT_DIR"
    
    # Start training
    echo ""
    print_header "📊 STARTING MODEL TRAINING"
    echo ""
    
    print_info "Training will process all unique locations in the dataset..."
    print_info "This may take several minutes depending on data size and number of locations"
    print_info "Log file: $LOG_FILE"
    echo ""
    
    # Run the training script with both console output and logging
    $PYTHON_CMD train_models.py \
        --data "$DATA_FILE" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        print_header "✅ TRAINING COMPLETED SUCCESSFULLY"
        echo ""
        print_success "Models saved to: $OUTPUT_DIR"
        print_success "Log saved to: $LOG_FILE"
        echo ""
        
        # Show summary of trained models
        print_info "Trained models:"
        if ls "$OUTPUT_DIR"/*_best.pkl 1> /dev/null 2>&1; then
            for model_file in "$OUTPUT_DIR"/*_best.pkl; do
                model_name=$(basename "$model_file")
                model_size=$(ls -lh "$model_file" | awk '{print $5}')
                echo "  - $model_name ($model_size)"
            done
        else
            print_warning "No model files found in $OUTPUT_DIR"
        fi
        
        echo ""
        print_info "Next steps:"
        echo "  1. Start the API: uvicorn api.main:app --reload"
        echo "  2. Test predictions: python test_system.py"
        echo "  3. View API docs: http://localhost:8000/docs"
        echo ""
        
        exit 0
    else
        print_error "Training failed with exit code: $TRAIN_EXIT_CODE"
        print_info "Check log file for details: $LOG_FILE"
        exit $TRAIN_EXIT_CODE
    fi
}

# Run main function
main
