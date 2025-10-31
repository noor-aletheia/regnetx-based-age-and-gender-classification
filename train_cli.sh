#!/bin/bash

# CLI wrapper for MobileNet training with Docker
# Usage: ./train_cli.sh [options]

set -e

# Default values
CONFIG_FILE="config.yaml"
DOCKER_COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="regnetx-training"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "RegNetX Age & Gender Classification Training CLI"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Model Selection:"
    echo "  --model MODEL               Train specific model(s) (space-separated)"
    echo "  --list-models              List all available models"
    echo ""
    echo "Training Parameters:"
    echo "  --epochs N                 Total number of epochs"
    echo "  --batch-size N             Batch size"
    echo "  --lr FLOAT                 Learning rate for Phase 1"
    echo "  --lr2 FLOAT                Learning rate for Phase 2"
    echo "  --input-size N             Input image size"
    echo ""
    echo "Augmentation:"
    echo "  --no-augmentation          Disable data augmentation"
    echo "  --augmentation-strength STR Set augmentation strength (light|medium|strong)"
    echo ""
    echo "Hardware:"
    echo "  --cpu                      Force CPU training"
    echo "  --mixed-precision          Enable mixed precision training"
    echo "  --no-mixed-precision       Disable mixed precision training"
    echo ""
    echo "Output:"
    echo "  --output-dir DIR           Output directory"
    echo "  --experiment-name NAME     Experiment name"
    echo ""
    echo "Other:"
    echo "  --dry-run                  Show configuration without training"
    echo "  --verbose                  Enable verbose logging"
    echo "  --rebuild                  Rebuild Docker image before training"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Train all models"
    echo "  $0 --model regnetx_008                      # Train specific model"
    echo "  $0 --model regnetx_006 regnetx_016           # Train multiple models"
    echo "  $0 --epochs 30 --batch-size 32              # Custom parameters"
    echo "  $0 --no-augmentation --experiment-name test # No augmentation with experiment name"
    echo "  $0 --list-models                            # List available models"
    echo "  $0 --dry-run --verbose                      # Check configuration"
}

# Check if Docker and docker-compose are available
check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're using docker-compose or docker compose
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        DOCKER_COMPOSE_CMD="docker compose"
    fi
}

# Check if required files exist
check_files() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file '$CONFIG_FILE' not found"
        exit 1
    fi
    
    if [[ ! -f "$DOCKER_COMPOSE_FILE" ]]; then
        print_error "Docker Compose file '$DOCKER_COMPOSE_FILE' not found"
        exit 1
    fi
    
    if [[ ! -d "data" ]]; then
        print_warning "Data directory 'data' not found"
        print_info "Please ensure your dataset is in the 'data' directory"
    fi
}

# Parse command line arguments
TRAIN_ARGS=()
REBUILD_IMAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --rebuild)
            REBUILD_IMAGE=true
            shift
            ;;
        --cpu)
            TRAIN_ARGS+=("--device" "cpu")
            shift
            ;;
        --list-models)
            TRAIN_ARGS+=("--list-models")
            shift
            ;;
        --model|--epochs|--batch-size|--lr|--lr2|--input-size|--output-dir|--experiment-name|--augmentation-strength)
            TRAIN_ARGS+=("$1" "$2")
            shift 2
            ;;
        --no-augmentation|--mixed-precision|--no-mixed-precision|--dry-run|--verbose)
            TRAIN_ARGS+=("$1")
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "RegNetX Training CLI"
    print_info "======================"
    
    # Check requirements
    check_requirements
    check_files
    
    # Rebuild image if requested
    if [[ "$REBUILD_IMAGE" == "true" ]]; then
        print_info "Rebuilding Docker image..."
        $DOCKER_COMPOSE_CMD build --no-cache
        print_success "Docker image rebuilt"
    fi
    
    # Check if we just want to list models
    if [[ " ${TRAIN_ARGS[@]} " =~ " --list-models " ]]; then
        print_info "Listing available models..."
        $DOCKER_COMPOSE_CMD run --rm $SERVICE_NAME python train_cli.py --list-models
        exit 0
    fi
    
    # Check if this is a dry run
    if [[ " ${TRAIN_ARGS[@]} " =~ " --dry-run " ]]; then
        print_info "Performing dry run..."
        $DOCKER_COMPOSE_CMD run --rm $SERVICE_NAME python train_cli.py "${TRAIN_ARGS[@]}"
        exit 0
    fi
    
    # Start training
    print_info "Starting training with arguments: ${TRAIN_ARGS[*]}"
    
    # Create outputs directory if it doesn't exist
    mkdir -p outputs/{models,logs}
    
    # Run training
    if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
        # No arguments, run default training
        print_info "Running default training (all models)..."
        $DOCKER_COMPOSE_CMD run --rm $SERVICE_NAME python train_cli.py
    else
        # Run with specified arguments
        $DOCKER_COMPOSE_CMD run --rm $SERVICE_NAME python train_cli.py "${TRAIN_ARGS[@]}"
    fi
    
    # Check exit code
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "Training completed successfully!"
        print_info "Results are available in the 'outputs' directory"
    else
        print_error "Training failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
}

# Run main function
main "$@"