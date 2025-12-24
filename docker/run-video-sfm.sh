#!/bin/bash
#
# Run COLMAP Video-to-SFM Pipeline in Docker
#
# This script launches a Docker container with GPU support and runs the
# video_to_sfm.py pipeline script.
#
# Usage:
#   ./run-video-sfm.sh <input_video> <output_dir> [options...]
#
# Examples:
#   # Basic usage - sparse reconstruction at 3 FPS
#   ./run-video-sfm.sh /path/to/video.mov /path/to/output
#
#   # High quality with dense reconstruction
#   ./run-video-sfm.sh /path/to/video.mov /path/to/output --dense --quality high
#
#   # Quick test with limited frames
#   ./run-video-sfm.sh /path/to/video.mov /path/to/output --max-frames 30
#
# All options after input/output are passed directly to video_to_sfm.py.
# Run with --help for all available options.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <input_video> <output_dir> [pipeline_options...]"
    echo ""
    echo "Arguments:"
    echo "  input_video    Path to input video file (MOV, MP4, etc.)"
    echo "  output_dir     Directory for output reconstruction files"
    echo ""
    echo "Pipeline options (passed to video_to_sfm.py):"
    echo "  --fps N              Frames per second to extract (default: 3.0)"
    echo "  --max-image-size N   Max image dimension in pixels (default: 1920)"
    echo "  --max-frames N       Max frames to extract, 0=unlimited (default: 0)"
    echo "  --max-num-features N SIFT features per image (default: 8192)"
    echo "  --sequential-overlap N  Consecutive frames to match (default: 10)"
    echo "  --loop-detection     Enable loop closure detection"
    echo "  --dense              Enable dense reconstruction"
    echo "  --quality PRESET     low/medium/high/extreme (default: high)"
    echo "  --no-gpu             Disable GPU acceleration"
    echo ""
    echo "Examples:"
    echo "  $0 video.mov output/"
    echo "  $0 video.mov output/ --dense --quality high"
    echo "  $0 video.mov output/ --max-frames 30 --fps 6"
}

if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
shift 2
EXTRA_ARGS="$@"

# Validate input video
if [ ! -f "$INPUT_VIDEO" ]; then
    echo -e "${RED}ERROR: Input video not found: $INPUT_VIDEO${NC}"
    exit 1
fi

# Get absolute paths
INPUT_VIDEO=$(realpath "$INPUT_VIDEO")
INPUT_DIR=$(dirname "$INPUT_VIDEO")
INPUT_FILENAME=$(basename "$INPUT_VIDEO")

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}COLMAP Video-to-SFM Pipeline${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Input video: $INPUT_VIDEO"
echo "Output dir:  $OUTPUT_DIR"
echo "Extra args:  $EXTRA_ARGS"
echo ""

# Check if local colmap:latest image exists, otherwise use official image
if docker image inspect colmap:latest >/dev/null 2>&1; then
    echo -e "${GREEN}Using local COLMAP Docker image...${NC}"
    COLMAP_IMAGE="colmap:latest"
else
    echo -e "${YELLOW}Local COLMAP image not found, pulling official image...${NC}"
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

# Build Docker arguments
DOCKER_ARGS=(
    --rm
    -v "${INPUT_DIR}:/input:ro"
    -v "${OUTPUT_DIR}:/output"
    -w /output
    --user "$(id -u):$(id -g)"
)

# GPU Detection
echo "Testing for GPU acceleration..."
if sudo docker run --rm --gpus all "${COLMAP_IMAGE}" nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}✅ GPU detected. Using --gpus all.${NC}"
    DOCKER_ARGS+=( --gpus all )
else
    echo -e "${YELLOW}⚠️  GPU not detected. Using CPU mode (slower).${NC}"
fi

# Get the directory where this script is located (for mounting video_to_sfm.py)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Mount the Python script
DOCKER_ARGS+=( -v "${SCRIPT_DIR}/video_to_sfm.py:/video_to_sfm.py:ro" )

# Run the pipeline
echo ""
echo -e "${GREEN}Starting pipeline...${NC}"
echo ""

docker run "${DOCKER_ARGS[@]}" "${COLMAP_IMAGE}" \
    python3 /video_to_sfm.py \
    --input "/input/${INPUT_FILENAME}" \
    --output /output \
    $EXTRA_ARGS

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Pipeline complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Output files are in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
