#!/usr/bin/env python3
"""
Video to Structure-from-Motion (SfM) Pipeline using COLMAP

This script orchestrates the conversion of a video file to a 3D reconstruction
using COLMAP. It handles frame extraction, feature extraction, matching, and
both sparse and dense reconstruction.

PROCESSING TIME ESTIMATES (4K @ 6 FPS, High Quality):
┌──────────────┬────────┬──────────────┬─────────────┬───────────┐
│ Video Length │ Frames │ Sparse (GPU) │ Dense (GPU) │ Total     │
├──────────────┼────────┼──────────────┼─────────────┼───────────┤
│ 1 second     │ 6      │ ~1-2 min     │ ~5-10 min   │ ~10-15min │
│ 30 seconds   │ 180    │ ~15-30 min   │ ~2-4 hours  │ ~3-5 hrs  │
│ 60 seconds   │ 360    │ ~45-90 min   │ ~5-10 hours │ ~6-12 hrs │
└──────────────┴────────┴──────────────┴─────────────┴───────────┘

PARAMETERS:
  --fps: Frame extraction rate. Higher = more frames = better quality but slower.
         Recommended: 2-3 fps for walking speed, 5-10 fps for fast motion.
  
  --max-image-size: Maximum dimension before downscaling. Reduces memory and time.
         4K (3840) -> 1080p (1920) gives ~4x speedup with minor quality loss.
  
  --max-num-features: SIFT features per image. More = better matching but slower.
         Range: 2048 (fast) to 8192 (high quality).
  
  --sequential-overlap: Number of consecutive frames to match. Higher = better
         for fast motion but slower. Default 10 is good for video.
  
  --loop-detection: Enable loop closure detection for videos that revisit areas.
         Adds processing time but improves reconstruction of looped paths.
  
  --dense: Enable dense reconstruction after sparse. Significantly slower but
         produces detailed point clouds suitable for meshing.
  
  --max-frames: Limit number of frames for testing. 0 = use all frames.

Author: Generated for COLMAP video-to-SFM pipeline
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StageTimer:
    """Track timing for pipeline stages with ETA estimation."""
    name: str
    start_time: float = 0
    end_time: float = 0
    total_items: int = 0
    processed_items: int = 0
    
    def start(self, total_items: int = 0):
        self.start_time = time.time()
        self.total_items = total_items
        self.processed_items = 0
        print(f"\n{'='*60}")
        print(f"▶ STARTING: {self.name}")
        if total_items > 0:
            print(f"  Items to process: {total_items}")
        print(f"{'='*60}")
    
    def update(self, processed: int):
        self.processed_items = processed
        if self.total_items > 0 and processed > 0:
            elapsed = time.time() - self.start_time
            rate = processed / elapsed
            remaining = (self.total_items - processed) / rate if rate > 0 else 0
            print(f"  Progress: {processed}/{self.total_items} "
                  f"({100*processed/self.total_items:.1f}%) "
                  f"| Elapsed: {format_time(elapsed)} "
                  f"| ETA: {format_time(remaining)}")
    
    def finish(self):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"✓ COMPLETED: {self.name} in {format_time(elapsed)}")
        return elapsed


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_command(cmd: list, desc: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with optional description."""
    if desc:
        print(f"  → {desc}")
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 3.0,
    max_image_size: int = 1920,
    max_frames: int = 0
) -> int:
    """
    Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        max_image_size: Maximum dimension (width or height) for output frames
        max_frames: Maximum number of frames to extract (0 = unlimited)
    
    Returns:
        Number of frames extracted
    """
    timer = StageTimer("Frame Extraction (ffmpeg)")
    
    # Get video info first
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Could not probe video: {result.stderr}")
        sys.exit(1)
    
    video_info = json.loads(result.stdout)
    duration = float(video_info["format"]["duration"])
    expected_frames = int(duration * fps)
    if max_frames > 0:
        expected_frames = min(expected_frames, max_frames)
    
    timer.start(expected_frames)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    vf_filters = [f"fps={fps}", f"scale='min({max_image_size},iw)':'-1'"]
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", ",".join(vf_filters),
        "-q:v", "2",  # High quality JPEG
    ]
    
    if max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    
    cmd.append(str(output_dir / "frame_%05d.jpg"))
    
    run_command(cmd, f"Extracting at {fps} fps, max size {max_image_size}px")
    
    # Count extracted frames
    num_frames = len(list(output_dir.glob("frame_*.jpg")))
    timer.finish()
    
    print(f"  Extracted {num_frames} frames to {output_dir}")
    return num_frames


def run_sparse_reconstruction(
    image_dir: Path,
    output_dir: Path,
    database_path: Path,
    max_image_size: int = 1920,
    max_num_features: int = 8192,
    sequential_overlap: int = 10,
    loop_detection: bool = False,
    use_gpu: bool = True
) -> bool:
    """
    Run COLMAP sparse reconstruction pipeline.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output files
        database_path: Path to COLMAP database file
        max_image_size: Maximum image dimension for feature extraction
        max_num_features: Maximum SIFT features per image
        sequential_overlap: Number of consecutive images to match
        loop_detection: Enable loop closure detection
        use_gpu: Use GPU acceleration if available
    
    Returns:
        True if reconstruction succeeded
    """
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = len(list(image_dir.glob("*.jpg")))
    
    # Stage 1: Feature Extraction
    timer = StageTimer("Feature Extraction (SIFT)")
    timer.start(num_images)
    
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",  # All frames from same camera
        "--SiftExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_num_features),
    ]
    if use_gpu:
        cmd.extend(["--FeatureExtraction.use_gpu", "1"])
    
    run_command(cmd, f"Extracting up to {max_num_features} SIFT features per image")
    timer.finish()
    
    # Stage 2: Sequential Matching
    timer = StageTimer("Sequential Matching")
    timer.start(num_images)
    
    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", str(sequential_overlap),
        "--SequentialMatching.quadratic_overlap", "1",
    ]
    if loop_detection:
        cmd.extend([
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.loop_detection_period", "10",
            "--SequentialMatching.loop_detection_num_images", "50",
        ])
    if use_gpu:
        cmd.extend(["--FeatureMatching.use_gpu", "1"])
    
    run_command(cmd, f"Matching with overlap={sequential_overlap}, loop_detection={loop_detection}")
    timer.finish()
    
    # Stage 3: Incremental Mapping (Sparse Reconstruction)
    timer = StageTimer("Sparse Reconstruction (Mapper)")
    timer.start(num_images)
    
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ]
    
    run_command(cmd, "Running incremental Structure-from-Motion")
    timer.finish()
    
    # Check if reconstruction succeeded
    model_dirs = list(sparse_dir.glob("*"))
    if not model_dirs:
        print("ERROR: No reconstruction models produced")
        return False
    
    print(f"  Created {len(model_dirs)} reconstruction model(s)")
    
    # Export PLY point cloud from best model (model 0)
    best_model = sparse_dir / "0"
    if best_model.exists():
        ply_path = output_dir / "sparse_pointcloud.ply"
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(best_model),
            "--output_path", str(ply_path),
            "--output_type", "PLY"
        ]
        run_command(cmd, f"Exporting point cloud to {ply_path}")
        print(f"  ✓ Sparse point cloud: {ply_path}")
    
    return True


def run_dense_reconstruction(
    image_dir: Path,
    output_dir: Path,
    sparse_model: Path,
    use_gpu: bool = True
) -> bool:
    """
    Run COLMAP dense reconstruction pipeline.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output files
        sparse_model: Path to sparse reconstruction model
        use_gpu: Use GPU acceleration (required for patch_match_stereo)
    
    Returns:
        True if reconstruction succeeded
    """
    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Image Undistortion
    timer = StageTimer("Image Undistortion")
    num_images = len(list(image_dir.glob("*.jpg")))
    timer.start(num_images)
    
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ]
    run_command(cmd, "Undistorting images for dense reconstruction")
    timer.finish()
    
    # Stage 2: Patch Match Stereo (GPU required)
    timer = StageTimer("Patch Match Stereo (Dense Depth)")
    timer.start(num_images)
    
    cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ]
    if use_gpu:
        cmd.extend(["--PatchMatchStereo.gpu_index", "0"])
    
    run_command(cmd, "Computing dense depth maps (this may take a while)")
    timer.finish()
    
    # Stage 3: Stereo Fusion
    timer = StageTimer("Stereo Fusion")
    timer.start(1)
    
    fused_ply = dense_dir / "fused.ply"
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply),
    ]
    run_command(cmd, "Fusing depth maps into dense point cloud")
    timer.finish()
    
    # Copy fused PLY to main output directory
    final_ply = output_dir / "dense_pointcloud.ply"
    if fused_ply.exists():
        shutil.copy(fused_ply, final_ply)
        print(f"  ✓ Dense point cloud: {final_ply}")
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to 3D reconstruction using COLMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input video file (MOV, MP4, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for reconstruction results"
    )
    
    # Frame extraction options
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Frames per second to extract (default: 3.0). "
             "Higher = more frames = better quality but slower. "
             "Recommended: 2-3 for walking, 5-10 for fast motion."
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1920,
        help="Maximum image dimension in pixels (default: 1920 for 1080p). "
             "Reducing from 4K (3840) to 1080p gives ~4x speedup."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to extract (default: 0 = unlimited). "
             "Useful for testing with a subset of the video."
    )
    
    # Feature extraction options
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum SIFT features per image (default: 8192). "
             "Range: 2048 (fast) to 8192 (high quality)."
    )
    
    # Matching options
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=10,
        help="Number of consecutive frames to match (default: 10). "
             "Higher = better for fast motion but slower."
    )
    parser.add_argument(
        "--loop-detection",
        action="store_true",
        help="Enable loop closure detection. Recommended for videos that "
             "revisit the same areas. Adds processing time."
    )
    
    # Reconstruction options
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Enable dense reconstruction after sparse. Significantly slower "
             "but produces detailed point clouds suitable for meshing."
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Run only sparse reconstruction (default behavior)."
    )
    
    # GPU options
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration. Not recommended - much slower."
    )
    
    # Quality presets
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "extreme"],
        default="high",
        help="Quality preset (default: high). Adjusts multiple parameters:\n"
             "  low: 1000px, 2048 features - fast testing\n"
             "  medium: 1600px, 4096 features - balanced\n"
             "  high: 1920px, 8192 features - good quality\n"
             "  extreme: 3200px, 8192 features - best quality, slowest"
    )
    
    args = parser.parse_args()
    
    # Apply quality presets (can be overridden by explicit args)
    quality_presets = {
        "low": {"max_image_size": 1000, "max_num_features": 2048},
        "medium": {"max_image_size": 1600, "max_num_features": 4096},
        "high": {"max_image_size": 1920, "max_num_features": 8192},
        "extreme": {"max_image_size": 3200, "max_num_features": 8192},
    }
    preset = quality_presets[args.quality]
    
    # Use preset values if not explicitly set
    if args.max_image_size == 1920:  # default value
        args.max_image_size = preset["max_image_size"]
    if args.max_num_features == 8192:  # default value
        args.max_num_features = preset["max_num_features"]
    
    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)
    
    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    database_path = args.output / "database.db"
    
    # Print configuration
    print("\n" + "="*60)
    print("VIDEO TO SFM PIPELINE - COLMAP")
    print("="*60)
    print(f"Input video:      {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Quality preset:   {args.quality}")
    print(f"FPS:              {args.fps}")
    print(f"Max image size:   {args.max_image_size}px")
    print(f"Max features:     {args.max_num_features}")
    print(f"Seq. overlap:     {args.sequential_overlap}")
    print(f"Loop detection:   {args.loop_detection}")
    print(f"Dense recon:      {args.dense}")
    print(f"GPU enabled:      {not args.no_gpu}")
    if args.max_frames > 0:
        print(f"Max frames:       {args.max_frames}")
    print("="*60)
    
    total_start = time.time()
    
    # Remove existing database to start fresh
    if database_path.exists():
        database_path.unlink()
    
    # Stage 1: Extract frames
    num_frames = extract_frames(
        video_path=args.input,
        output_dir=frames_dir,
        fps=args.fps,
        max_image_size=args.max_image_size,
        max_frames=args.max_frames
    )
    
    if num_frames == 0:
        print("ERROR: No frames extracted from video")
        sys.exit(1)
    
    # Stage 2: Sparse reconstruction
    success = run_sparse_reconstruction(
        image_dir=frames_dir,
        output_dir=args.output,
        database_path=database_path,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
        sequential_overlap=args.sequential_overlap,
        loop_detection=args.loop_detection,
        use_gpu=not args.no_gpu
    )
    
    if not success:
        print("ERROR: Sparse reconstruction failed")
        sys.exit(1)
    
    # Stage 3: Dense reconstruction (optional)
    if args.dense:
        sparse_model = args.output / "sparse" / "0"
        if sparse_model.exists():
            success = run_dense_reconstruction(
                image_dir=frames_dir,
                output_dir=args.output,
                sparse_model=sparse_model,
                use_gpu=not args.no_gpu
            )
            if not success:
                print("WARNING: Dense reconstruction failed")
        else:
            print("WARNING: No sparse model found for dense reconstruction")
    
    # Final summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {format_time(total_time)}")
    print(f"\nOutput files in {args.output}:")
    
    for f in sorted(args.output.glob("*.ply")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    sparse_dir = args.output / "sparse" / "0"
    if sparse_dir.exists():
        print(f"  • sparse/0/ (COLMAP model files)")
    
    if args.dense:
        dense_dir = args.output / "dense"
        if dense_dir.exists():
            print(f"  • dense/ (dense reconstruction files)")
    
    print("="*60)


if __name__ == "__main__":
    main()
