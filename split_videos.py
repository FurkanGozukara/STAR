import argparse
import logging
import os
import sys
from pathlib import Path

# Imports adjusted for PySceneDetect v0.7.dev0 structure
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    CropRegion,
    FrameTimecode,
    Interpolation,
    SceneManager,
    open_video,
)
from scenedetect.backends import AVAILABLE_BACKENDS
from scenedetect.output import (
    SceneMetadata,
    VideoMetadata,
    is_ffmpeg_available,
    is_mkvmerge_available,
    split_video_ffmpeg,
    split_video_mkvmerge,
)
from scenedetect.video_stream import VideoOpenFailure
import scenedetect.platform # For scenedetect.platform.init_logger
import scenedetect # For StatsManager in v0.7+

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def custom_filename_formatter(video: VideoMetadata, scene: SceneMetadata) -> str:
    """
    Custom formatter for output filenames.
    Format: original_file_name_frame_begin_frame_end.mp4
    """
    # Use video.name which is the video filename without extension
    # Add the .mp4 extension (or your preferred extension)
    return f"{video.name}_frame_{scene.start.get_frames()}-{scene.end.get_frames()}.mp4"


def parse_weights(weights_str: str) -> ContentDetector.Components:
    """Parse a comma or space-separated string of 4 floats into ContentDetector.Components."""
    try:
        parts = [float(w.strip()) for w in weights_str.replace(",", " ").split()]
        if len(parts) != 4:
            raise ValueError("Weights must be 4 floating-point numbers (H,S,L,E).")
        return ContentDetector.Components(delta_hue=parts[0], delta_sat=parts[1], delta_lum=parts[2], delta_edges=parts[3])
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_crop_region(crop_str: str) -> CropRegion:
    """Parse a string of 4 ints (X0 Y0 X1 Y1) into a CropRegion tuple."""
    try:
        parts = [int(c.strip()) for c in crop_str.split()]
        if len(parts) != 4:
            raise ValueError("Crop region must be 4 integers (X0 Y0 X1 Y1).")
        return tuple(parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Split a video into scenes with frame-accurate cuts. Uses AdaptiveDetector by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input/Output Arguments ---
    group_io = parser.add_argument_group("Input/Output Options")
    group_io.add_argument(
        "-i",
        "--input",
        type=str,
        default="input.mp4",
        help="Path to the input video file.",
    )
    group_io.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="split_videos",
        help="Directory to save the split video clips.",
    )
    group_io.add_argument(
        "--stats",
        "-s",
        type=str,
        default=None,
        metavar="CSV_FILE",
        help="Path to save a CSV file with frame metrics (e.g., for tuning).",
    )
    group_io.add_argument(
        "--logfile",
        "-l",
        type=str,
        default=None,
        metavar="LOG_FILE",
        help="Path to save a detailed log file.",
    )
    group_io.add_argument(
        "--verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "none"],
        help="Level of detail for console log messages.",
    )
    progress_group = group_io.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show a progress bar during processing (default).",
    )
    progress_group.add_argument(
        "--no-show-progress",
        action="store_false",
        dest="show_progress",
        help="Disable the progress bar.",
    )
    group_io.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all console output (overrides verbosity and show-progress).",
    )

    # --- Global Processing Options ---
    group_global_proc = parser.add_argument_group("Global Processing Options")
    group_global_proc.add_argument(
        "--framerate",
        "-f",
        type=float,
        default=None,
        metavar="FPS",
        help="Override detected video framerate (frames per second).",
    )
    group_global_proc.add_argument(
        "--min-scene-len",
        "-m",
        type=str,
        default="0.6s",
        help="Minimum length of any scene. Format: frames (e.g., 15), "
        "seconds (e.g., 0.5s), or timecode (e.g., 00:00:00.500).",
    )
    group_global_proc.add_argument(
        "--drop-short-scenes",
        action="store_true",
        help="Drop scenes shorter than --min-scene-len instead of merging with neighbors.",
    )
    group_global_proc.add_argument(
        "--merge-last-scene",
        action="store_true",
        help="Merge the last scene with the previous one if it's shorter than --min-scene-len.",
    )
    group_global_proc.add_argument(
        "--backend",
        "-b",
        type=str,
        default="opencv",
        choices=list(AVAILABLE_BACKENDS.keys()),
        help="Video processing backend to use.",
    )
    group_global_proc.add_argument(
        "--crop",
        type=parse_crop_region,
        default=None,
        metavar="'X0 Y0 X1 Y1'",
        help="Crop input video before processing. Format: 'X0 Y0 X1 Y1' (e.g., '10 20 640 480').",
    )
    group_global_proc.add_argument(
        "--downscale",
        type=int,
        default=0,
        metavar="N",
        help="Integer factor to downscale video by (e.g., 2 means 1/2 resolution). "
        "0 for auto (default), 1 for no downscaling.",
    )
    group_global_proc.add_argument(
        "--downscale-method",
        type=str,
        default="linear",
        choices=[interp.name.lower() for interp in Interpolation],
        help="Interpolation method for downscaling."
    )
    group_global_proc.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        metavar="N",
        help="Number of frames to skip between processed frames (0 for no skip). "
        "Reduces accuracy but speeds up processing. Not recommended with stats file.",
    )

    # --- Time Options ---
    group_time = parser.add_argument_group("Time Control Options (for processing a video segment)")
    group_time.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="TIMECODE",
        help="Time to start processing from. Format: frames, seconds (e.g. 10.5s), or HH:MM:SS.sss.",
    )
    group_time.add_argument(
        "--duration",
        type=str,
        default=None,
        metavar="TIMECODE",
        help="Maximum duration to process. Mutually exclusive with --end.",
    )
    group_time.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="TIMECODE",
        help="Time to stop processing at. Mutually exclusive with --duration.",
    )

    # --- Adaptive Detector Options ---
    group_adaptive = parser.add_argument_group("Adaptive Detector Options (`detect-adaptive`)")
    group_adaptive.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=3.0,
        help="Threshold for AdaptiveDetector (adaptive_ratio). Lower is more sensitive.",
    )
    group_adaptive.add_argument(
        "--min-content-val",
        type=float,
        default=15.0,
        metavar="VAL",
        help="Minimum 'content_val' (from ContentDetector) needed to trigger a cut for AdaptiveDetector.",
    )
    group_adaptive.add_argument(
        "--frame-window", # Command-line argument name
        type=int,
        default=2,
        metavar="FRAMES",
        help="Size of the window (number of frames before and after) for AdaptiveDetector's rolling average.",
    )
    group_adaptive.add_argument(
        "--weights",
        type=parse_weights,
        default="1.0,1.0,1.0,0.0",
        metavar="'H S L E'",
        help="Comma or space-separated weights for Hue, Saturation, Luma, Edges components. Example: '1.0 1.0 1.0 0.0'.",
    )
    group_adaptive.add_argument(
        "--luma-only",
        action="store_true",
        help="Only use luma (brightness) for content scoring. Overrides --weights to 0,0,1,0.",
    )
    group_adaptive.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        metavar="N",
        help="Size of kernel for edge detection (if used). Must be odd, >=3. Default: auto.",
    )

    # --- Video Splitting Options ---
    group_split = parser.add_argument_group("Video Splitting Options (`split-video`)")
    group_split.add_argument(
        "--ffmpeg-args",
        type=str,
        default=None,
        help="Custom arguments to pass directly to ffmpeg for splitting. Overrides other encoding quality flags.",
    )
    group_split.add_argument(
        "--copy-streams",
        action="store_true",
        help="Use ffmpeg to copy video/audio streams without re-encoding (faster, but cuts only on keyframes).",
    )
    group_split.add_argument(
        "--mkvmerge",
        action="store_true",
        help="Use mkvmerge to split video (faster, cuts only on keyframes, output is MKV).",
    )
    group_split.add_argument(
        "--high-quality",
        "-hq",
        action="store_true",
        help="Use high-quality ffmpeg re-encoding settings (slower, larger files). "
             "Equivalent to --rate-factor 17 --preset slow.",
    )
    group_split.add_argument(
        "--rate-factor",
        type=int,
        default=22,
        metavar="CRF",
        help="Constant Rate Factor for ffmpeg (libx264/libx265). Lower is better quality (0-51).",
    )
    group_split.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=[
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow", "placebo"
        ],
        help="Encoding preset for ffmpeg (libx264/libx265). Affects speed vs. compression.",
    )
    group_split.add_argument(
        "--quiet-ffmpeg",
        action="store_true",
        help="Suppress ffmpeg's console output during splitting.",
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbosity = "none"
        args.show_progress = False

    scenedetect.platform.init_logger(
        log_level=getattr(logging, args.verbosity.upper(), logging.INFO),
        show_stdout=(args.verbosity.lower() != "none"),
        log_file=args.logfile,
    )

    input_video_path = Path(args.input)
    output_dir_path = Path(args.output_dir)

    if not input_video_path.is_file():
        logger.error(f"Error: Input video file not found at '{input_video_path}'")
        sys.exit(1)

    if args.mkvmerge and not is_mkvmerge_available():
        logger.error("Error: --mkvmerge specified, but mkvmerge not found in PATH.")
        sys.exit(1)
    if not args.mkvmerge and not is_ffmpeg_available():
        logger.error("Error: ffmpeg not found in PATH. Required for splitting unless --mkvmerge is used.")
        sys.exit(1)

    if args.copy_streams and args.mkvmerge: logger.warning("--copy-streams ignored with --mkvmerge.")
    if args.ffmpeg_args:
        if args.copy_streams: logger.warning("--copy-streams ignored with --ffmpeg-args.")
        if args.high_quality: logger.warning("--high-quality ignored with --ffmpeg-args.")
        if args.preset != parser.get_default('preset'): logger.warning("--preset ignored with --ffmpeg-args.")
        if args.rate_factor != parser.get_default('rate_factor'): logger.warning("--rate-factor ignored with --ffmpeg-args.")
    elif args.copy_streams:
        if args.high_quality: logger.warning("--high-quality ignored with --copy-streams.")
        if args.preset != parser.get_default('preset'): logger.warning("--preset ignored with --copy-streams.")
        if args.rate_factor != parser.get_default('rate_factor'): logger.warning("--rate-factor ignored with --copy-streams.")
    elif args.mkvmerge:
        if args.high_quality: logger.warning("--high-quality ignored with --mkvmerge.")
        if args.preset != parser.get_default('preset'): logger.warning("--preset ignored with --mkvmerge.")
        if args.rate_factor != parser.get_default('rate_factor'): logger.warning("--rate-factor ignored with --mkvmerge.")

    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: '{output_dir_path.resolve()}'")
    except OSError as e:
        logger.error(f"Error creating output directory '{output_dir_path}': {e}")
        sys.exit(1)

    video = None
    try:
        logger.info(f"Opening video: '{input_video_path}' using backend: {args.backend}...")
        video_params = {}
        if args.backend == "pyav": video_params = {"threading_mode": "AUTO", "suppress_output": False}
        elif args.backend == "opencv": video_params = {"max_decode_attempts": 5}
        video = open_video(str(input_video_path), framerate=args.framerate, backend=args.backend, **video_params)

    except VideoOpenFailure as e:
        logger.error(f"Error opening video file '{input_video_path}': {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Error with video parameters: {e}")
        sys.exit(1)

    stats_manager = scenedetect.StatsManager() if args.stats else None # Use scenedetect.StatsManager
    scene_manager = SceneManager(stats_manager=stats_manager)

    scene_manager.auto_downscale = (args.downscale == 0)
    if args.downscale > 0:
        scene_manager.downscale = args.downscale
    scene_manager.interpolation = Interpolation[args.downscale_method.upper()]

    if args.crop:
        try:
            scene_manager.crop = args.crop
        except ValueError as e:
            logger.error(f"Invalid --crop region: {e}")
            sys.exit(1)

    detector_weights = args.weights
    if args.luma_only:
        detector_weights = ContentDetector.LUMA_ONLY_WEIGHTS

    try:
        detector_min_scene_len_ft = FrameTimecode(args.min_scene_len, video.frame_rate)
    except ValueError as e:
        logger.error(f"Invalid --min-scene-len format for detector: {e}")
        sys.exit(1)

    scene_manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=args.threshold,
            min_content_val=args.min_content_val,
            window_width=args.frame_window, # Corrected argument name
            weights=detector_weights,
            kernel_size=args.kernel_size,
            min_scene_len=detector_min_scene_len_ft
        )
    )
    logger.info(f"Using AdaptiveDetector with threshold: {args.threshold}")

    start_time_ft = None
    if args.start:
        try: start_time_ft = FrameTimecode(args.start, video.frame_rate)
        except ValueError as e: logger.error(f"Invalid --start time format: {e}"); sys.exit(1)

    end_time_ft = None
    if args.end:
        try: end_time_ft = FrameTimecode(args.end, video.frame_rate)
        except ValueError as e: logger.error(f"Invalid --end time format: {e}"); sys.exit(1)

    duration_ft = None
    if args.duration:
        try: duration_ft = FrameTimecode(args.duration, video.frame_rate)
        except ValueError as e: logger.error(f"Invalid --duration time format: {e}"); sys.exit(1)

    if args.end and args.duration:
        logger.error("Cannot specify both --end and --duration.")
        sys.exit(1)

    try:
        if start_time_ft:
            logger.info(f"Seeking to start time: {start_time_ft.get_timecode()}")
            video.seek(start_time_ft)

        logger.info("Detecting scenes...")
        num_frames_processed = scene_manager.detect_scenes(
            video=video,
            end_time=end_time_ft,
            duration=duration_ft,
            frame_skip=args.frame_skip,
            show_progress=args.show_progress,
        )
        logger.info(f"Processed {num_frames_processed} frames.")

        scene_list = scene_manager.get_scene_list(start_in_scene=True)

        try: global_min_scene_len_ft = FrameTimecode(args.min_scene_len, video.frame_rate)
        except ValueError as e: logger.error(f"Invalid global --min-scene-len format: {e}"); sys.exit(1)

        filtered_scene_list = []
        if scene_list:
            for i, (start, end) in enumerate(scene_list):
                current_scene_duration = end - start
                if current_scene_duration < global_min_scene_len_ft:
                    if args.drop_short_scenes:
                        logger.info(f"Dropping short scene ({i+1}): {start.get_timecode()} - {end.get_timecode()} (duration: {current_scene_duration.get_timecode()})")
                        continue
                    elif filtered_scene_list and not (args.merge_last_scene and i == len(scene_list) -1 and len(scene_list) > 1) :
                        prev_start, _ = filtered_scene_list.pop()
                        filtered_scene_list.append((prev_start, end))
                        logger.info(f"Merging short scene ({i+1}) with previous. New merged scene ends at: {end.get_timecode()}")
                        continue
                filtered_scene_list.append((start, end))
            scene_list = filtered_scene_list

            if args.merge_last_scene and len(scene_list) > 1:
                last_start, last_end = scene_list[-1]
                if (last_end - last_start) < global_min_scene_len_ft:
                    logger.info(f"Merging last scene ({last_start.get_timecode()} - {last_end.get_timecode()}) with previous as it is shorter than min_scene_len.")
                    prev_start, _ = scene_list[-2]
                    scene_list = scene_list[:-2] + [(prev_start, last_end)]

        if not scene_list:
            logger.info("No scenes meet criteria after all filtering.")
            sys.exit(0)

        logger.info(f"Found {len(scene_list)} scenes after all filtering. Splitting video...")

        final_ffmpeg_args = ""
        if args.ffmpeg_args: final_ffmpeg_args = args.ffmpeg_args
        elif args.copy_streams: final_ffmpeg_args = "-map 0:v:0 -map 0:a? -map 0:s? -c:v copy -c:a copy -avoid_negative_ts make_zero"
        elif args.high_quality: final_ffmpeg_args = "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset slow -crf 17 -c:a aac -avoid_negative_ts make_zero"
        else: final_ffmpeg_args = (f"-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset {args.preset} -crf {args.rate_factor} -c:a aac -avoid_negative_ts make_zero")

        if args.mkvmerge:
            logger.info("Using mkvmerge for splitting.")
            return_code = split_video_mkvmerge(
                input_video_path=str(input_video_path), scene_list=scene_list, output_dir=str(output_dir_path),
                video_name=video.name, show_output=(not args.quiet_ffmpeg), formatter=custom_filename_formatter
            )
        else:
            logger.info(f"Using ffmpeg for splitting with args: {final_ffmpeg_args}")
            return_code = split_video_ffmpeg(
                input_video_path=str(input_video_path), scene_list=scene_list, output_dir=str(output_dir_path),
                video_name=video.name, arg_override=final_ffmpeg_args, show_progress=args.show_progress,
                show_output=(not args.quiet_ffmpeg), formatter=custom_filename_formatter,
            )

        if return_code == 0: logger.info("Video splitting completed successfully.")
        else: logger.error(f"Video splitting failed. {'mkvmerge' if args.mkvmerge else 'ffmpeg'} returned code: {return_code}")

        if args.stats and stats_manager:
            stats_path = Path(args.stats)
            try:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                if stats_manager._base_timecode is None and video: stats_manager._base_timecode = video.base_timecode
                with open(stats_path, "w", newline="") as f: stats_manager.save_to_csv(csv_file=f)
                logger.info(f"Frame metrics saved to: {stats_path}")
            except IOError as e: logger.error(f"Error saving stats file '{stats_path}': {e}")
            except AttributeError as e: logger.error(f"Error saving stats file, possibly missing base_timecode: {e}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if video is not None:
            del video

if __name__ == "__main__":
    main()