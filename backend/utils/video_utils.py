"""
Video and audio helper utilities for real-time and batch analysis workflows.
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional
import shutil

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> Optional[Path]:
    """
    Extract audio from a video file and return the path to a temporary WAV file.

    Attempts multiple strategies (moviepy, librosa, ffmpeg) for resilience on different
    platforms. Returns None if extraction ultimately fails.
    """
    try:
        # Create temporary audio file
        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
        logger.info("Extracting audio from video: %s -> %s", video_path, audio_path)

        # Try moviepy first for best codec coverage
        try:
            import moviepy

            video_clip = moviepy.VideoFileClip(str(video_path))
            audio_clip = video_clip.audio

            if audio_clip is not None:
                audio_clip.write_audiofile(str(audio_path), fps=target_sr)
                audio_clip.close()
                video_clip.close()
                if audio_path.exists():
                    logger.info("Audio extracted using moviepy: %s", audio_path)
                    return audio_path
            else:
                logger.warning("No audio track found in video %s", video_path)
                return None
        except ImportError:
            logger.debug("moviepy not available, falling back to librosa/ffmpeg")
        except Exception as moviepy_error:
            logger.warning("moviepy failed: %s, trying alternative method", moviepy_error)

        # Fallback: librosa direct load
        try:
            import librosa
            import soundfile as sf

            y, sr = librosa.load(str(video_path), sr=target_sr)
            sf.write(str(audio_path), y, sr)

            if audio_path.exists():
                logger.info("Audio extracted using librosa: %s", audio_path)
                return audio_path
            logger.error("Librosa extraction did not produce audio file")
        except ImportError:
            logger.debug("librosa not available for audio extraction")
        except Exception as librosa_error:
            logger.warning("Librosa failed to read %s: %s", video_path, librosa_error)

        # Final fallback: ffmpeg command line
        try:
            import imageio_ffmpeg as ffmpeg

            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_path,
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(target_sr),
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ]
            logger.debug("Running ffmpeg command: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and audio_path.exists():
                logger.info("Audio extracted using ffmpeg: %s", audio_path)
                return audio_path
            logger.error("FFmpeg extraction failed: %s", result.stderr)
        except Exception as ffmpeg_error:
            logger.warning("FFmpeg audio extraction failed: %s", ffmpeg_error)

        logger.error(
            "All extraction methods failed. Ensure the video has an audio track and supported codec."
        )
        return None
    except Exception as exc:
        logger.exception("Error extracting audio from %s: %s", video_path, exc)
        return None

def generate_video_thumbnail(video_path: Path, output_path: Path, timestamp: float = 1.0) -> Optional[Path]:
    """
    Capture a single frame from the video to use as a thumbnail.

    Attempts moviepy first, then ffmpeg via imageio_ffmpeg. Returns the
    output_path on success, or None if generation fails.
    """
    try:
        logger.info("Generating thumbnail for %s -> %s", video_path, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer moviepy for in-process frame extraction
        try:
            import moviepy

            clip = moviepy.VideoFileClip(str(video_path))
            duration = clip.duration or 0
            capture_time = min(max(timestamp, 0), max(duration - 0.1, 0))
            frame = clip.get_frame(capture_time)
            from PIL import Image

            image = Image.fromarray(frame)
            image.save(output_path, format="JPEG", quality=90)
            clip.close()
            logger.info("Thumbnail generated using moviepy at %ss", capture_time)
            return output_path
        except ImportError:
            logger.debug("moviepy not available for thumbnail generation")
        except Exception as moviepy_error:
            logger.warning("moviepy thumbnail extraction failed: %s", moviepy_error)

        # Fallback to ffmpeg if available
        try:
            import imageio_ffmpeg as ffmpeg

            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_path,
                "-ss",
                str(max(timestamp, 0)),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(output_path),
            ]
            logger.debug("Running ffmpeg for thumbnail: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and output_path.exists():
                logger.info("Thumbnail generated using ffmpeg")
                return output_path
            logger.error("FFmpeg thumbnail generation failed: %s", result.stderr)
        except ImportError:
            logger.debug("imageio_ffmpeg not installed; cannot fallback to ffmpeg for thumbnails")
        except Exception as ffmpeg_error:
            logger.warning("FFmpeg thumbnail generation failed: %s", ffmpeg_error)

        # As a final fallback, attempt to copy a default placeholder if provided
        placeholder_dir = Path(__file__).parent / "placeholders"
        for extension in (".jpg", ".jpeg", ".png"):
            placeholder = placeholder_dir / f"default_thumbnail{extension}"
            if placeholder.exists():
                shutil.copy2(placeholder, output_path)
                logger.info("Thumbnail placeholder copied from %s", placeholder)
                return output_path

        logger.error("Unable to generate thumbnail for %s", video_path)
        return None
    except Exception as exc:
        logger.exception("Error creating thumbnail for %s: %s", video_path, exc)
        return None

