"""
Video Processing Module

Handles:
1. Smart cropping with face detection (16:9 to 9:16 conversion)
2. Dynamic caption generation and burning
3. Audio muxing for multi-lingual output
4. Reel generation with viral moments

Uses FFmpeg, MoviePy, and MediaPipe for video processing.
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import asyncio

import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip
)
from moviepy.video.fx.all import resize

from config import get_settings
from models import ViralMoment, TranscriptSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass


@dataclass
class CaptionStyle:
    """Style configuration for captions."""
    font: str = "Arial-Bold"
    font_size: int = 60
    color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    bg_color: Optional[str] = None
    position: Tuple[str, str] = ("center", "bottom")
    margin_bottom: int = 150
    highlight_color: str = "#FF6B35"  # Orange highlight for keywords
    word_highlight: bool = True


class FaceTracker:
    """
    Face detection and tracking using MediaPipe.
    
    Keeps the speaker centered when converting landscape to portrait video.
    """
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short range, 1=full range
            min_detection_confidence=0.5
        )
        self.tracking_history: List[Tuple[float, float]] = []
        self.max_history = 30  # Keep last 30 frames for smoothing
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Detect faces in a frame.
        
        Returns:
            List of (x, y, width, height) normalized coordinates
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = bbox.xmin
                y = bbox.ymin
                width = bbox.width
                height = bbox.height
                faces.append((x, y, width, height))
        
        return faces
    
    def get_center_of_interest(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Get the point of interest (face center) for cropping.
        
        Returns:
            (x, y) normalized coordinates (0-1)
        """
        faces = self.detect_faces(frame)
        
        if faces:
            # Use the largest face (closest to camera)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            center_x = x + w / 2
            center_y = y + h / 2
            
            self.tracking_history.append((center_x, center_y))
            
            # Keep history limited
            if len(self.tracking_history) > self.max_history:
                self.tracking_history.pop(0)
            
            # Return smoothed center
            avg_x = sum(p[0] for p in self.tracking_history) / len(self.tracking_history)
            avg_y = sum(p[1] for p in self.tracking_history) / len(self.tracking_history)
            
            return (avg_x, avg_y)
        
        # No face detected, use previous center or default
        if self.tracking_history:
            return self.tracking_history[-1]
        
        return (0.5, 0.5)  # Default center
    
    def reset(self):
        """Reset tracking history."""
        self.tracking_history.clear()


class CaptionGenerator:
    """
    Generate "Alex Hormozi style" animated captions.
    
    Features:
    - Word-by-word highlighting
    - Bold, readable fonts
    - High contrast with stroke
    - Smooth animations
    """
    
    def __init__(self, style: Optional[CaptionStyle] = None):
        self.style = style or CaptionStyle()
        self.settings = get_settings()
    
    def create_caption_clip(
        self,
        text: str,
        duration: float,
        video_size: Tuple[int, int],
        highlight_words: Optional[List[str]] = None
    ) -> TextClip:
        """
        Create a caption text clip.
        
        Args:
            text: Caption text
            duration: Display duration
            video_size: (width, height) of target video
            highlight_words: Words to highlight
            
        Returns:
            MoviePy TextClip
        """
        # Wrap text for readability
        wrapped_text = self._wrap_text(text, max_chars_per_line=25)
        
        # Create base text clip
        txt_clip = TextClip(
            wrapped_text,
            fontsize=self.style.font_size,
            color=self.style.color,
            stroke_color=self.style.stroke_color,
            stroke_width=self.style.stroke_width,
            font=self.style.font,
            method='caption',
            size=(video_size[0] - 100, None),  # Leave margins
            align='center'
        ).set_duration(duration)
        
        # Position at bottom center
        txt_clip = txt_clip.set_position((
            self.style.position[0],
            video_size[1] - txt_clip.h - self.style.margin_bottom
        ))
        
        return txt_clip
    
    def create_animated_captions(
        self,
        transcript_segments: List[TranscriptSegment],
        video_duration: float,
        video_size: Tuple[int, int]
    ) -> List[TextClip]:
        """
        Create word-by-word animated captions.
        
        Args:
            transcript_segments: Transcript with timestamps
            video_duration: Total video duration
            video_size: (width, height)
            
        Returns:
            List of TextClip objects
        """
        clips = []
        
        for segment in transcript_segments:
            # Skip very short segments
            if segment.end - segment.start < 0.3:
                continue
            
            clip = self.create_caption_clip(
                segment.text,
                segment.end - segment.start,
                video_size
            ).set_start(segment.start)
            
            clips.append(clip)
        
        return clips
    
    def _wrap_text(self, text: str, max_chars_per_line: int = 25) -> str:
        """Wrap text into multiple lines for better readability."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)


class VideoProcessor:
    """
    Main video processing orchestrator.
    
    Handles the complete pipeline:
    1. Load video and audio
    2. Detect faces and track movement
    3. Convert 16:9 to 9:16 with smart cropping
    4. Add captions
    5. Export final reel
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.face_tracker = FaceTracker()
        self.caption_generator = CaptionGenerator()
        self.output_path = Path(self.settings.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    async def process_reel(
        self,
        video_path: Path,
        audio_path: Path,
        viral_moment: ViralMoment,
        transcript_segments: List[TranscriptSegment],
        output_filename: Optional[str] = None,
        add_captions: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """
        Process a single viral moment into a reel.
        
        Args:
            video_path: Source video file
            audio_path: Audio file (can be different language)
            viral_moment: The viral moment to extract
            transcript_segments: Transcript for caption generation
            output_filename: Optional output filename
            add_captions: Whether to add captions
            progress_callback: Optional progress callback (0-100)
            
        Returns:
            Path to generated reel
        """
        try:
            loop = asyncio.get_event_loop()
            
            def run_processing():
                return self._process_reel_sync(
                    video_path,
                    audio_path,
                    viral_moment,
                    transcript_segments,
                    output_filename,
                    add_captions,
                    progress_callback
                )
            
            return await loop.run_in_executor(None, run_processing)
            
        except Exception as e:
            logger.error(f"Reel processing error: {str(e)}")
            raise VideoProcessingError(f"Failed to process reel: {str(e)}")
    
    def _process_reel_sync(
        self,
        video_path: Path,
        audio_path: Path,
        viral_moment: ViralMoment,
        transcript_segments: List[TranscriptSegment],
        output_filename: Optional[str] = None,
        add_captions: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Synchronous version of process_reel."""
        
        # Load video
        logger.info(f"Loading video: {video_path}")
        video = VideoFileClip(str(video_path))
        
        # Extract segment
        start_time = viral_moment.start_time
        end_time = viral_moment.end_time
        
        logger.info(f"Extracting segment: {start_time:.1f}s - {end_time:.1f}s")
        segment = video.subclip(start_time, end_time)
        
        if progress_callback:
            progress_callback(20)
        
        # Convert to 9:16 with face tracking
        logger.info("Converting to 9:16 with face tracking...")
        vertical_segment = self._convert_to_vertical(segment)
        
        if progress_callback:
            progress_callback(50)
        
        # Load and sync audio
        logger.info(f"Loading audio: {audio_path}")
        audio = AudioFileClip(str(audio_path))
        audio_segment = audio.subclip(start_time, end_time)
        
        # Set audio
        vertical_segment = vertical_segment.set_audio(audio_segment)
        
        if progress_callback:
            progress_callback(60)
        
        clips_to_compose = [vertical_segment]
        
        # Add captions if requested
        if add_captions:
            logger.info("Generating captions...")
            
            # Filter transcript to this segment
            segment_transcript = [
                TranscriptSegment(
                    start=seg.start - start_time,
                    end=seg.end - start_time,
                    text=seg.text,
                    confidence=seg.confidence
                )
                for seg in transcript_segments
                if start_time <= seg.start < end_time
            ]
            
            caption_clips = self.caption_generator.create_animated_captions(
                segment_transcript,
                vertical_segment.duration,
                vertical_segment.size
            )
            
            clips_to_compose.extend(caption_clips)
        
        if progress_callback:
            progress_callback(80)
        
        # Compose final video
        logger.info("Composing final video...")
        final = CompositeVideoClip(clips_to_compose)
        
        # Generate output filename
        if not output_filename:
            output_filename = f"reel_{viral_moment.start_time:.0f}_{viral_moment.end_time:.0f}.mp4"
        
        output_path = self.output_path / output_filename
        
        # Export
        logger.info(f"Exporting to: {output_path}")
        final.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(self.settings.temp_path / 'temp_audio.m4a'),
            remove_temp=True,
            fps=30,
            preset='medium',
            threads=4
        )
        
        # Cleanup
        video.close()
        segment.close()
        audio.close()
        vertical_segment.close()
        final.close()
        
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Reel exported: {output_path}")
        return output_path
    
    def _convert_to_vertical(
        self,
        clip: VideoFileClip,
        target_ratio: float = 9/16
    ) -> VideoFileClip:
        """
        Convert landscape video to portrait with face tracking.
        
        Args:
            clip: Input video clip (typically 16:9)
            target_ratio: Target aspect ratio (default 9:16)
            
        Returns:
            Cropped portrait video clip
        """
        original_w, original_h = clip.size
        target_w = int(original_h * target_ratio)
        
        # Target dimensions
        target_size = (target_w, original_h)
        
        # If video is already portrait or square, just resize
        if original_w <= original_h:
            return resize(clip, height=1920)  # Standard vertical height
        
        # Process frames with face tracking
        def process_frame(get_frame, t):
            frame = get_frame(t)
            
            # Get center of interest
            center_x, _ = self.face_tracker.get_center_of_interest(frame)
            
            # Calculate crop region
            crop_x = int(center_x * original_w - target_w / 2)
            
            # Clamp to video bounds
            crop_x = max(0, min(crop_x, original_w - target_w))
            
            # Crop frame
            cropped = frame[:, crop_x:crop_x + target_w]
            
            return cropped
        
        # Apply transformation
        vertical_clip = clip.fl(process_frame)
        
        return vertical_clip
    
    async def mux_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Mux (combine) video with different audio track.
        
        Args:
            video_path: Video file
            audio_path: Audio file (different language)
            output_path: Optional output path
            
        Returns:
            Path to muxed video
        """
        if not output_path:
            output_path = self.output_path / f"{video_path.stem}_muxed.mp4"
        
        try:
            # Use FFmpeg for efficient muxing
            cmd = [
                self.settings.ffmpeg_path,
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', 'aac',   # Encode audio to AAC
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                '-shortest',      # End when shortest input ends
                '-y',             # Overwrite output
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise VideoProcessingError(f"FFmpeg error: {stderr.decode()}")
            
            logger.info(f"Audio muxed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Muxing error: {str(e)}")
            raise VideoProcessingError(f"Failed to mux audio: {str(e)}")
    
    async def generate_reels_batch(
        self,
        video_path: Path,
        audio_path: Path,
        viral_moments: List[ViralMoment],
        transcript_segments: List[TranscriptSegment],
        max_reels: int = 10
    ) -> List[Path]:
        """
        Generate multiple reels from viral moments.
        
        Args:
            video_path: Source video
            audio_path: Audio file
            viral_moments: List of viral moments
            transcript_segments: Transcript segments
            max_reels: Maximum number of reels to generate
            
        Returns:
            List of paths to generated reels
        """
        output_paths = []
        
        for i, moment in enumerate(viral_moments[:max_reels]):
            try:
                logger.info(f"Generating reel {i+1}/{min(len(viral_moments), max_reels)}")
                
                output_filename = f"reel_{i+1:02d}_{moment.start_time:.0f}.mp4"
                
                reel_path = await self.process_reel(
                    video_path,
                    audio_path,
                    moment,
                    transcript_segments,
                    output_filename
                )
                
                output_paths.append(reel_path)
                
            except Exception as e:
                logger.error(f"Failed to generate reel {i+1}: {str(e)}")
                continue
        
        return output_paths


# Singleton instance
_processor_instance: Optional[VideoProcessor] = None


def get_video_processor() -> VideoProcessor:
    """Get or create singleton processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = VideoProcessor()
    return _processor_instance
