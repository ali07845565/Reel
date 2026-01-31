"""
Viral Moment Detection Module

This module implements the "Intelligence" layer for detecting viral moments in videos:
1. Transcription using OpenAI Whisper (Large-v3)
2. LLM Analysis using GPT-4o or Gemini 1.5 Pro for hook detection
3. Audio Energy Analysis using Librosa for validation
4. Combined scoring for final viral moment selection

The module provides a pipeline that takes audio input and returns
timestamped viral moments with confidence scores.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import asyncio
import numpy as np

import whisper
import librosa
from openai import AsyncOpenAI
import google.generativeai as genai

from config import get_settings
from models import (
    TranscriptSegment, 
    ViralMoment, 
    AudioAnalysisResult,
    VideoMetadata
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class AnalysisError(Exception):
    """Raised when LLM analysis fails."""
    pass


class AudioAnalysisError(Exception):
    """Raised when audio energy analysis fails."""
    pass


@dataclass
class ViralDetectionConfig:
    """Configuration for viral moment detection."""
    # Whisper settings
    whisper_model: str = "large-v3"
    
    # LLM settings
    llm_provider: str = "openai"  # or "google"
    llm_model: str = "gpt-4o"  # or "gemini-1.5-pro"
    llm_temperature: float = 0.3
    
    # Audio analysis settings
    energy_window_size: float = 1.0  # seconds
    peak_detection_threshold: float = 0.7
    
    # Viral moment criteria
    min_segment_duration: float = 30.0  # seconds
    max_segment_duration: float = 90.0  # seconds
    target_segment_duration: float = 60.0  # seconds
    
    # Scoring weights
    llm_weight: float = 0.6
    energy_weight: float = 0.4
    
    # Number of moments to return
    max_moments: int = 10


class WhisperTranscriber:
    """
    OpenAI Whisper-based transcription with timestamp alignment.
    
    Uses the Large-v3 model for best accuracy across languages.
    """
    
    def __init__(self, model_name: str = "large-v3"):
        self.model_name = model_name
        self.model = None
        self._model_lock = asyncio.Lock()
    
    async def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            async with self._model_lock:
                if self.model is None:
                    logger.info(f"Loading Whisper model: {self.model_name}")
                    loop = asyncio.get_event_loop()
                    self.model = await loop.run_in_executor(
                        None, 
                        lambda: whisper.load_model(self.model_name)
                    )
                    logger.info("Whisper model loaded successfully")
    
    async def transcribe(
        self, 
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (auto-detect if None)
            task: "transcribe" or "translate"
            
        Returns:
            List of TranscriptSegment with timestamps
        """
        await self._load_model()
        
        try:
            loop = asyncio.get_event_loop()
            
            def run_transcription():
                options = {
                    "task": task,
                    "word_timestamps": True,
                    "verbose": False,
                }
                if language:
                    options["language"] = language
                
                result = self.model.transcribe(str(audio_path), **options)
                return result
            
            result = await loop.run_in_executor(None, run_transcription)
            
            # Parse segments
            segments = []
            for seg in result.get("segments", []):
                segment = TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    confidence=seg.get("avg_logprob", 0.0)
                )
                segments.append(segment)
            
            logger.info(f"Transcribed {len(segments)} segments from audio")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")
    
    async def transcribe_with_diarization(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None
    ) -> List[TranscriptSegment]:
        """
        Transcribe with speaker diarization (who spoke when).
        
        Note: This is a placeholder. Full diarization requires pyannote.audio
        or similar. For now, returns regular transcription.
        """
        return await self.transcribe(audio_path)


class LLMViralAnalyzer:
    """
    LLM-based viral moment detection using GPT-4o or Gemini.
    
    Analyzes transcript text to identify:
    - Hooks (attention-grabbing openings)
    - Climaxes (peak emotional/intellectual moments)
    - High-value insights (actionable knowledge)
    - Emotional peaks (laughter, surprise, inspiration)
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        self.settings = get_settings()
        
        if provider == "openai":
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        elif provider == "google":
            genai.configure(api_key=self.settings.google_api_key)
            self.client = None
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _build_prompt(
        self, 
        transcript: List[TranscriptSegment],
        video_metadata: Optional[VideoMetadata] = None,
        target_duration: float = 60.0,
        num_moments: int = 10
    ) -> str:
        """Build the prompt for viral moment detection."""
        
        # Format transcript for the LLM
        transcript_text = "\n".join([
            f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}"
            for seg in transcript
        ])
        
        video_context = ""
        if video_metadata:
            video_context = f"""
Video Title: {video_metadata.title}
Video Duration: {video_metadata.duration_string}
Channel: {video_metadata.uploader or 'Unknown'}
"""
        
        prompt = f"""You are an expert content strategist specializing in viral short-form video.
Your task is to analyze a video transcript and identify the most viral-worthy moments.

{video_context}

TRANSCRIPT:
{transcript_text}

ANALYSIS TASK:
Identify {num_moments} segments that have the highest viral potential for short-form content (TikTok, Reels, Shorts).

For each segment, provide:
1. Start and end timestamps (in seconds)
2. The hook/climax text (the key phrase that makes it viral)
3. Hook type: "hook" (attention grabber), "climax" (peak moment), "insight" (valuable knowledge), or "emotional" (laughter, surprise, inspiration)
4. Confidence score (0.0-1.0) for viral potential
5. Brief explanation of why this moment is viral-worthy
6. Key keywords/topics (3-5 tags)

CRITERIA FOR VIRAL MOMENTS:
- HOOKS: First 3 seconds MUST grab attention (controversy, curiosity gap, bold claim)
- CLIMAXES: Peak emotional or intellectual moments, revelations, plot twists
- INSIGHTS: Actionable advice, "aha" moments, valuable knowledge bombs
- EMOTIONAL: Laughter, surprise, inspiration, relatability, shared experiences

TARGET SEGMENT DURATION: {target_duration:.0f} seconds (range: 30-90s)

IMPORTANT: Return ONLY valid JSON in this exact format:
{{
  "viral_moments": [
    {{
      "start_time": 45.5,
      "end_time": 105.5,
      "hook_text": "The key viral phrase here...",
      "hook_type": "hook|climax|insight|emotional",
      "confidence_score": 0.92,
      "explanation": "Why this moment is viral-worthy...",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ]
}}

Analyze the transcript and return the JSON response."""
        
        return prompt
    
    async def analyze(
        self,
        transcript: List[TranscriptSegment],
        video_metadata: Optional[VideoMetadata] = None,
        target_duration: float = 60.0,
        num_moments: int = 10
    ) -> List[ViralMoment]:
        """
        Analyze transcript to find viral moments.
        
        Args:
            transcript: List of transcript segments
            video_metadata: Optional video metadata for context
            target_duration: Target duration for each viral moment
            num_moments: Number of moments to identify
            
        Returns:
            List of ViralMoment objects
        """
        prompt = self._build_prompt(transcript, video_metadata, target_duration, num_moments)
        
        try:
            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert viral content strategist. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.settings.llm_temperature,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
            elif self.provider == "google":
                model = genai.GenerativeModel(self.model)
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=self.settings.llm_temperature,
                        response_mime_type="application/json"
                    )
                )
                content = response.text
            
            # Parse JSON response
            data = json.loads(content)
            moments_data = data.get("viral_moments", [])
            
            # Convert to ViralMoment objects
            moments = []
            for m in moments_data:
                moment = ViralMoment(
                    start_time=m["start_time"],
                    end_time=m["end_time"],
                    duration=m["end_time"] - m["start_time"],
                    hook_text=m["hook_text"],
                    hook_type=m["hook_type"],
                    confidence_score=m["confidence_score"],
                    explanation=m.get("explanation"),
                    keywords=m.get("keywords", [])
                )
                moments.append(moment)
            
            # Sort by confidence score
            moments.sort(key=lambda x: x.confidence_score, reverse=True)
            
            logger.info(f"LLM identified {len(moments)} viral moments")
            return moments[:num_moments]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise AnalysisError(f"Invalid JSON from LLM: {str(e)}")
        except Exception as e:
            logger.error(f"LLM analysis error: {str(e)}")
            raise AnalysisError(f"Failed to analyze transcript: {str(e)}")


class AudioEnergyAnalyzer:
    """
    Audio energy analysis using Librosa.
    
    Detects:
    - Volume peaks (excitement, emphasis)
    - Laughter segments (comedy detection)
    - Music intensity (background energy)
    - Overall energy patterns
    
    Used to validate LLM-detected viral moments.
    """
    
    def __init__(self):
        self.sample_rate = 22050  # Librosa default
    
    async def analyze(
        self, 
        audio_path: Path,
        transcript_segments: Optional[List[TranscriptSegment]] = None
    ) -> AudioAnalysisResult:
        """
        Analyze audio for energy patterns.
        
        Args:
            audio_path: Path to audio file
            transcript_segments: Optional transcript for alignment
            
        Returns:
            AudioAnalysisResult with energy data
        """
        try:
            loop = asyncio.get_event_loop()
            
            def run_analysis():
                # Load audio
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
                
                # Calculate RMS energy over time
                hop_length = 512
                frame_length = 2048
                rms = librosa.feature.rms(
                    y=y, 
                    frame_length=frame_length,
                    hop_length=hop_length
                )[0]
                
                # Convert frames to timestamps
                times = librosa.frames_to_time(
                    np.arange(len(rms)), 
                    sr=sr,
                    hop_length=hop_length
                )
                
                # Normalize energy
                rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
                
                # Detect peaks (excitement moments)
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(
                    rms_normalized, 
                    height=0.6,
                    distance=sr // hop_length * 2  # Min 2 seconds between peaks
                )
                peak_times = times[peaks].tolist()
                
                # Detect laughter (high-frequency bursts)
                laughter_segments = self._detect_laughter(y, sr, hop_length)
                
                # Analyze music intensity (spectral features)
                music_intensity = self._analyze_music_intensity(y, sr, hop_length)
                
                # Overall energy per second
                energy_per_second = self._aggregate_energy_by_second(
                    times, rms_normalized
                )
                
                return AudioAnalysisResult(
                    peaks=peak_times,
                    laughter_segments=laughter_segments,
                    music_intensity=music_intensity,
                    overall_energy=energy_per_second
                )
            
            result = await loop.run_in_executor(None, run_analysis)
            
            logger.info(
                f"Audio analysis complete: {len(result.peaks)} peaks, "
                f"{len(result.laughter_segments)} laughter segments"
            )
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            raise AudioAnalysisError(f"Failed to analyze audio: {str(e)}")
    
    def _detect_laughter(
        self, 
        y: np.ndarray, 
        sr: int,
        hop_length: int
    ) -> List[Tuple[float, float]]:
        """
        Detect laughter segments based on audio characteristics.
        
        Laughter typically has:
        - High energy
        - Rapid amplitude modulation
        - Specific frequency patterns
        """
        laughter_segments = []
        
        # Calculate zero crossing rate (laughter has high ZCR)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        # Calculate spectral centroid (laughter has higher centroid)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        )[0]
        
        # Detect segments with laughter-like characteristics
        frame_time = hop_length / sr
        window_size = int(1.0 / frame_time)  # 1 second window
        
        i = 0
        while i < len(zcr) - window_size:
            window_zcr = zcr[i:i + window_size]
            window_centroid = spectral_centroid[i:i + window_size]
            
            # Laughter detection criteria
            if (np.mean(window_zcr) > 0.1 and 
                np.mean(window_centroid) > 2000):
                
                start_time = i * frame_time
                end_time = (i + window_size) * frame_time
                laughter_segments.append((start_time, end_time))
                i += window_size  # Skip ahead
            else:
                i += window_size // 2
        
        return laughter_segments
    
    def _analyze_music_intensity(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int
    ) -> List[Dict[str, Any]]:
        """Analyze music intensity over time."""
        # Spectral contrast (difference between peaks and valleys)
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=hop_length
        )
        
        # Average contrast across frequency bands
        avg_contrast = np.mean(contrast, axis=0)
        
        # Normalize
        contrast_normalized = (avg_contrast - avg_contrast.min()) / (
            avg_contrast.max() - avg_contrast.min() + 1e-8
        )
        
        times = librosa.frames_to_time(
            np.arange(len(contrast_normalized)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Sample every 5 seconds
        intensity_data = []
        for i in range(0, len(times), int(5.0 * sr / hop_length)):
            intensity_data.append({
                "time": float(times[i]),
                "intensity": float(contrast_normalized[i])
            })
        
        return intensity_data
    
    def _aggregate_energy_by_second(
        self,
        times: np.ndarray,
        energy: np.ndarray
    ) -> List[float]:
        """Aggregate energy values by second."""
        max_time = int(times[-1]) + 1
        energy_per_second = []
        
        for second in range(max_time):
            mask = (times >= second) & (times < second + 1)
            if np.any(mask):
                energy_per_second.append(float(np.mean(energy[mask])))
            else:
                energy_per_second.append(0.0)
        
        return energy_per_second
    
    def calculate_segment_energy_score(
        self,
        start_time: float,
        end_time: float,
        audio_result: AudioAnalysisResult
    ) -> float:
        """
        Calculate energy score for a specific time segment.
        
        Returns a score between 0-1 based on:
        - Presence of energy peaks
        - Laughter detection
        - Music intensity
        - Overall energy level
        """
        score = 0.0
        weights = {
            "peaks": 0.4,
            "laughter": 0.3,
            "energy": 0.3
        }
        
        # Check for peaks in segment
        peaks_in_segment = [
            p for p in audio_result.peaks 
            if start_time <= p <= end_time
        ]
        if peaks_in_segment:
            peak_score = min(len(peaks_in_segment) / 3, 1.0)
            score += weights["peaks"] * peak_score
        
        # Check for laughter in segment
        laughter_in_segment = [
            (s, e) for s, e in audio_result.laughter_segments
            if not (e < start_time or s > end_time)
        ]
        if laughter_in_segment:
            laughter_score = min(len(laughter_in_segment) / 2, 1.0)
            score += weights["laughter"] * laughter_score
        
        # Check average energy in segment
        start_idx = int(start_time)
        end_idx = min(int(end_time) + 1, len(audio_result.overall_energy))
        if start_idx < len(audio_result.overall_energy):
            segment_energy = audio_result.overall_energy[start_idx:end_idx]
            if segment_energy:
                avg_energy = np.mean(segment_energy)
                score += weights["energy"] * avg_energy
        
        return min(score, 1.0)


class ViralMomentDetector:
    """
    Main orchestrator for viral moment detection.
    
    Combines:
    1. Whisper transcription
    2. LLM analysis for content-based viral detection
    3. Audio energy analysis for validation
    4. Combined scoring and ranking
    """
    
    def __init__(self, config: Optional[ViralDetectionConfig] = None):
        self.config = config or ViralDetectionConfig()
        self.settings = get_settings()
        
        # Initialize components
        self.transcriber = WhisperTranscriber(self.config.whisper_model)
        self.analyzer = LLMViralAnalyzer(
            self.config.llm_provider,
            self.config.llm_model
        )
        self.energy_analyzer = AudioEnergyAnalyzer()
    
    async def detect_viral_moments(
        self,
        audio_path: Path,
        video_metadata: Optional[VideoMetadata] = None,
        language: Optional[str] = None,
        num_moments: Optional[int] = None
    ) -> Tuple[List[ViralMoment], List[TranscriptSegment], AudioAnalysisResult]:
        """
        Full pipeline to detect viral moments from audio.
        
        Args:
            audio_path: Path to audio file
            video_metadata: Optional video metadata
            language: Optional language hint
            num_moments: Number of moments to return (default from config)
            
        Returns:
            Tuple of (viral_moments, transcript_segments, audio_analysis)
        """
        num_moments = num_moments or self.config.max_moments
        
        logger.info(f"Starting viral moment detection for: {audio_path}")
        
        # Step 1: Transcribe audio
        logger.info("Step 1/3: Transcribing audio with Whisper...")
        transcript = await self.transcriber.transcribe(audio_path, language)
        
        # Step 2: LLM Analysis
        logger.info("Step 2/3: Analyzing content with LLM...")
        llm_moments = await self.analyzer.analyze(
            transcript,
            video_metadata,
            self.config.target_segment_duration,
            num_moments
        )
        
        # Step 3: Audio Energy Analysis
        logger.info("Step 3/3: Analyzing audio energy patterns...")
        audio_analysis = await self.energy_analyzer.analyze(audio_path, transcript)
        
        # Step 4: Combine scores
        logger.info("Combining LLM and audio scores...")
        final_moments = self._combine_scores(llm_moments, audio_analysis)
        
        logger.info(f"Detected {len(final_moments)} viral moments")
        
        return final_moments, transcript, audio_analysis
    
    def _combine_scores(
        self,
        llm_moments: List[ViralMoment],
        audio_analysis: AudioAnalysisResult
    ) -> List[ViralMoment]:
        """
        Combine LLM confidence with audio energy scores.
        
        Formula: final_score = (llm_score * llm_weight) + (energy_score * energy_weight)
        """
        combined_moments = []
        
        for moment in llm_moments:
            # Calculate energy score for this segment
            energy_score = self.energy_analyzer.calculate_segment_energy_score(
                moment.start_time,
                moment.end_time,
                audio_analysis
            )
            
            # Combine scores
            final_score = (
                moment.confidence_score * self.config.llm_weight +
                energy_score * self.config.energy_weight
            )
            
            # Update moment with combined score
            updated_moment = ViralMoment(
                start_time=moment.start_time,
                end_time=moment.end_time,
                duration=moment.duration,
                hook_text=moment.hook_text,
                hook_type=moment.hook_type,
                confidence_score=round(final_score, 3),
                energy_score=round(energy_score, 3),
                explanation=moment.explanation,
                keywords=moment.keywords
            )
            
            combined_moments.append(updated_moment)
        
        # Re-sort by combined score
        combined_moments.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return combined_moments
    
    async def detect_viral_moments_batch(
        self,
        audio_paths: List[Path],
        video_metadata_list: Optional[List[VideoMetadata]] = None,
        languages: Optional[List[str]] = None
    ) -> List[Tuple[List[ViralMoment], List[TranscriptSegment], AudioAnalysisResult]]:
        """
        Process multiple videos in batch.
        
        Args:
            audio_paths: List of audio file paths
            video_metadata_list: Optional list of metadata
            languages: Optional list of language codes
            
        Returns:
            List of results for each video
        """
        tasks = []
        for i, audio_path in enumerate(audio_paths):
            metadata = video_metadata_list[i] if video_metadata_list else None
            language = languages[i] if languages else None
            
            task = self.detect_viral_moments(audio_path, metadata, language)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)


# Singleton instance
_detector_instance: Optional[ViralMomentDetector] = None


def get_viral_detector(config: Optional[ViralDetectionConfig] = None) -> ViralMomentDetector:
    """Get or create singleton detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ViralMomentDetector(config)
    return _detector_instance


# Convenience functions
async def detect_viral_moments(
    audio_path: Path,
    video_metadata: Optional[VideoMetadata] = None,
    language: Optional[str] = None,
    num_moments: int = 10
) -> List[ViralMoment]:
    """Detect viral moments from audio file."""
    detector = get_viral_detector()
    moments, _, _ = await detector.detect_viral_moments(
        audio_path, video_metadata, language, num_moments
    )
    return moments


async def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None
) -> List[TranscriptSegment]:
    """Transcribe audio file."""
    detector = get_viral_detector()
    return await detector.transcriber.transcribe(audio_path, language)
