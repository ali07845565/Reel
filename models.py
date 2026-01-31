"""
Pydantic models for request/response validation and data structures.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class ProcessingStatus(str, Enum):
    """Status enum for video processing jobs."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioTrack(BaseModel):
    """Represents an audio track from a YouTube video."""
    language: str = Field(..., description="Language code (e.g., 'en', 'es', 'fr')")
    language_name: str = Field(..., description="Human-readable language name")
    format_id: str = Field(..., description="yt-dlp format identifier")
    codec: Optional[str] = Field(None, description="Audio codec")
    bitrate: Optional[int] = Field(None, description="Audio bitrate in kbps")
    is_default: bool = Field(False, description="Whether this is the default audio track")
    is_original: bool = Field(False, description="Whether this is the original audio")


class VideoMetadata(BaseModel):
    """Metadata extracted from a YouTube video."""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    duration: int = Field(..., description="Video duration in seconds")
    duration_string: str = Field(..., description="Human-readable duration")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    uploader: Optional[str] = Field(None, description="Channel/uploader name")
    upload_date: Optional[str] = Field(None, description="Upload date (YYYYMMDD)")
    view_count: Optional[int] = Field(None, description="View count")
    available_languages: List[AudioTrack] = Field(default_factory=list)
    original_language: Optional[str] = Field(None, description="Original video language")


class TranscriptSegment(BaseModel):
    """A single segment of the transcript with timestamp."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, description="Transcription confidence")


class ViralMoment(BaseModel):
    """A detected viral moment with metadata."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    hook_text: str = Field(..., description="The hook/viral text content")
    hook_type: str = Field(..., description="Type: hook, climax, insight, emotional")
    confidence_score: float = Field(..., ge=0, le=1, description="AI confidence score")
    energy_score: Optional[float] = Field(None, description="Audio energy validation score")
    explanation: Optional[str] = Field(None, description="Why this moment was selected")
    keywords: List[str] = Field(default_factory=list, description="Key topics/keywords")


class AudioAnalysisResult(BaseModel):
    """Results from audio energy analysis."""
    peaks: List[float] = Field(default_factory=list, description="Timestamp of energy peaks")
    laughter_segments: List[tuple] = Field(default_factory=list, description="Detected laughter segments")
    music_intensity: List[Dict[str, Any]] = Field(default_factory=list, description="Music intensity over time")
    overall_energy: List[float] = Field(default_factory=list, description="Energy levels per second")


class ProcessingJob(BaseModel):
    """Represents a video processing job."""
    job_id: str = Field(..., description="Unique job identifier")
    youtube_url: str = Field(..., description="Source YouTube URL")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    selected_language: Optional[str] = Field(None, description="Selected audio language")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    progress_percent: int = Field(default=0, ge=0, le=100)
    
    # Results
    metadata: Optional[VideoMetadata] = Field(None)
    transcript: Optional[List[TranscriptSegment]] = Field(None)
    viral_moments: Optional[List[ViralMoment]] = Field(None)
    generated_reels: Optional[List[Dict[str, Any]]] = Field(None)


# Request/Response Models

class VideoUrlRequest(BaseModel):
    """Request to fetch video metadata."""
    url: str = Field(..., description="YouTube URL")


class LanguageSelectionRequest(BaseModel):
    """Request to select language and start processing."""
    job_id: str = Field(..., description="Job ID from metadata fetch")
    language_code: str = Field(..., description="Selected language code")
    num_reels: int = Field(default=5, ge=1, le=10, description="Number of reels to generate")
    reel_duration: int = Field(default=60, ge=30, le=90, description="Target reel duration in seconds")


class MetadataResponse(BaseModel):
    """Response with video metadata and available languages."""
    job_id: str
    metadata: VideoMetadata
    message: str = "Video metadata fetched successfully"


class ProcessingResponse(BaseModel):
    """Response after starting processing."""
    job_id: str
    status: ProcessingStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response with current job status."""
    job_id: str
    status: ProcessingStatus
    progress_percent: int
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class ReelPreview(BaseModel):
    """Preview of a generated reel."""
    reel_id: str
    job_id: str
    start_time: float
    end_time: float
    duration: float
    hook_text: str
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None
    confidence_score: float
    file_size: Optional[int] = None


class ReelsListResponse(BaseModel):
    """Response with list of generated reels."""
    job_id: str
    status: ProcessingStatus
    reels: List[ReelPreview]
    total_reels: int
