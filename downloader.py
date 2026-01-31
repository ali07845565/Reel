"""
YouTube Downloader Module

This module handles:
- Extracting video metadata including available audio tracks
- Downloading videos with specific audio language tracks
- Managing multi-lingual audio stream selection
- Caching and temporary file management

Uses yt-dlp for robust YouTube downloading capabilities.
"""

import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import aiohttp
import yt_dlp
from yt_dlp.utils import DownloadError

from config import get_settings
from models import VideoMetadata, AudioTrack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDownloadError(Exception):
    """Custom exception for YouTube download errors."""
    pass


class LanguageNotAvailableError(Exception):
    """Raised when requested language is not available."""
    pass


@dataclass
class DownloadProgress:
    """Tracks download progress."""
    status: str
    percent: float
    speed: Optional[str] = None
    eta: Optional[str] = None
    downloaded_bytes: int = 0
    total_bytes: int = 0


class YouTubeDownloader:
    """
    Advanced YouTube downloader with multi-language audio support.
    
    Features:
    - Extract metadata with all available audio tracks
    - Download specific language audio streams
    - Progress tracking callbacks
    - Automatic retry with fallback
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.download_path = Path(self.settings.download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Base yt-dlp options
        self.ydl_opts_base = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
    
    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from various URL formats."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:shorts\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise YouTubeDownloadError(f"Could not extract video ID from URL: {url}")
    
    def _format_duration(self, seconds: int) -> str:
        """Format seconds into human-readable duration."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
    
    def _parse_audio_tracks(self, info: Dict[str, Any]) -> List[AudioTrack]:
        """
        Parse available audio tracks from yt-dlp info.
        
        YouTube can have multiple audio tracks for different languages.
        This extracts all available audio formats with their language info.
        """
        audio_tracks = []
        seen_languages = set()
        
        formats = info.get('formats', [])
        
        # Group formats by language
        language_formats: Dict[str, List[Dict]] = {}
        
        for fmt in formats:
            # Only consider audio formats
            if fmt.get('vcodec') != 'none':
                continue
            
            # Extract language info
            language = fmt.get('language') or fmt.get('lang') or 'unknown'
            
            # Skip if no audio
            if not fmt.get('acodec') or fmt.get('acodec') == 'none':
                continue
            
            if language not in language_formats:
                language_formats[language] = []
            
            language_formats[language].append(fmt)
        
        # Also check for automatic captions/audio tracks
        automatic_captions = info.get('automatic_captions', {})
        subtitles = info.get('subtitles', {})
        
        # Create AudioTrack objects for each language
        original_language = info.get('language') or 'en'
        
        for lang_code, formats in language_formats.items():
            # Get best quality format for this language
            best_format = max(formats, key=lambda x: x.get('abr') or 0)
            
            # Map language codes to names
            lang_name = self._get_language_name(lang_code)
            
            is_original = lang_code == original_language or lang_code == 'unknown'
            
            track = AudioTrack(
                language=lang_code,
                language_name=lang_name,
                format_id=best_format['format_id'],
                codec=best_format.get('acodec'),
                bitrate=best_format.get('abr'),
                is_default=is_original,
                is_original=is_original
            )
            
            audio_tracks.append(track)
            seen_languages.add(lang_code)
        
        # If no audio tracks found, add a default one
        if not audio_tracks:
            # Find best audio-only format
            audio_formats = [f for f in formats 
                           if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
            
            if audio_formats:
                best_audio = max(audio_formats, key=lambda x: x.get('abr') or 0)
                audio_tracks.append(AudioTrack(
                    language='unknown',
                    language_name='Original Audio',
                    format_id=best_audio['format_id'],
                    codec=best_audio.get('acodec'),
                    bitrate=best_audio.get('abr'),
                    is_default=True,
                    is_original=True
                ))
        
        # Sort: original first, then by language name
        audio_tracks.sort(key=lambda x: (not x.is_original, x.language_name))
        
        return audio_tracks
    
    def _get_language_name(self, code: str) -> str:
        """Convert language code to human-readable name."""
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'zh-Hans': 'Chinese (Simplified)',
            'zh-Hant': 'Chinese (Traditional)',
            'hi': 'Hindi',
            'ar': 'Arabic',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'el': 'Greek',
            'he': 'Hebrew',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Tagalog',
            'uk': 'Ukrainian',
            'ro': 'Romanian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'unknown': 'Original Audio',
        }
        
        return language_map.get(code, f"Language ({code})")
    
    async def get_metadata(self, url: str) -> VideoMetadata:
        """
        Fetch video metadata including available audio tracks.
        
        Args:
            url: YouTube video URL
            
        Returns:
            VideoMetadata with all available languages
            
        Raises:
            YouTubeDownloadError: If video cannot be accessed
        """
        try:
            loop = asyncio.get_event_loop()
            
            ydl_opts = {
                **self.ydl_opts_base,
                'listformats': False,
            }
            
            def extract():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=False)
            
            info = await loop.run_in_executor(None, extract)
            
            if not info:
                raise YouTubeDownloadError("Could not extract video information")
            
            # Parse audio tracks
            audio_tracks = self._parse_audio_tracks(info)
            
            # Build metadata
            metadata = VideoMetadata(
                video_id=info.get('id', self._extract_video_id(url)),
                title=info.get('title', 'Unknown Title'),
                description=info.get('description'),
                duration=info.get('duration', 0),
                duration_string=self._format_duration(info.get('duration', 0)),
                thumbnail_url=info.get('thumbnail'),
                uploader=info.get('uploader'),
                upload_date=info.get('upload_date'),
                view_count=info.get('view_count'),
                available_languages=audio_tracks,
                original_language=info.get('language')
            )
            
            logger.info(f"Fetched metadata for video: {metadata.video_id}")
            logger.info(f"Available languages: {[t.language for t in audio_tracks]}")
            
            return metadata
            
        except DownloadError as e:
            logger.error(f"yt-dlp error: {str(e)}")
            raise YouTubeDownloadError(f"Failed to fetch video: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise YouTubeDownloadError(f"Unexpected error: {str(e)}")
    
    async def download_video(
        self,
        url: str,
        output_path: Optional[Path] = None,
        language_code: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        video_quality: str = "720p"
    ) -> Tuple[Path, Path]:
        """
        Download video and audio (optionally specific language).
        
        Args:
            url: YouTube video URL
            output_path: Directory to save files (default: settings.download_path)
            language_code: Specific audio language to download (None for default)
            progress_callback: Optional callback for progress updates
            video_quality: Target video quality (e.g., "720p", "1080p", "best")
            
        Returns:
            Tuple of (video_file_path, audio_file_path)
            
        Raises:
            YouTubeDownloadError: If download fails
            LanguageNotAvailableError: If requested language not found
        """
        output_path = output_path or self.download_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_id = self._extract_video_id(url)
        
        # First get metadata to check available languages
        metadata = await self.get_metadata(url)
        
        # Validate language selection
        selected_track = None
        if language_code:
            selected_track = next(
                (t for t in metadata.available_languages if t.language == language_code),
                None
            )
            if not selected_track:
                available = [t.language for t in metadata.available_languages]
                raise LanguageNotAvailableError(
                    f"Language '{language_code}' not available. Available: {available}"
                )
        
        # Progress hook for yt-dlp
        def progress_hook(d):
            if progress_callback and d['status'] in ['downloading', 'finished']:
                progress = DownloadProgress(
                    status=d['status'],
                    percent=d.get('downloaded_bytes', 0) / d.get('total_bytes', 1) * 100,
                    speed=d.get('speed_string'),
                    eta=d.get('eta_string'),
                    downloaded_bytes=d.get('downloaded_bytes', 0),
                    total_bytes=d.get('total_bytes', 0)
                )
                # Use asyncio to call the callback safely
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._safe_callback(progress_callback, progress))
                except:
                    pass
        
        # Build format selector
        if selected_track:
            # Download specific video format + selected audio language
            format_spec = f"bestvideo[height<=?720][ext=mp4]+{selected_track.format_id}"
        else:
            # Download best quality with default audio
            format_spec = f"bestvideo[height<=?720][ext=mp4]+bestaudio[ext=m4a]"
        
        ydl_opts = {
            **self.ydl_opts_base,
            'format': format_spec,
            'outtmpl': str(output_path / f'{video_id}_%(format_id)s.%(ext)s'),
            'progress_hooks': [progress_hook],
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
        try:
            loop = asyncio.get_event_loop()
            
            def download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return info
            
            info = await loop.run_in_executor(None, download)
            
            # Find downloaded files
            video_file = None
            audio_file = None
            
            for fmt in info.get('formats', []):
                if fmt.get('vcodec') != 'none' and fmt.get('acodec') == 'none':
                    # Video-only format
                    expected_path = output_path / f"{video_id}_{fmt['format_id']}.mp4"
                    if expected_path.exists():
                        video_file = expected_path
                
                elif fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                    # Audio-only format
                    expected_path = output_path / f"{video_id}_{fmt['format_id']}.m4a"
                    if expected_path.exists():
                        audio_file = expected_path
            
            # If merged file exists, use that
            merged_path = output_path / f"{video_id}.mp4"
            if merged_path.exists():
                video_file = merged_path
                audio_file = merged_path
            
            if not video_file:
                raise YouTubeDownloadError("Video file not found after download")
            
            logger.info(f"Downloaded video: {video_file}")
            if audio_file and audio_file != video_file:
                logger.info(f"Downloaded audio: {audio_file}")
            
            return video_file, audio_file or video_file
            
        except DownloadError as e:
            logger.error(f"Download error: {str(e)}")
            raise YouTubeDownloadError(f"Download failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            raise YouTubeDownloadError(f"Download error: {str(e)}")
    
    async def _safe_callback(self, callback: Callable, *args):
        """Safely invoke callback, catching any errors."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.warning(f"Callback error: {e}")
    
    async def download_audio_only(
        self,
        url: str,
        output_path: Optional[Path] = None,
        language_code: Optional[str] = None,
        format: str = "mp3"
    ) -> Path:
        """
        Download audio only (for transcription).
        
        Args:
            url: YouTube video URL
            output_path: Directory to save file
            language_code: Specific audio language
            format: Output audio format (mp3, wav, m4a)
            
        Returns:
            Path to downloaded audio file
        """
        output_path = output_path or self.download_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_id = self._extract_video_id(url)
        
        ydl_opts = {
            **self.ydl_opts_base,
            'format': 'bestaudio/best',
            'outtmpl': str(output_path / f'{video_id}_audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': format,
                'preferredquality': '192',
            }],
        }
        
        try:
            loop = asyncio.get_event_loop()
            
            def download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=True)
            
            await loop.run_in_executor(None, download)
            
            audio_file = output_path / f"{video_id}_audio.{format}"
            if audio_file.exists():
                return audio_file
            
            # Try alternative extensions
            for ext in ['mp3', 'm4a', 'wav', 'webm']:
                alt_file = output_path / f"{video_id}_audio.{ext}"
                if alt_file.exists():
                    return alt_file
            
            raise YouTubeDownloadError("Audio file not found after download")
            
        except Exception as e:
            raise YouTubeDownloadError(f"Audio download failed: {str(e)}")
    
    def cleanup(self, video_id: Optional[str] = None):
        """
        Clean up downloaded files.
        
        Args:
            video_id: If provided, only clean files for this video
        """
        if video_id:
            patterns = [
                f"{video_id}_*.mp4",
                f"{video_id}_*.m4a",
                f"{video_id}_*.mp3",
                f"{video_id}_*.wav",
                f"{video_id}_*.webm",
            ]
            for pattern in patterns:
                for file in self.download_path.glob(pattern):
                    try:
                        file.unlink()
                        logger.info(f"Deleted: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file}: {e}")
        else:
            # Clean all files older than 24 hours (optional safety)
            pass


# Singleton instance
_downloader_instance: Optional[YouTubeDownloader] = None


def get_downloader() -> YouTubeDownloader:
    """Get or create singleton downloader instance."""
    global _downloader_instance
    if _downloader_instance is None:
        _downloader_instance = YouTubeDownloader()
    return _downloader_instance


# Convenience functions for direct use
async def fetch_video_metadata(url: str) -> VideoMetadata:
    """Fetch metadata for a YouTube video."""
    downloader = get_downloader()
    return await downloader.get_metadata(url)


async def download_video_with_language(
    url: str,
    language_code: Optional[str] = None,
    output_path: Optional[Path] = None
) -> Tuple[Path, Path]:
    """Download video with specific language audio."""
    downloader = get_downloader()
    return await downloader.download_video(url, output_path, language_code)


async def download_audio_for_transcription(
    url: str,
    output_path: Optional[Path] = None
) -> Path:
    """Download audio optimized for transcription."""
    downloader = get_downloader()
    return await downloader.download_audio_only(url, output_path, format="wav")
