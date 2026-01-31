"""
FastAPI Backend for Viral Reel Generator

API Endpoints:
- POST /api/v1/videos/fetch-metadata - Fetch video metadata and languages
- POST /api/v1/videos/process - Start processing with selected language
- GET /api/v1/videos/status/{job_id} - Check processing status
- GET /api/v1/videos/reels/{job_id} - Get generated reels
- GET /api/v1/videos/download/{reel_id} - Download a specific reel

WebSocket:
- /ws/jobs/{job_id} - Real-time progress updates
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings, ensure_directories
from models import (
    VideoUrlRequest, LanguageSelectionRequest, MetadataResponse,
    ProcessingResponse, JobStatusResponse, ReelsListResponse, ReelPreview,
    ProcessingJob, ProcessingStatus, VideoMetadata
)
from downloader import fetch_video_metadata, download_audio_for_transcription, get_downloader
from viral_detector import detect_viral_moments, get_viral_detector
from video_processor import get_video_processor

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory job storage (replace with Redis in production)
jobs: dict[str, ProcessingJob] = {}

# Active WebSocket connections
active_connections: dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Viral Reel Generator API...")
    ensure_directories()
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Viral Reel Generator API",
    description="AI-Powered Multi-Lingual Viral Reel Generator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - Update this with your Vercel URL after deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Local React development
        "http://localhost:8000",     # Local API testing
        "https://localhost:3000",
        # Add your Vercel URL here after deployment, e.g.:
        # "https://viral-reel-generator.vercel.app",
        # "https://your-app.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (generated reels)
settings = get_settings()
app.mount("/reels", StaticFiles(directory=str(settings.output_path)), name="reels")


# ============================================================================
# Helper Functions
# ============================================================================

def create_job(url: str) -> ProcessingJob:
    """Create a new processing job."""
    job_id = str(uuid.uuid4())[:8]
    job = ProcessingJob(
        job_id=job_id,
        youtube_url=url,
        status=ProcessingStatus.PENDING
    )
    jobs[job_id] = job
    return job


async def update_job_status(
    job_id: str, 
    status: ProcessingStatus, 
    progress: Optional[int] = None,
    error_message: Optional[str] = None
):
    """Update job status and notify WebSocket clients."""
    if job_id not in jobs:
        return
    
    job = jobs[job_id]
    job.status = status
    job.updated_at = datetime.utcnow()
    
    if progress is not None:
        job.progress_percent = progress
    
    if error_message:
        job.error_message = error_message
    
    if status == ProcessingStatus.COMPLETED:
        job.completed_at = datetime.utcnow()
    
    # Notify WebSocket clients
    await notify_clients(job_id, {
        "job_id": job_id,
        "status": status.value,
        "progress": job.progress_percent,
        "message": error_message
    })


async def notify_clients(job_id: str, message: dict):
    """Send message to all connected WebSocket clients for a job."""
    if job_id not in active_connections:
        return
    
    disconnected = []
    for ws in active_connections[job_id]:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    # Remove disconnected clients
    for ws in disconnected:
        active_connections[job_id].remove(ws)


async def process_video_job(
    job_id: str,
    language_code: str,
    num_reels: int,
    reel_duration: int
):
    """
    Background task to process video.
    
    Pipeline:
    1. Download video/audio
    2. Transcribe with Whisper
    3. Detect viral moments with LLM
    4. Analyze audio energy
    5. Generate reels with face tracking and captions
    """
    settings = get_settings()
    job = jobs[job_id]
    
    try:
        # Step 1: Download
        await update_job_status(job_id, ProcessingStatus.DOWNLOADING, 10)
        
        downloader = get_downloader()
        
        # Download audio for transcription
        audio_path = await download_audio_for_transcription(
            job.youtube_url,
            settings.download_path
        )
        
        # Download video
        video_path, _ = await downloader.download_video(
            job.youtube_url,
            settings.download_path,
            language_code
        )
        
        await update_job_status(job_id, ProcessingStatus.TRANSCRIBING, 30)
        
        # Step 2: Transcribe
        detector = get_viral_detector()
        transcript = await detector.transcriber.transcribe(
            audio_path,
            language=language_code
        )
        job.transcript = transcript
        
        await update_job_status(job_id, ProcessingStatus.ANALYZING, 50)
        
        # Step 3 & 4: Detect viral moments (includes LLM + audio analysis)
        viral_moments, _, audio_analysis = await detector.detect_viral_moments(
            audio_path,
            job.metadata,
            language=language_code,
            num_moments=num_reels
        )
        job.viral_moments = viral_moments
        
        await update_job_status(job_id, ProcessingStatus.PROCESSING, 70)
        
        # Step 5: Generate reels
        processor = get_video_processor()
        
        reel_paths = await processor.generate_reels_batch(
            video_path,
            audio_path,
            viral_moments,
            transcript,
            max_reels=num_reels
        )
        
        # Build reel previews
        generated_reels = []
        for i, (moment, path) in enumerate(zip(viral_moments[:len(reel_paths)], reel_paths)):
            reel_preview = ReelPreview(
                reel_id=f"{job_id}_{i}",
                job_id=job_id,
                start_time=moment.start_time,
                end_time=moment.end_time,
                duration=moment.duration,
                hook_text=moment.hook_text,
                thumbnail_url=None,  # Could generate thumbnail
                video_url=f"/reels/{path.name}",
                confidence_score=moment.confidence_score,
                file_size=path.stat().st_size if path.exists() else None
            )
            generated_reels.append(reel_preview)
        
        job.generated_reels = [r.model_dump() for r in generated_reels]
        
        await update_job_status(job_id, ProcessingStatus.COMPLETED, 100)
        
        logger.info(f"Job {job_id} completed successfully. Generated {len(reel_paths)} reels.")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        await update_job_status(
            job_id, 
            ProcessingStatus.FAILED,
            error_message=str(e)
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Viral Reel Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "fetch_metadata": "POST /api/v1/videos/fetch-metadata",
            "start_processing": "POST /api/v1/videos/process",
            "check_status": "GET /api/v1/videos/status/{job_id}",
            "get_reels": "GET /api/v1/videos/reels/{job_id}",
            "download_reel": "GET /api/v1/videos/download/{reel_id}"
        }
    }


@app.post("/api/v1/videos/fetch-metadata", response_model=MetadataResponse)
async def fetch_metadata(request: VideoUrlRequest):
    """
    Fetch video metadata and available languages.
    
    Returns video info including all available audio tracks/languages.
    Creates a job ID for subsequent processing steps.
    """
    try:
        # Create job
        job = create_job(request.url)
        
        # Fetch metadata
        metadata = await fetch_video_metadata(request.url)
        job.metadata = metadata
        
        logger.info(f"Fetched metadata for job {job.job_id}: {metadata.title}")
        
        return MetadataResponse(
            job_id=job.job_id,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/videos/process", response_model=ProcessingResponse)
async def start_processing(
    request: LanguageSelectionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start processing video with selected language.
    
    This initiates the full pipeline:
    1. Download video with selected audio
    2. Transcribe with Whisper
    3. Detect viral moments with AI
    4. Generate reels
    
    Processing happens in the background. Use /status/{job_id} to check progress.
    """
    job_id = request.job_id
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Validate language
    available_langs = [l.language for l in job.metadata.available_languages]
    if request.language_code not in available_langs:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{request.language_code}' not available. Available: {available_langs}"
        )
    
    job.selected_language = request.language_code
    
    # Start background processing
    background_tasks.add_task(
        process_video_job,
        job_id,
        request.language_code,
        request.num_reels,
        request.reel_duration
    )
    
    logger.info(f"Started processing job {job_id} with language {request.language_code}")
    
    return ProcessingResponse(
        job_id=job_id,
        status=ProcessingStatus.DOWNLOADING,
        message="Processing started. Check status endpoint for progress."
    )


@app.get("/api/v1/videos/status/{job_id}", response_model=JobStatusResponse)
async def check_status(job_id: str):
    """
    Check processing status and progress.
    
    Returns current status, progress percentage, and results if completed.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = JobStatusResponse(
        job_id=job_id,
        status=job.status,
        progress_percent=job.progress_percent,
        message=job.error_message
    )
    
    # Include results if completed
    if job.status == ProcessingStatus.COMPLETED and job.generated_reels:
        response.results = {
            "reels_count": len(job.generated_reels),
            "reels": job.generated_reels
        }
    
    return response


@app.get("/api/v1/videos/reels/{job_id}", response_model=ReelsListResponse)
async def get_reels(job_id: str):
    """
    Get all generated reels for a job.
    
    Returns list of reel previews with download URLs.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    reels = []
    if job.generated_reels:
        reels = [ReelPreview(**r) for r in job.generated_reels]
    
    return ReelsListResponse(
        job_id=job_id,
        status=job.status,
        reels=reels,
        total_reels=len(reels)
    )


@app.get("/api/v1/videos/download/{reel_id}")
async def download_reel(reel_id: str):
    """
    Download a specific reel.
    
    reel_id format: {job_id}_{index}
    """
    try:
        parts = reel_id.split("_")
        job_id = parts[0]
        index = int(parts[1]) if len(parts) > 1 else 0
        
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        if not job.generated_reels or index >= len(job.generated_reels):
            raise HTTPException(status_code=404, detail="Reel not found")
        
        reel = job.generated_reels[index]
        video_path = Path(settings.output_path) / reel["video_url"].replace("/reels/", "")
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"viral_reel_{reel_id}.mp4"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all jobs (for debugging/admin)."""
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "progress": j.progress_percent,
                "url": j.youtube_url,
                "created_at": j.created_at.isoformat()
            }
            for j in jobs.values()
        ]
    }


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and cleanup files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Cleanup downloaded files
    downloader = get_downloader()
    if job.metadata:
        downloader.cleanup(job.metadata.video_id)
    
    # Remove job
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket for real-time job progress updates.
    
    Connect to receive live progress updates during processing.
    """
    await websocket.accept()
    
    if job_id not in active_connections:
        active_connections[job_id] = []
    
    active_connections[job_id].append(websocket)
    
    try:
        # Send current status
        if job_id in jobs:
            job = jobs[job_id]
            await websocket.send_json({
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress_percent,
                "message": job.error_message
            })
        
        # Keep connection alive and listen for client messages
        while True:
            message = await websocket.receive_text()
            # Handle client messages if needed
            
    except WebSocketDisconnect:
        if job_id in active_connections:
            active_connections[job_id].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if job_id in active_connections:
            active_connections[job_id].remove(websocket)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": len(jobs),
        "version": "1.0.0"
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
