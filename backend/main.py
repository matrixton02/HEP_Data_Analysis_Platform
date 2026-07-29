from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import timedelta
import json
import os
import numpy as np
import pandas as pd
from database import get_db, init_db
from models import User, Dataset, FilterJob, MLRun
from auth import (
    hash_password, authenticate_user, create_access_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from tasks import start_filter_job, start_ml_run

# Initialize FastAPI app
app = FastAPI(title="Particle Physics Analysis Platform")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ============= Pydantic Schemas =============

class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str


class DatasetInfo(BaseModel):
    id: int
    name: str
    description: Optional[str]
    total_events: int
    file_size_mb: float
    columns: List[str]
    
    class Config:
        from_attributes = True


class DatasetStats(BaseModel):
    statistics: dict
    correlation_matrix: dict


class FilterJobCreate(BaseModel):
    source_dataset_id: int
    custom_name: str
    filter_config: dict
    selected_columns: List[str]


class FilterJobStatus(BaseModel):
    id: int
    custom_name: str
    source_dataset_name: str
    status: str
    progress: int
    filtered_events: Optional[int]
    output_size_mb: Optional[float]
    created_at: str
    completed_at: Optional[str]
    
    class Config:
        from_attributes = True


class MLRunCreate(BaseModel):
    filter_job_id: int
    model_type: str  # "bsm", "higgs", "Z_boson", "Higgs_boson"
    target_particle: Optional[str]=None
    energy_range_gev: Optional[dict]=None


class MLRunStatus(BaseModel):
    id: int
    model_type: str
    target_particle: Optional[str]
    status: str
    progress: int
    results: Optional[dict]
    created_at: str
    completed_at: Optional[str]
    
    class Config:
        from_attributes = True


# ============= API Endpoints =============

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("🚀 Server started successfully!")


@app.get("/")
async def root():
    """Serve login page"""
    return FileResponse("frontend/login.html")


@app.get("/home")
async def home():
    """Serve home page"""
    return FileResponse("frontend/home.html")


@app.get("/datasets")
async def datasets_page():
    """Serve datasets page"""
    return FileResponse("frontend/datasets.html")


@app.get("/filter")
async def filter_page():
    """Serve filter page"""
    return FileResponse("frontend/filter.html")


@app.get("/run")
async def run_page():
    """Serve ML run page"""
    return FileResponse("frontend/run.html")


# ============= Authentication Endpoints =============

@app.post("/api/register", response_model=Token)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register new user"""
    # Check if username exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password)
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    access_token = create_access_token(
        data={"sub": new_user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": new_user.id,
        "username": new_user.username
    }


@app.post("/api/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = authenticate_user(db, user_data.username, user_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


# ============= Dataset Endpoints =============

@app.get("/api/datasets", response_model=List[DatasetInfo])
async def get_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all available datasets"""
    datasets = db.query(Dataset).all()
    
    return [
        DatasetInfo(
            id=ds.id,
            name=ds.name,
            description=ds.description,
            total_events=ds.total_events,
            file_size_mb=ds.file_size_mb,
            columns=ds.columns
        )
        for ds in datasets
    ]


@app.get("/api/datasets/{dataset_id}/stats", response_model=DatasetStats)
async def get_dataset_stats(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dataset statistics and correlation matrix"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetStats(
        statistics=dataset.statistics,
        correlation_matrix=dataset.correlation_matrix
    )


# ============= Filter Job Endpoints =============

@app.post("/api/filter-jobs", status_code=201)
async def create_filter_job(
    job_data: FilterJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new filter job"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == job_data.source_dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create filter job
    new_job = FilterJob(
        user_id=current_user.id,
        source_dataset_id=job_data.source_dataset_id,
        custom_name=job_data.custom_name,
        filter_config=job_data.filter_config,
        selected_columns=job_data.selected_columns,
        status="pending"
    )
    
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    
    # Start background processing
    await start_filter_job(new_job.id)
    
    return {
        "job_id": new_job.id,
        "status": "pending",
        "message": "Filter job created and processing started"
    }


@app.get("/api/filter-jobs", response_model=List[FilterJobStatus])
async def get_user_filter_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all filter jobs for current user"""
    jobs = db.query(FilterJob).filter(FilterJob.user_id == current_user.id).all()
    
    result = []
    for job in jobs:
        dataset = db.query(Dataset).filter(Dataset.id == job.source_dataset_id).first()
        
        result.append(FilterJobStatus(
            id=job.id,
            custom_name=job.custom_name,
            source_dataset_name=dataset.name if dataset else "Unknown",
            status=job.status,
            progress=job.progress,
            filtered_events=job.filtered_events,
            output_size_mb=job.output_size_mb,
            created_at=job.created_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None
        ))
    
    return result


@app.get("/api/filter-jobs/{job_id}")
async def get_filter_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific filter job status"""
    job = db.query(FilterJob).filter(
        FilterJob.id == job_id,
        FilterJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Filter job not found")
    
    dataset = db.query(Dataset).filter(Dataset.id == job.source_dataset_id).first()
    
    return FilterJobStatus(
        id=job.id,
        custom_name=job.custom_name,
        source_dataset_name=dataset.name if dataset else "Unknown",
        status=job.status,
        progress=job.progress,
        filtered_events=job.filtered_events,
        output_size_mb=job.output_size_mb,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )


@app.get("/api/filter-jobs/{job_id}/download")
async def download_filtered_data(
    job_id: int,
    token: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download filtered dataset"""
    job = db.query(FilterJob).filter(
        FilterJob.id == job_id,
        FilterJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Filter job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Verify file exists
    if not os.path.exists(job.output_file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Output file not found: {job.output_file_path}"
        )
    
    # Verify file is readable
    if not os.access(job.output_file_path, os.R_OK):
        raise HTTPException(
            status_code=403, 
            detail=f"Cannot read output file: {job.output_file_path}"
        )
    
    print(f"📥 Downloading: {job.output_file_path}")
    
    return FileResponse(
        path=job.output_file_path,
        filename=f"{job.custom_name}.csv",
        media_type="text/csv"
    )
# ============= ML Run Endpoints =============

@app.post("/api/ml-runs", status_code=201)
async def create_ml_run(
    run_data: MLRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new ML analysis run"""
    # Verify filter job exists and belongs to user
    filter_job = db.query(FilterJob).filter(
        FilterJob.id == run_data.filter_job_id,
        FilterJob.user_id == current_user.id
    ).first()
    
    if not filter_job:
        raise HTTPException(status_code=404, detail="Filter job not found")
    
    if filter_job.status != "completed":
        raise HTTPException(status_code=400, detail="Filter job not completed yet")
    
    # Create ML run
    new_run = MLRun(
        user_id=current_user.id,
        filter_job_id=run_data.filter_job_id,
        model_type=run_data.model_type,
        target_particle=run_data.target_particle,
        energy_range_gev=run_data.energy_range_gev,
        status="pending"
    )
    
    db.add(new_run)
    db.commit()
    db.refresh(new_run)
    
    # Start background processing
    await start_ml_run(new_run.id)
    
    return {
        "run_id": new_run.id,
        "status": "pending",
        "message": "ML run created and processing started"
    }


@app.get("/api/ml-runs", response_model=List[MLRunStatus])
async def get_user_ml_runs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all ML runs for current user"""
    runs = db.query(MLRun).filter(MLRun.user_id == current_user.id).all()
    
    return [
        MLRunStatus(
            id=run.id,
            model_type=run.model_type,
            target_particle=run.target_particle,
            status=run.status,
            progress=run.progress,
            results=run.results,
            created_at=run.created_at.isoformat(),
            completed_at=run.completed_at.isoformat() if run.completed_at else None
        )
        for run in runs
    ]


@app.get("/api/ml-runs/{run_id}")
async def get_ml_run_status(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific ML run status and results"""
    run = db.query(MLRun).filter(
        MLRun.id == run_id,
        MLRun.user_id == current_user.id
    ).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="ML run not found")
    
    return MLRunStatus(
        id=run.id,
        model_type=run.model_type,
        target_particle=run.target_particle,
        status=run.status,
        progress=run.progress,
        results=run.results,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None
    )


# ============= Utility Endpoints =============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
