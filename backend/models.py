from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    filter_jobs = relationship("FilterJob", back_populates="user")
    ml_runs = relationship("MLRun", back_populates="user")


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    file_path = Column(String(500), nullable=False)
    file_size_mb = Column(Float)
    total_events = Column(Integer)
    
    # Statistical metadata (JSON format for flexibility)
    # Structure: {"column_name": {"min": x, "max": y, "mean": z, "median": m, "std": s}}
    statistics = Column(JSON)
    
    # Available columns in the dataset
    columns = Column(JSON)  # List of column names
    
    # Correlation matrix (stored as JSON)
    correlation_matrix = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    filter_jobs = relationship("FilterJob", back_populates="source_dataset")


class FilterJob(Base):
    __tablename__ = "filter_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    source_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    # User-friendly name for this filtered dataset
    custom_name = Column(String(100), nullable=False)
    
    # Filter configuration (JSON)
    # Structure: {"column_name": {"min": x, "max": y, "selected": true/false}}
    filter_config = Column(JSON, nullable=False)
    
    # Selected columns to keep
    selected_columns = Column(JSON, nullable=False)
    
    # Job status: "pending", "processing", "completed", "failed"
    status = Column(String(20), default="pending", nullable=False)
    
    # Progress percentage (0-100)
    progress = Column(Integer, default=0)
    
    # Output file path (when completed)
    output_file_path = Column(String(500))
    
    # Result statistics
    filtered_events = Column(Integer)
    output_size_mb = Column(Float)
    
    # Error message if failed
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="filter_jobs")
    source_dataset = relationship("Dataset", back_populates="filter_jobs")
    ml_runs = relationship("MLRun", back_populates="filtered_dataset")


class MLRun(Base):
    __tablename__ = "ml_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filter_job_id = Column(Integer, ForeignKey("filter_jobs.id"), nullable=False)
    
    # ML model type: "bsm", "higgs", or particle signature like "Z_boson", "Higgs_boson"
    model_type = Column(String(50), nullable=False)
    
    # For energy signature searches
    target_particle = Column(String(50))  # e.g., "Higgs", "Z_boson"
    energy_range_gev = Column(JSON)  # {"min": x, "max": y}
    
    # Status: "pending", "running", "completed", "failed"
    status = Column(String(20), default="pending", nullable=False)
    progress = Column(Integer, default=0)
    
    # Results (JSON format - model dependent)
    # For BSM: {"anomalies_found": N, "anomaly_scores": [...], "top_events": [...]}
    # For Higgs: {"candidate_events": N, "mass_peak": x, "confidence": y}
    # For signature: {"matches_found": N, "energy_distribution": [...]}
    results = Column(JSON)
    
    # Output visualization paths
    plot_paths = Column(JSON)  # List of generated plot file paths
    
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="ml_runs")
    filtered_dataset = relationship("FilterJob", back_populates="ml_runs")
