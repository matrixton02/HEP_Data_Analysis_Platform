"""
Utility script to add datasets to the database

Usage:
    python add_dataset.py <file_path> <dataset_name> <description>

Example:
    python add_dataset.py data/raw/higgs_data.csv "Higgs ZZ Decay" "CERN Open Data - Higgs to ZZ events"
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from models import Dataset


def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate column statistics"""
    stats = {}
    
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            stats[col] = {
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'mean': float(data[col].mean()),
                'median': float(data[col].median()),
                'std': float(data[col].std())
            }
    
    return stats


def calculate_correlation_matrix(data: pd.DataFrame) -> dict:
    """Calculate correlation matrix for numeric columns"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_data = data[numeric_cols]
    
    corr_matrix = numeric_data.corr()
    
    # Convert to nested dict
    result = {}
    for col in corr_matrix.columns:
        result[col] = corr_matrix[col].to_dict()
    
    return result


def add_dataset(file_path: str, name: str, description: str = ""):
    """Add a dataset to the database"""
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return
    
    print(f"📊 Loading dataset from {file_path}...")
    
    # Load data
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
    else:
        print("❌ Error: Unsupported file format. Use CSV or Parquet.")
        return
    
    print(f"✓ Loaded {len(data)} events with {len(data.columns)} columns")
    
    # Calculate statistics
    print("📈 Calculating statistics...")
    stats = calculate_statistics(data)
    
    print("🔗 Calculating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(data)
    
    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if dataset already exists
        existing = db.query(Dataset).filter(Dataset.name == name).first()
        if existing:
            print(f"⚠️  Dataset '{name}' already exists. Updating...")
            existing.description = description
            existing.file_path = os.path.abspath(file_path)
            existing.file_size_mb = file_size_mb
            existing.total_events = len(data)
            existing.statistics = stats
            existing.columns = list(data.columns)
            existing.correlation_matrix = corr_matrix
            
            db.commit()
            print(f"✅ Dataset '{name}' updated successfully!")
        else:
            # Create new dataset
            dataset = Dataset(
                name=name,
                description=description,
                file_path=os.path.abspath(file_path),
                file_size_mb=file_size_mb,
                total_events=len(data),
                statistics=stats,
                columns=list(data.columns),
                correlation_matrix=corr_matrix
            )
            
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            
            print(f"✅ Dataset '{name}' added successfully!")
            print(f"   ID: {dataset.id}")
            print(f"   Events: {dataset.total_events:,}")
            print(f"   Size: {dataset.file_size_mb:.2f} MB")
            print(f"   Columns: {len(dataset.columns)}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_dataset.py <file_path> <dataset_name> [description]")
        print("\nExample:")
        print('  python add_dataset.py data/raw/higgs.csv "Higgs ZZ Decay" "CERN Open Data"')
        sys.exit(1)
    
    file_path = sys.argv[1]
    name = sys.argv[2]
    description = sys.argv[3] if len(sys.argv) > 3 else ""
    print(file_path)
    add_dataset(file_path, name, description)
