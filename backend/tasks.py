import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models import FilterJob, MLRun, Dataset
from database import SessionLocal
import json
import os

# Placeholder for ROOT file handling - will integrate uproot later
# For now, we'll use CSV/parquet as demonstration


class BackgroundTaskManager:
    """Manages background tasks for filtering and ML processing"""
    
    def __init__(self):
        self.running_tasks = {}
    
    async def process_filter_job(self, job_id: int):
        """Process a data filtering job in background"""
        db = SessionLocal()
        
        try:
            # Get job details
            job = db.query(FilterJob).filter(FilterJob.id == job_id).first()
            if not job:
                return
            
            # Update status
            job.status = "processing"
            job.started_at = datetime.utcnow()
            job.progress = 0
            db.commit()
            
            # Get source dataset
            dataset = db.query(Dataset).filter(Dataset.id == job.source_dataset_id).first()
            
            # Load data (placeholder - will use uproot for ROOT files)
            print(f"🔄 Loading dataset: {dataset.name}")
            # data = uproot.open(dataset.file_path)
            # For demo, using pandas
            data = pd.read_csv(dataset.file_path)  # Replace with ROOT handling
            
            job.progress = 20
            db.commit()
            
            # Apply filters
            print(f"🔄 Applying filters...")
            filtered_data = self._apply_filters(data, job.filter_config, job.selected_columns)
            
            job.progress = 60
            db.commit()
            
            # Save filtered data
            base_dir=os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir,"..","data","processed")
            output_dir=os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"filtered_{job.id}_{job.user_id}.csv"  # Will use ROOT format
            output_path = os.path.join(output_dir, output_filename)
            
            filtered_data.to_csv(output_path, index=False)
            
            job.progress = 90
            db.commit()
            
            # Update job with results
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.output_file_path = output_path
            job.filtered_events = len(filtered_data)
            job.output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            job.progress = 100
            
            db.commit()
            print(f"✅ Filter job {job_id} completed!")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
            print(f"❌ Filter job {job_id} failed: {e}")
        
        finally:
            db.close()
    
    def _apply_filters(self, data: pd.DataFrame, filter_config: dict, selected_columns: list) -> pd.DataFrame:
        """Apply threshold filters to data"""
        filtered = data.copy()
        
        # Apply threshold filters
        for col, config in filter_config.items():
            if config.get("selected", True) and col in filtered.columns:
                if "min" in config and config["min"] is not None:
                    filtered = filtered[filtered[col] >= config["min"]]
                if "max" in config and config["max"] is not None:
                    filtered = filtered[filtered[col] <= config["max"]]
        
        # Select only specified columns
        available_cols = [col for col in selected_columns if col in filtered.columns]
        filtered = filtered[available_cols]
        
        return filtered
    
    async def process_ml_run(self, run_id: int):
        """Process an ML analysis job in background"""
        db = SessionLocal()
        
        try:
            # Get run details
            run = db.query(MLRun).filter(MLRun.id == run_id).first()
            if not run:
                return
            
            # Update status
            run.status = "running"
            run.started_at = datetime.utcnow()
            run.progress = 0
            db.commit()
            
            # Get filtered dataset
            filter_job = db.query(FilterJob).filter(FilterJob.id == run.filter_job_id).first()
            
            if filter_job.status != "completed":
                raise Exception("Filtered dataset not ready")
            
            # Load filtered data
            data = pd.read_csv(filter_job.output_file_path)
            
            run.progress = 20
            db.commit()
            
            # Run appropriate ML model
            if run.model_type == "bsm":
                results = await self._run_bsm_analysis(data, run)
            elif run.model_type == "higgs":
                results = await self._run_higgs_analysis(data, run)
            else:  # Energy signature search
                results = await self._run_signature_search(data, run)
            
            run.progress = 80
            db.commit()
            
            # Save results
            run.results = results
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.progress = 100
            
            db.commit()
            print(f"✅ ML run {run_id} completed!")
            
        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.commit()
            print(f"❌ ML run {run_id} failed: {e}")
        
        finally:
            db.close()

    async def _run_higgs_analysis(self, data: pd.DataFrame, run: MLRun) -> dict:
        """Run Higgs boson signal analysis"""
        # Import here to avoid circular imports
        from ml_models import higgs_analyzer
    
        print(f"🔬 Starting Higgs analysis...")
        analyzer = higgs_analyzer.HiggsAnalyzer(decay_channel="ZZ*")
    # Run analysis
        results = analyzer.analyze(data)
    
        return results
    
    async def _run_bsm_analysis(self, data: pd.DataFrame, run: MLRun) -> dict:
        """Run Beyond Standard Model anomaly detection"""
        # Placeholder - will implement actual ML model
        await asyncio.sleep(2)  # Simulate processing
        
        return {
            "anomalies_found": 42,
            "anomaly_threshold": 0.95,
            "top_anomalies": [
                {"event_id": 1234, "score": 0.99},
                {"event_id": 5678, "score": 0.97}
            ]
        }
    
    async def _run_signature_search(self, data: pd.DataFrame, run: MLRun) -> dict:
        """Run energy signature search for specific particles"""
        # Placeholder - will implement actual search
        await asyncio.sleep(2)  # Simulate processing
        
        target = run.target_particle
        energy_range = run.energy_range_gev
        
        return {
            "target_particle": target,
            "energy_range_gev": energy_range,
            "matches_found": 89,
            "mean_energy": 91.2,
            "std_energy": 2.3
        }


# Global task manager instance
task_manager = BackgroundTaskManager()


async def start_filter_job(job_id: int):
    """Start a filter job in background"""
    asyncio.create_task(task_manager.process_filter_job(job_id))


async def start_ml_run(run_id: int):
    """Start an ML run in background"""
    asyncio.create_task(task_manager.process_ml_run(run_id))
