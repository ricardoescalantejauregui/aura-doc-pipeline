from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.services.worker import submit_job, get_job, list_jobs, purge_job

router = APIRouter()

class JobIn(BaseModel):
    data_dir: str = "data/samples"
    max_workers: Optional[int] = None
    per_file_timeout: Optional[float] = None  # en segundos (opcional)

class JobOut(BaseModel):
    id: str
    status: str
    total: int
    processed: int
    succeeded: int
    failed: int
    data_dir: str
    out_dir: str
    errors: List[Dict[str, Any]] = []
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

@router.post("/jobs", response_model=Dict[str, str])
def create_job(body: JobIn):
    job_id = submit_job(
        data_dir=body.data_dir,
        max_workers=body.max_workers,
        per_file_timeout=body.per_file_timeout,
    )
    return {"id": job_id}

@router.get("/jobs", response_model=List[JobOut])
def jobs_list():
    return list_jobs()

@router.get("/jobs/{job_id}", response_model=JobOut)
def jobs_get(job_id: str):
    try:
        return get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

@router.delete("/jobs/{job_id}", response_model=Dict[str, bool])
def jobs_purge(job_id: str):
    return {"ok": purge_job(job_id)}
