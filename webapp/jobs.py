"""In-memory job registry and worker logic for the QuickEDA web app."""

from __future__ import annotations

import os
import shutil
import threading
from typing import Any, Dict, Optional

from quickeda import EDAReport


class JobRegistry:
    """Thread-safe in-memory store for analysis jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(self, job_id: str, upload_path: str, temp_dir: Optional[str] = None) -> None:
        """Register a new job with pending status."""
        resolved_temp_dir = temp_dir or os.path.dirname(upload_path)
        with self._lock:
            self._jobs[job_id] = {
                'status': 'pending',
                'upload_path': upload_path,
                'result_path': None,
                'error': None,
                'temp_dir': resolved_temp_dir,
            }

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return a shallow copy of the job metadata if it exists."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def update_status(
        self,
        job_id: str,
        status: str,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update status and optional result/error fields for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False

            job['status'] = status
            if result_path is not None:
                job['result_path'] = result_path

            if error is not None:
                job['error'] = error
            elif status in {'pending', 'running', 'done'}:
                job['error'] = None

            return True

    def cleanup(self, job_id: str) -> bool:
        """Delete a job and remove its temporary directory."""
        with self._lock:
            job = self._jobs.pop(job_id, None)

        if job is None:
            return False

        temp_dir = job.get('temp_dir')
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    def cleanup_all(self) -> None:
        """Clean all known job directories."""
        with self._lock:
            job_ids = list(self._jobs.keys())

        for job_id in job_ids:
            self.cleanup(job_id)


def run_analysis(
    job_id: str,
    upload_path: str,
    config: Dict[str, Any],
    registry: JobRegistry,
) -> None:
    """Run EDA analysis in a background worker and update job status."""
    job = registry.get(job_id)
    if job is None:
        return

    result_path = os.path.join(job['temp_dir'], 'report.html')

    try:
        registry.update_status(job_id, 'running')
        report = EDAReport(data=upload_path, **config)

        if report.target is not None:
            report.train_baseline_models()

        report.generate_report(result_path)
        registry.update_status(job_id, 'done', result_path=result_path)
    except Exception as exc:  # pragma: no cover - defensive exception boundary
        registry.update_status(job_id, 'error', error=str(exc))
