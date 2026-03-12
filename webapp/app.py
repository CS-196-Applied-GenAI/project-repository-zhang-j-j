"""Flask app factory and route definitions for the QuickEDA web wrapper."""

from __future__ import annotations

import atexit
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .jobs import JobRegistry, run_analysis

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.parquet'}
DEFAULT_CONFIG = {
    'target': None,
    'problem_type': None,
    'random_seed': 42,
    'train_test_split_ratio': 0.8,
    'num_top_features': 10,
    'missing_threshold': 0.5,
}


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_mapping(
        TESTING=False,
        MAX_UPLOAD_BYTES=MAX_UPLOAD_BYTES,
        JOB_REGISTRY=JobRegistry(),
        ANALYSIS_RUNNER=run_analysis,
    )

    if test_config:
        app.config.update(test_config)

    registry: JobRegistry = app.config['JOB_REGISTRY']

    if not app.config.get('TESTING', False):
        atexit.register(registry.cleanup_all)

    @app.get('/')
    def index() -> Any:
        return render_template('index.html')

    @app.post('/upload')
    def upload() -> Tuple[Any, int] | Any:
        if 'file' not in request.files:
            return _error('file is required', 400)

        uploaded_file = request.files['file']
        if not uploaded_file or not uploaded_file.filename:
            return _error('file is required', 400)

        ext = Path(uploaded_file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _error('only .csv and .parquet files are supported', 400)

        file_size = _file_size(uploaded_file)
        if file_size == 0:
            return _error('uploaded file is empty', 400)
        if file_size > app.config['MAX_UPLOAD_BYTES']:
            return _error('uploaded file exceeds 50 MB size limit', 400)

        job_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp(prefix=f'quickeda_{job_id}_')
        saved_name = secure_filename(uploaded_file.filename) or f'upload{ext}'
        upload_path = os.path.join(temp_dir, saved_name)
        uploaded_file.save(upload_path)

        try:
            columns = _extract_columns(upload_path, ext)
        except Exception as exc:
            # temp dir may not be tracked yet, so remove directly
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            return _error(f'unable to read uploaded file: {exc}', 400)

        registry.register(job_id=job_id, upload_path=upload_path, temp_dir=temp_dir)
        return jsonify({'job_id': job_id, 'columns': columns})

    @app.post('/analyze')
    def analyze() -> Tuple[Any, int] | Any:
        payload = request.get_json(silent=True) or {}
        job_id = payload.get('job_id')

        if not job_id:
            return _error('job_id is required', 400)

        job = registry.get(job_id)
        if job is None:
            return _error('job not found', 404)

        config, validation_error = _build_config(payload)
        if validation_error is not None:
            return _error(validation_error, 400)

        worker = threading.Thread(
            target=app.config['ANALYSIS_RUNNER'],
            args=(job_id, job['upload_path'], config, registry),
            daemon=True,
        )
        worker.start()

        return jsonify({'job_id': job_id}), 202

    @app.get('/status/<job_id>')
    def status(job_id: str) -> Tuple[Any, int] | Any:
        job = registry.get(job_id)
        if job is None:
            return _error('job not found', 404)

        return jsonify({'status': job['status'], 'error': job.get('error')})

    @app.get('/report/<job_id>')
    def report(job_id: str) -> Tuple[Any, int] | Any:
        report_path = _resolve_report_path(registry.get(job_id))
        if report_path is None:
            return _error('job not found or report not ready', 404)

        return send_file(report_path, mimetype='text/html')

    @app.get('/download/<job_id>')
    def download(job_id: str) -> Tuple[Any, int] | Any:
        report_path = _resolve_report_path(registry.get(job_id))
        if report_path is None:
            return _error('job not found or report not ready', 404)

        return send_file(
            report_path,
            mimetype='text/html',
            as_attachment=True,
            download_name='report.html',
        )

    return app


def _file_size(uploaded_file: FileStorage) -> int:
    """Return uploaded file size in bytes without consuming file contents."""
    current_pos = uploaded_file.stream.tell()
    uploaded_file.stream.seek(0, os.SEEK_END)
    size = uploaded_file.stream.tell()
    uploaded_file.stream.seek(current_pos)
    return size


def _extract_columns(file_path: str, extension: str) -> list[str]:
    """Read dataset column names with minimal I/O."""
    if extension == '.csv':
        return pd.read_csv(file_path, nrows=0).columns.tolist()

    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(file_path).names)
    except Exception:
        return pd.read_parquet(file_path).columns.tolist()


def _coerce_int(value: Any, field_name: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, f'{field_name} must be an integer'


def _coerce_float(value: Any, field_name: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, f'{field_name} must be a number'


def _build_config(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    config: Dict[str, Any] = {}

    target = payload.get('target', DEFAULT_CONFIG['target'])
    if target in ('', None):
        target = None
    elif not isinstance(target, str):
        return {}, 'target must be a string or null'
    config['target'] = target

    problem_type = payload.get('problem_type', DEFAULT_CONFIG['problem_type'])
    if problem_type in ('', None):
        problem_type = None
    if problem_type not in {None, 'classification', 'regression'}:
        return {}, 'problem_type must be classification, regression, or null'
    config['problem_type'] = problem_type

    random_seed = payload.get('random_seed', DEFAULT_CONFIG['random_seed'])
    random_seed_int, error = _coerce_int(random_seed, 'random_seed')
    if error is not None:
        return {}, error
    config['random_seed'] = random_seed_int

    split_ratio = payload.get(
        'train_test_split_ratio', DEFAULT_CONFIG['train_test_split_ratio']
    )
    split_ratio_float, error = _coerce_float(split_ratio, 'train_test_split_ratio')
    if error is not None:
        return {}, error
    if split_ratio_float is None or not (0 < split_ratio_float < 1):
        return {}, 'train_test_split_ratio must be between 0 and 1'
    config['train_test_split_ratio'] = split_ratio_float

    top_features = payload.get('num_top_features', DEFAULT_CONFIG['num_top_features'])
    top_features_int, error = _coerce_int(top_features, 'num_top_features')
    if error is not None:
        return {}, error
    if top_features_int is None or top_features_int < 1:
        return {}, 'num_top_features must be at least 1'
    config['num_top_features'] = top_features_int

    missing_threshold = payload.get(
        'missing_threshold', DEFAULT_CONFIG['missing_threshold']
    )
    missing_threshold_float, error = _coerce_float(
        missing_threshold, 'missing_threshold'
    )
    if error is not None:
        return {}, error
    if missing_threshold_float is None or not (0 <= missing_threshold_float <= 1):
        return {}, 'missing_threshold must be between 0 and 1'
    config['missing_threshold'] = missing_threshold_float

    return config, None


def _resolve_report_path(job: Optional[Dict[str, Any]]) -> Optional[str]:
    if job is None or job.get('status') != 'done':
        return None

    report_path = job.get('result_path')
    if not report_path or not os.path.exists(report_path):
        return None

    return report_path


def _error(message: str, status_code: int) -> Tuple[Any, int]:
    return jsonify({'error': message}), status_code


if __name__ == '__main__':
    create_app().run(debug=True)
