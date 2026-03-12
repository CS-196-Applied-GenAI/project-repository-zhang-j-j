"""Shared fixtures for web wrapper tests."""

import os
import tempfile
import uuid

import pytest

from webapp.app import create_app
from webapp.jobs import JobRegistry


@pytest.fixture
def registry():
    return JobRegistry()


@pytest.fixture
def app(registry):
    return create_app({'TESTING': True, 'JOB_REGISTRY': registry})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def registered_job_id(registry):
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix='quickeda_test_')
    upload_path = os.path.join(temp_dir, 'data.csv')

    with open(upload_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('feature,target\n1,0\n2,1\n3,0\n')

    registry.register(job_id, upload_path, temp_dir=temp_dir)
    yield job_id
    registry.cleanup(job_id)


@pytest.fixture
def completed_job_id(registry):
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix='quickeda_test_')
    upload_path = os.path.join(temp_dir, 'data.csv')
    report_path = os.path.join(temp_dir, 'report.html')

    with open(upload_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('feature,target\n1,0\n')

    with open(report_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('<html><body>ok</body></html>')

    registry.register(job_id, upload_path, temp_dir=temp_dir)
    registry.update_status(job_id, 'done', result_path=report_path)

    yield job_id
    registry.cleanup(job_id)


@pytest.fixture
def failed_job_id(registry):
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix='quickeda_test_')
    upload_path = os.path.join(temp_dir, 'data.csv')

    with open(upload_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('feature,target\n1,0\n')

    registry.register(job_id, upload_path, temp_dir=temp_dir)
    registry.update_status(job_id, 'error', error='bad data')

    yield job_id
    registry.cleanup(job_id)
