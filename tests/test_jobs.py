"""Unit tests for JobRegistry and analysis worker."""

import os
import tempfile
import uuid

from webapp.jobs import JobRegistry, run_analysis


class _FakeReport:
    def __init__(self, data, **config):
        self.data = data
        self.target = config.get('target')

    def train_baseline_models(self):
        return None

    def generate_report(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write('<html><body>report</body></html>')


class _FailingReport:
    def __init__(self, data, **config):
        self.target = config.get('target')

    def train_baseline_models(self):
        raise RuntimeError('bad data')

    def generate_report(self, output_path):
        raise RuntimeError('bad data')


def test_register_and_get_job():
    registry = JobRegistry()
    job_id = str(uuid.uuid4())
    registry.register(job_id, 'sample.csv', temp_dir='tmp_dir')

    job = registry.get(job_id)
    assert job is not None
    assert job['status'] == 'pending'
    assert job['upload_path'] == 'sample.csv'


def test_run_analysis_sets_status_done(monkeypatch):
    monkeypatch.setattr('webapp.jobs.EDAReport', _FakeReport)

    registry = JobRegistry()
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix='quickeda_job_')
    upload_path = os.path.join(temp_dir, 'data.csv')

    with open(upload_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('age,target\n25,1\n')

    registry.register(job_id, upload_path, temp_dir=temp_dir)

    run_analysis(job_id, upload_path, {'target': 'target'}, registry)

    job = registry.get(job_id)
    assert job['status'] == 'done'
    assert job['result_path'] is not None
    assert os.path.exists(job['result_path'])

    registry.cleanup(job_id)


def test_run_analysis_sets_status_error_on_exception(monkeypatch):
    monkeypatch.setattr('webapp.jobs.EDAReport', _FailingReport)

    registry = JobRegistry()
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix='quickeda_job_')
    upload_path = os.path.join(temp_dir, 'data.csv')

    with open(upload_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('age,target\n25,1\n')

    registry.register(job_id, upload_path, temp_dir=temp_dir)

    run_analysis(job_id, upload_path, {'target': 'target'}, registry)

    job = registry.get(job_id)
    assert job['status'] == 'error'
    assert 'bad data' in job['error']

    registry.cleanup(job_id)


def test_cleanup_removes_temp_dir(tmp_path):
    registry = JobRegistry()
    job_id = str(uuid.uuid4())
    temp_dir = str(tmp_path / 'job_dir')
    os.makedirs(temp_dir)
    upload_path = os.path.join(temp_dir, 'data.csv')

    with open(upload_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('a,b\n1,2\n')

    registry.register(job_id, upload_path, temp_dir=temp_dir)
    registry.cleanup(job_id)

    assert not os.path.exists(temp_dir)
    assert registry.get(job_id) is None
