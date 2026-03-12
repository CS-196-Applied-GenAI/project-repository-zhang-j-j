"""Route tests for POST /analyze."""

import time


def test_analyze_returns_202_for_valid_job(client, app, registered_job_id):
    calls = {'count': 0}

    def fake_runner(job_id, upload_path, config, registry):
        calls['count'] += 1

    app.config['ANALYSIS_RUNNER'] = fake_runner

    response = client.post('/analyze', json={'job_id': registered_job_id})
    assert response.status_code == 202

    time.sleep(0.05)
    assert calls['count'] == 1


def test_analyze_returns_404_for_unknown_job(client):
    response = client.post('/analyze', json={'job_id': 'nonexistent-uuid'})
    assert response.status_code == 404


def test_analyze_rejects_invalid_problem_type(client, registered_job_id):
    response = client.post(
        '/analyze',
        json={'job_id': registered_job_id, 'problem_type': 'unsupervised'},
    )
    assert response.status_code == 400


def test_analyze_rejects_invalid_split_ratio(client, registered_job_id):
    response = client.post(
        '/analyze',
        json={'job_id': registered_job_id, 'train_test_split_ratio': 1.5},
    )
    assert response.status_code == 400


def test_analyze_error_propagated_to_status(client, registered_job_id, monkeypatch):
    class FailingReport:
        def __init__(self, data, **config):
            raise ValueError('bad data')

    monkeypatch.setattr('webapp.jobs.EDAReport', FailingReport)

    response = client.post('/analyze', json={'job_id': registered_job_id})
    assert response.status_code == 202

    for _ in range(20):
        status_resp = client.get(f'/status/{registered_job_id}')
        payload = status_resp.get_json()
        if payload['status'] == 'error':
            break
        time.sleep(0.05)

    status_resp = client.get(f'/status/{registered_job_id}')
    payload = status_resp.get_json()
    assert payload['status'] == 'error'
    assert 'bad data' in payload['error']
