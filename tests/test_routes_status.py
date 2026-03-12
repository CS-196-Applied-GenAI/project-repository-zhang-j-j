"""Route tests for GET /status/<job_id>."""


def test_status_pending(client, registered_job_id):
    response = client.get(f'/status/{registered_job_id}')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'pending'


def test_status_done(client, completed_job_id):
    response = client.get(f'/status/{completed_job_id}')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'done'


def test_status_error(client, failed_job_id):
    response = client.get(f'/status/{failed_job_id}')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'error'
    assert payload['error'] is not None


def test_status_unknown_returns_404(client):
    response = client.get('/status/does-not-exist')
    assert response.status_code == 404
