"""Route tests for report retrieval and download endpoints."""


def test_report_returns_html(client, completed_job_id):
    response = client.get(f'/report/{completed_job_id}')
    assert response.status_code == 200
    assert b'<html' in response.data


def test_report_returns_404_for_pending_job(client, registered_job_id):
    response = client.get(f'/report/{registered_job_id}')
    assert response.status_code == 404


def test_download_returns_attachment_header(client, completed_job_id):
    response = client.get(f'/download/{completed_job_id}')
    assert response.status_code == 200
    assert 'attachment' in response.headers['Content-Disposition']
    assert 'report.html' in response.headers['Content-Disposition']


def test_download_returns_404_for_unknown_job(client):
    response = client.get('/download/does-not-exist')
    assert response.status_code == 404
