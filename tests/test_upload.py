"""Route tests for upload and index page."""

import io


def test_index_returns_200(client):
    response = client.get('/')
    assert response.status_code == 200


def test_index_contains_upload_form(client):
    response = client.get('/')
    assert b'<form' in response.data
    assert b'enctype="multipart/form-data"' in response.data


def test_upload_csv_returns_job_id_and_columns(client):
    csv_data = b'age,income,target\n25,50000,1\n30,60000,0'
    response = client.post('/upload', data={'file': (io.BytesIO(csv_data), 'test.csv')})

    assert response.status_code == 200
    payload = response.get_json()
    assert 'job_id' in payload
    assert payload['columns'] == ['age', 'income', 'target']


def test_upload_rejects_missing_file(client):
    response = client.post('/upload', data={})
    assert response.status_code == 400


def test_upload_rejects_unsupported_extension(client):
    response = client.post('/upload', data={'file': (io.BytesIO(b'data'), 'test.xlsx')})
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post('/upload', data={'file': (io.BytesIO(b''), 'test.csv')})
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client, app):
    app.config['MAX_UPLOAD_BYTES'] = 10
    big_data = b'a,b\n' + b'1,2\n' * 100

    response = client.post('/upload', data={'file': (io.BytesIO(big_data), 'test.csv')})
    assert response.status_code == 400
