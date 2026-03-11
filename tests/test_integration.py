"""Integration test for full web wrapper flow."""

import io
import time


def test_full_flow_csv(client, monkeypatch):
    class FakeReport:
        def __init__(self, data, **config):
            self.target = config.get('target')

        def train_baseline_models(self):
            return None

        def generate_report(self, output_path):
            with open(output_path, 'w', encoding='utf-8') as file_handle:
                file_handle.write('<html><body>integration report</body></html>')

    monkeypatch.setattr('webapp.jobs.EDAReport', FakeReport)

    csv_data = b'age,income,target\n25,50000,1\n30,60000,0\n35,70000,1'

    upload_resp = client.post('/upload', data={'file': (io.BytesIO(csv_data), 'test.csv')})
    assert upload_resp.status_code == 200

    payload = upload_resp.get_json()
    job_id = payload['job_id']
    columns = payload['columns']
    assert 'target' in columns

    analyze_resp = client.post(
        '/analyze',
        json={
            'job_id': job_id,
            'target': 'target',
            'problem_type': 'classification',
        },
    )
    assert analyze_resp.status_code == 202

    done = False
    for _ in range(30):
        status_resp = client.get(f'/status/{job_id}')
        assert status_resp.status_code == 200
        status_payload = status_resp.get_json()
        if status_payload['status'] == 'done':
            done = True
            break
        if status_payload['status'] == 'error':
            raise AssertionError(f"Analysis failed: {status_payload['error']}")
        time.sleep(0.05)

    assert done, 'Job did not reach done status within timeout'

    report_resp = client.get(f'/report/{job_id}')
    assert report_resp.status_code == 200
    assert b'<html' in report_resp.data

    dl_resp = client.get(f'/download/{job_id}')
    assert dl_resp.status_code == 200
    assert 'attachment' in dl_resp.headers['Content-Disposition']
