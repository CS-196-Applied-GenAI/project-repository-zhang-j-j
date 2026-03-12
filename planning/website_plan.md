# Website Plan: QuickEDA Web App

## Summary of Decisions

| Concern | Decision |
|---|---|
| Backend | Flask |
| Frontend | Plain HTML/CSS/JS with Jinja2 templates |
| Job execution | Asynchronous background threads with polling |
| Report delivery | Inline iframe preview + download button |
| File storage | Temporary (in-memory/tempfile, cleaned on restart) |
| Config options | All `EDAReport` params exposed, each with a default value |
| Target column selection | Dynamic dropdown populated after file upload |

---

## Project Structure

```
webapp/
  app.py                  # Flask app factory and all route definitions
  jobs.py                 # In-memory job registry and background worker logic
  templates/
    index.html            # Upload form + analysis options
    result.html           # Inline report iframe + download button
  static/
    style.css
    app.js                # File upload, column fetch, polling logic
tests/
  test_upload.py          # File upload route unit tests
  test_jobs.py            # Job registry and worker unit tests
  test_routes_analyze.py  # Analyze submission route tests
  test_routes_status.py   # Status polling route tests
  test_routes_report.py   # Report retrieval and download route tests
  test_integration.py     # End-to-end flow integration tests
```

Existing `quickeda/` package is used as-is; the web layer wraps it without modification.

---

## Phase 1: Project Setup & TDD Foundation

**Goal:** Establish the Flask app skeleton, configure pytest, and write the first failing test.

### Tasks

1. Add `flask` to `requirements.txt`.
2. Create `webapp/app.py` with a Flask app factory function `create_app()` that returns a configured `Flask` instance. Register a placeholder `GET /` route returning `200 OK`.
3. Create `webapp/jobs.py` as an empty module with a `JobRegistry` class stub.
4. Create `tests/test_upload.py` with a pytest fixture that instantiates the app in test mode using `app.test_client()`.

### Tests to Write First (Failing)

```python
# tests/test_upload.py
def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200

def test_index_contains_upload_form(client):
    response = client.get("/")
    assert b'<form' in response.data
    assert b'enctype="multipart/form-data"' in response.data
```

### TDD Instruction
Write these tests, run them (they fail), then implement `create_app()` and `index.html` until they pass.

---

## Phase 2: File Upload Endpoint

**Goal:** `POST /upload` accepts a CSV or Parquet file, saves it to a `tempfile.mkdtemp()` directory, reads its column names, and returns them as JSON along with a `job_id` (UUID).

### Endpoint Contract

```
POST /upload
Content-Type: multipart/form-data
Body: file=<dataset file>

Response 200:
{
  "job_id": "<uuid>",
  "columns": ["col1", "col2", ...]
}

Response 400:
{
  "error": "<reason>"
}
```

### Validation Rules
- File must be present in the request.
- File extension must be `.csv` or `.parquet`.
- File must not be empty (0 bytes).
- File size must not exceed 50 MB.

### Tasks

1. In `webapp/jobs.py`, implement `JobRegistry` with:
   - `register(job_id, upload_path)` — stores `{status: "pending", upload_path, result_path: None, error: None}`.
   - `get(job_id)` — returns the job dict or `None`.
2. In `webapp/app.py`, implement `POST /upload`:
   - Save uploaded file to a temporary directory using `tempfile.mkdtemp()`.
   - Read column names with `pandas.read_csv()` or `pandas.read_parquet()` (read only the header, e.g., `nrows=0`).
   - Call `registry.register(job_id, upload_path)`.
   - Return `{"job_id": job_id, "columns": columns}`.

### Tests to Write First (Failing)

```python
# tests/test_upload.py
import io

def test_upload_csv_returns_job_id_and_columns(client):
    csv_data = b"age,income,target\n25,50000,1\n30,60000,0"
    response = client.post("/upload", data={"file": (io.BytesIO(csv_data), "test.csv")})
    assert response.status_code == 200
    json = response.get_json()
    assert "job_id" in json
    assert json["columns"] == ["age", "income", "target"]

def test_upload_rejects_missing_file(client):
    response = client.post("/upload", data={})
    assert response.status_code == 400

def test_upload_rejects_unsupported_extension(client):
    response = client.post("/upload", data={"file": (io.BytesIO(b"data"), "test.xlsx")})
    assert response.status_code == 400

def test_upload_rejects_empty_file(client):
    response = client.post("/upload", data={"file": (io.BytesIO(b""), "test.csv")})
    assert response.status_code == 400

def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr("webapp.app.MAX_UPLOAD_BYTES", 10)
    big_data = b"a,b\n" + b"1,2\n" * 100
    response = client.post("/upload", data={"file": (io.BytesIO(big_data), "test.csv")})
    assert response.status_code == 400
```

---

## Phase 3: Analysis Job Submission

**Goal:** `POST /analyze` accepts a `job_id` and all config options, validates them, and spawns a background thread that runs `EDAReport` and saves the HTML report.

### Endpoint Contract

```
POST /analyze
Content-Type: application/json
Body:
{
  "job_id": "<uuid>",
  "target": "<column name or null>",
  "problem_type": "<classification|regression|null>",
  "random_seed": 42,
  "train_test_split_ratio": 0.8,
  "num_top_features": 10,
  "missing_threshold": 0.5
}

Response 202:
{
  "job_id": "<uuid>"
}

Response 400:
{
  "error": "<reason>"
}

Response 404:
{
  "error": "job not found"
}
```

### Config Defaults (applied server-side if field is absent or null)

| Field | Default |
|---|---|
| `target` | `null` (no target) |
| `problem_type` | `null` (auto-infer) |
| `random_seed` | `42` |
| `train_test_split_ratio` | `0.8` |
| `num_top_features` | `10` |
| `missing_threshold` | `0.5` |

### Tasks

1. Add `JobRegistry.update_status(job_id, status, result_path=None, error=None)` to `jobs.py`.
2. Add `run_analysis(job_id, upload_path, config, registry)` function to `jobs.py`. This function:
   - Sets `status = "running"`.
   - Instantiates `EDAReport(data=upload_path, **config)`.
   - Calls `report.generate_report(output_path)` saving to a temp file.
   - Sets `status = "done"` and `result_path`.
   - On any exception, sets `status = "error"` and `error = str(e)`.
3. In `webapp/app.py`, implement `POST /analyze`:
   - Look up the job, return 404 if not found.
   - Validate `problem_type` is one of `null`, `"classification"`, `"regression"`.
   - Validate numeric params are within sensible bounds (e.g., split ratio in (0, 1)).
   - Spawn `threading.Thread(target=run_analysis, ...)` and start it.
   - Return `202`.

### Tests to Write First (Failing)

```python
# tests/test_routes_analyze.py
from unittest.mock import patch, MagicMock

def test_analyze_returns_202_for_valid_job(client, registered_job_id):
    with patch("webapp.jobs.run_analysis"):
        response = client.post("/analyze", json={"job_id": registered_job_id})
        assert response.status_code == 202

def test_analyze_returns_404_for_unknown_job(client):
    response = client.post("/analyze", json={"job_id": "nonexistent-uuid"})
    assert response.status_code == 404

def test_analyze_rejects_invalid_problem_type(client, registered_job_id):
    response = client.post("/analyze", json={
        "job_id": registered_job_id, "problem_type": "unsupervised"
    })
    assert response.status_code == 400

def test_analyze_rejects_invalid_split_ratio(client, registered_job_id):
    response = client.post("/analyze", json={
        "job_id": registered_job_id, "train_test_split_ratio": 1.5
    })
    assert response.status_code == 400

# tests/test_jobs.py
def test_run_analysis_sets_status_done(tmp_path, monkeypatch):
    # monkeypatch EDAReport to avoid running real analysis
    ...

def test_run_analysis_sets_status_error_on_exception(tmp_path, monkeypatch):
    ...
```

---

## Phase 4: Job Status Polling

**Goal:** `GET /status/<job_id>` returns the current job status as JSON. The frontend polls this every 2 seconds and updates a progress UI.

### Endpoint Contract

```
GET /status/<job_id>

Response 200:
{
  "status": "pending" | "running" | "done" | "error",
  "error": "<message or null>"
}

Response 404:
{
  "error": "job not found"
}
```

### Tasks

1. Implement `GET /status/<job_id>` in `app.py`.
2. In `static/app.js`, implement `pollStatus(jobId)`:
   - Uses `setInterval` to call `GET /status/<job_id>` every 2000ms.
   - Shows a spinner/progress message while `pending` or `running`.
   - On `done`, clears the interval and navigates to/loads the result view.
   - On `error`, clears the interval and displays the error message.

### Tests to Write First (Failing)

```python
# tests/test_routes_status.py
def test_status_pending(client, registered_job_id):
    response = client.get(f"/status/{registered_job_id}")
    assert response.status_code == 200
    assert response.get_json()["status"] == "pending"

def test_status_done(client, completed_job_id):
    response = client.get(f"/status/{completed_job_id}")
    assert response.get_json()["status"] == "done"

def test_status_error(client, failed_job_id):
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error"] is not None

def test_status_unknown_returns_404(client):
    response = client.get("/status/does-not-exist")
    assert response.status_code == 404
```

---

## Phase 5: Report Retrieval & Download

**Goal:** Two endpoints serve the completed report — one for inline rendering inside an `<iframe>`, one as a file download.

### Endpoint Contracts

```
GET /report/<job_id>
→ 200: serves the HTML report file (Content-Type: text/html)
→ 404: job not found or not yet complete

GET /download/<job_id>
→ 200: serves the HTML file as an attachment (Content-Disposition: attachment; filename="report.html")
→ 404: job not found or not yet complete
```

### Tasks

1. Implement `GET /report/<job_id>` using `flask.send_file()`.
2. Implement `GET /download/<job_id>` using `flask.send_file(..., as_attachment=True)`.
3. In `result.html`, embed the report using `<iframe src="/report/<job_id>" width="100%" height="800px">`.
4. Add a `<a href="/download/<job_id>" download>Download Report</a>` button.

### Tests to Write First (Failing)

```python
# tests/test_routes_report.py
def test_report_returns_html(client, completed_job_id):
    response = client.get(f"/report/{completed_job_id}")
    assert response.status_code == 200
    assert b"<html" in response.data

def test_report_returns_404_for_pending_job(client, registered_job_id):
    response = client.get(f"/report/{registered_job_id}")
    assert response.status_code == 404

def test_download_returns_attachment_header(client, completed_job_id):
    response = client.get(f"/download/{completed_job_id}")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert "report.html" in response.headers["Content-Disposition"]

def test_download_returns_404_for_unknown_job(client):
    response = client.get("/download/does-not-exist")
    assert response.status_code == 404
```

---

## Phase 6: Frontend UI

**Goal:** Build a clean single-page flow using plain HTML, CSS, and vanilla JS.

### Page Flow

```
[1. Upload Page]
  - File picker (CSV/Parquet only)
  - "Upload" button
  → on success: reveals config panel

[2. Config Panel] (revealed after upload)
  - Target column: <select> populated with columns from upload response
    - First option: "(none - unsupervised)"
  - Problem type: <select> [Auto-detect, Classification, Regression]
  - Random seed: <input type="number" value="42">
  - Train/test split ratio: <input type="number" step="0.05" min="0.1" max="0.95" value="0.8">
  - Number of top features: <input type="number" min="1" value="10">
  - Missing value threshold: <input type="number" step="0.05" min="0.0" max="1.0" value="0.5">
  - "Run Analysis" button

[3. Progress View] (replaces config panel)
  - Spinner + status message ("Uploading...", "Running analysis...", etc.)
  - Polls GET /status/<job_id> every 2s

[4. Result View] (replaces progress view)
  - <iframe> rendering the full HTML report
  - "Download Report" button
  - "Start Over" button (reloads page)
```

### Tasks

1. Create `webapp/templates/index.html` with all four sections (upload, config, progress, result), using CSS `display: none` to show/hide sections.
2. Create `webapp/static/app.js` implementing:
   - `handleUpload()` — `fetch POST /upload`, populate dropdown, reveal config.
   - `handleAnalyze()` — `fetch POST /analyze` with form values, start polling.
   - `pollStatus(jobId)` — interval polling, update status text, reveal result on done.
3. Create `webapp/static/style.css` with minimal clean styling.

### Tests to Write First (Failing)

```python
# tests/test_integration.py
def test_full_flow_csv(client):
    csv_data = b"age,income,target\n25,50000,1\n30,60000,0\n35,70000,1"

    # Step 1: Upload
    upload_resp = client.post("/upload",
        data={"file": (io.BytesIO(csv_data), "test.csv")})
    assert upload_resp.status_code == 200
    job_id = upload_resp.get_json()["job_id"]
    columns = upload_resp.get_json()["columns"]
    assert "target" in columns

    # Step 2: Submit analysis (mock EDAReport)
    with patch("webapp.jobs.EDAReport") as MockReport:
        instance = MockReport.return_value
        instance.generate_report.return_value = None
        # write a fake report file in the expected location
        ...
        analyze_resp = client.post("/analyze", json={
            "job_id": job_id,
            "target": "target",
            "problem_type": "classification"
        })
        assert analyze_resp.status_code == 202

    # Step 3: Poll until done (with timeout)
    ...

    # Step 4: Fetch report
    report_resp = client.get(f"/report/{job_id}")
    assert report_resp.status_code == 200

    # Step 5: Download
    dl_resp = client.get(f"/download/{job_id}")
    assert dl_resp.status_code == 200
```

---

## Phase 7: Error Handling & Edge Cases

**Goal:** Ensure errors at every stage are surfaced clearly to the user.

### Error Scenarios to Handle

| Scenario | Expected Behavior |
|---|---|
| Upload non-CSV/Parquet file | 400 with message, shown in UI |
| Upload file > 50 MB | 400 with message |
| `EDAReport` raises during analysis | Job status = "error", error message shown in UI |
| Polling a job that never existed | 404, UI shows "Job not found" |
| Accessing `/report/<id>` before job is done | 404 |
| Target column not in dataset (race condition) | Caught in `run_analysis`, sets error status |

### Tests to Write First (Failing)

```python
# tests/test_routes_analyze.py (additions)
def test_analyze_error_propagated_to_status(client, registered_job_id, monkeypatch):
    def failing_analysis(*args, **kwargs):
        raise ValueError("bad data")
    monkeypatch.setattr("webapp.jobs.run_analysis", failing_analysis)
    client.post("/analyze", json={"job_id": registered_job_id})
    # wait briefly for thread
    import time; time.sleep(0.1)
    status_resp = client.get(f"/status/{registered_job_id}")
    data = status_resp.get_json()
    assert data["status"] == "error"
    assert "bad data" in data["error"]
```

---

## Phase 8: Cleanup

**Goal:** Ensure temporary files are removed appropriately.

### Strategy
- Each job's upload file and report file are stored in a per-job `tempfile.mkdtemp()` directory.
- `JobRegistry` stores the temp directory path per job.
- Add `JobRegistry.cleanup(job_id)` that calls `shutil.rmtree(temp_dir, ignore_errors=True)` and removes the job from the registry.
- Call `cleanup()` after a download is served, or register an `atexit` handler to clean all temp dirs on shutdown.

### Tests to Write First (Failing)

```python
# tests/test_jobs.py (additions)
def test_cleanup_removes_temp_dir(tmp_path):
    registry = JobRegistry()
    job_id = str(uuid.uuid4())
    temp_dir = str(tmp_path / "job_dir")
    os.makedirs(temp_dir)
    registry.register(job_id, temp_dir + "/data.csv", temp_dir=temp_dir)
    registry.cleanup(job_id)
    assert not os.path.exists(temp_dir)
    assert registry.get(job_id) is None
```

---

## Test File Checklist

| File | Covers |
|---|---|
| `tests/test_upload.py` | `POST /upload` — valid, invalid format, empty, oversized, columns returned |
| `tests/test_jobs.py` | `JobRegistry` CRUD, `run_analysis` status transitions, cleanup |
| `tests/test_routes_analyze.py` | `POST /analyze` — valid submission, 404, validation errors, error propagation |
| `tests/test_routes_status.py` | `GET /status/<id>` — all status values, 404 |
| `tests/test_routes_report.py` | `GET /report/<id>`, `GET /download/<id>` — headers, 404 for incomplete jobs |
| `tests/test_integration.py` | Full upload → configure → analyze → poll → report → download flow |

---

## TDD Workflow (apply to every phase)

1. **Write the test** for the feature before writing any implementation code.
2. **Run the test** — confirm it fails for the right reason (not an import error).
3. **Write the minimum implementation** to make the test pass.
4. **Run all tests** — confirm no regressions.
5. **Refactor** if needed, keeping tests green.

Use `pytest -x` (stop on first failure) to work phase by phase.

Use `unittest.mock.patch` to mock `EDAReport` in all web-layer tests so tests run fast and don't depend on real data.

Use `io.BytesIO` wrapped in a tuple `(io.BytesIO(data), "filename.csv")` to simulate file uploads with Flask's test client.

---

## Implementation Order

```
Phase 1  →  Project setup + first test
Phase 2  →  POST /upload
Phase 3  →  POST /analyze + background worker
Phase 4  →  GET /status/<id> + JS polling
Phase 5  →  GET /report/<id> + GET /download/<id>
Phase 6  →  Frontend HTML/CSS/JS
Phase 7  →  Error handling
Phase 8  →  Temp file cleanup
```
