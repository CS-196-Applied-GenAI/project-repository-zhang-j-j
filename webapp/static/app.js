const state = {
    jobId: null,
    pollTimer: null,
};

const uploadForm = document.getElementById('upload-form');
const analyzeForm = document.getElementById('analyze-form');
const uploadView = document.getElementById('upload-view');
const configView = document.getElementById('config-view');
const progressView = document.getElementById('progress-view');
const resultView = document.getElementById('result-view');
const errorMessage = document.getElementById('error-message');
const targetSelect = document.getElementById('target');
const statusText = document.getElementById('status-text');
const reportFrame = document.getElementById('report-frame');
const downloadLink = document.getElementById('download-link');
const startOverButton = document.getElementById('start-over');
const uploadButton = document.getElementById('upload-button');
const analyzeButton = document.getElementById('analyze-button');

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function clearError() {
    errorMessage.textContent = '';
    errorMessage.classList.add('hidden');
}

function showSection(sectionElement) {
    for (const section of [uploadView, configView, progressView, resultView]) {
        section.classList.toggle('hidden', section !== sectionElement);
    }
}

function populateTargetOptions(columns) {
    targetSelect.innerHTML = '';

    const noneOption = document.createElement('option');
    noneOption.value = '';
    noneOption.textContent = '(none - unsupervised)';
    targetSelect.appendChild(noneOption);

    for (const column of columns) {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        targetSelect.appendChild(option);
    }
}

async function handleUpload(event) {
    event.preventDefault();
    clearError();

    const fileInput = document.getElementById('dataset-file');
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('Select a file to upload.');
        return;
    }

    uploadButton.disabled = true;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Upload failed.');
            return;
        }

        state.jobId = data.job_id;
        populateTargetOptions(data.columns || []);
        showSection(configView);
    } catch (error) {
        showError('Network error while uploading file.');
    } finally {
        uploadButton.disabled = false;
    }
}

function buildAnalyzePayload() {
    const parseIntField = (id) => Number.parseInt(document.getElementById(id).value, 10);
    const parseFloatField = (id) => Number.parseFloat(document.getElementById(id).value);

    return {
        job_id: state.jobId,
        target: targetSelect.value || null,
        problem_type: document.getElementById('problem-type').value || null,
        random_seed: parseIntField('random-seed'),
        train_test_split_ratio: parseFloatField('split-ratio'),
        num_top_features: parseIntField('num-top-features'),
        missing_threshold: parseFloatField('missing-threshold'),
    };
}

function showResult(jobId) {
    reportFrame.src = `/report/${encodeURIComponent(jobId)}`;
    downloadLink.href = `/download/${encodeURIComponent(jobId)}`;
    showSection(resultView);
}

function stopPolling() {
    if (state.pollTimer !== null) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
    }
}

function pollStatus(jobId) {
    stopPolling();
    statusText.textContent = 'Running analysis...';

    state.pollTimer = setInterval(async () => {
        try {
            const response = await fetch(`/status/${encodeURIComponent(jobId)}`);
            const data = await response.json();

            if (!response.ok) {
                stopPolling();
                showError(data.error || 'Job not found.');
                showSection(configView);
                return;
            }

            if (data.status === 'pending') {
                statusText.textContent = 'Waiting for worker...';
                return;
            }

            if (data.status === 'running') {
                statusText.textContent = 'Running analysis...';
                return;
            }

            if (data.status === 'done') {
                stopPolling();
                showResult(jobId);
                return;
            }

            if (data.status === 'error') {
                stopPolling();
                showError(data.error || 'Analysis failed.');
                showSection(configView);
            }
        } catch (error) {
            stopPolling();
            showError('Status polling failed.');
            showSection(configView);
        }
    }, 2000);
}

async function handleAnalyze(event) {
    event.preventDefault();
    clearError();

    if (!state.jobId) {
        showError('Upload a dataset first.');
        return;
    }

    analyzeButton.disabled = true;

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(buildAnalyzePayload()),
        });
        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Analysis request failed.');
            return;
        }

        showSection(progressView);
        pollStatus(data.job_id);
    } catch (error) {
        showError('Network error while starting analysis.');
    } finally {
        analyzeButton.disabled = false;
    }
}

uploadForm.addEventListener('submit', handleUpload);
analyzeForm.addEventListener('submit', handleAnalyze);
startOverButton.addEventListener('click', () => {
    window.location.reload();
});
