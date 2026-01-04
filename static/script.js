
document.addEventListener('DOMContentLoaded', () => {

    // Tab Switching Logic
    const tabs = document.querySelectorAll('.tab-btn');
    const sections = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Add active class to click tab
            tab.classList.add('active');

            // Hide all sections
            sections.forEach(s => s.style.display = 'none');

            // Show target section
            const targetId = tab.dataset.tab === 'analyze' ? 'analyzeSection' : 'digitizeSection';
            const targetSection = document.getElementById(targetId);
            targetSection.style.display = 'block';
            targetSection.classList.add('animate-up');
        });
    });

    // --- ANALYZE LOGIC ---
    setupUploadParams(
        'dropZoneAnalyze',
        'fileInputAnalyze',
        'fileInfoAnalyze',
        'analyzeBtn',
        'removeFileAnalyze',
        'errorAnalyze',
        'resultCardAnalyze',
        'analyzeUploadCard',
        (file) => file.name.endsWith('.csv'),
        '/predict',
        (data) => { // Success Callback
            document.getElementById('resultValue').textContent = data.prediction.class_name;
            document.getElementById('resultCardAnalyze').querySelector('.card-content').style.display = 'block';
        }
    );

    document.getElementById('resetBtnAnalyze').addEventListener('click', () => {
        resetUI('dropZoneAnalyze', 'fileInputAnalyze', 'fileInfoAnalyze', 'analyzeBtn', 'errorAnalyze', 'resultCardAnalyze', 'analyzeUploadCard');
    });

    // --- DIGITIZER LOGIC ---
    setupUploadParams(
        'dropZoneDigitize',
        'fileInputDigitize',
        'fileInfoDigitize',
        'digitizeBtn',
        'removeFileDigitize',
        'errorDigitize',
        null, // No result card switch for download
        'digitizeUploadCard',
        (file) => file.type.startsWith('image/'),
        '/convert',
        (blob, filename) => { // Success Callback for Blob (Download)
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename || 'digitized_ecg.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        },
        true // isDownload
    );


    // Reusable Upload Logic Factory
    function setupUploadParams(dropZoneId, fileInputId, fileInfoId, btnId, removeBtnId, errorId, resultCardId, uploadCardId, validateFn, endpoint, successCallback, isDownload = false) {
        const dropZone = document.getElementById(dropZoneId);
        const fileInput = document.getElementById(fileInputId);
        const btn = document.getElementById(btnId);
        const fileInfo = document.getElementById(fileInfoId);
        const removeBtn = document.getElementById(removeBtnId);
        const errorEl = document.getElementById(errorId);
        const resultCard = resultCardId ? document.getElementById(resultCardId) : null;
        const uploadCard = document.getElementById(uploadCardId);
        const spinner = btn.querySelector('.spinner');

        let currentFile = null;

        const showError = (msg) => errorEl.textContent = msg;
        const clearError = () => errorEl.textContent = '';

        const handleFile = (file) => {
            if (file && validateFn(file)) {
                currentFile = file;
                fileInfo.querySelector('.file-name').textContent = file.name;
                fileInfo.style.display = 'flex';
                btn.disabled = false;
                clearError();
            } else {
                showError('Invalid file type.');
            }
        };

        dropZone.addEventListener('click', (e) => {
            if (e.target !== removeBtn && !removeBtn.contains(e.target)) fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });

        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });

        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentFile = null;
            fileInput.value = '';
            fileInfo.style.display = 'none';
            btn.disabled = true;
            clearError();
        });

        btn.addEventListener('click', async () => {
            if (!currentFile) return;

            btn.disabled = true;
            const originalText = btn.querySelector('span').textContent;
            btn.querySelector('span').textContent = 'Processing...';
            spinner.style.display = 'inline-block';
            clearError();

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const response = await fetch(endpoint, { method: 'POST', body: formData });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || 'Operation failed');
                }

                if (isDownload) {
                    // For file download
                    const blob = await response.blob();
                    const contentDisposition = response.headers.get('Content-Disposition');
                    let filename = 'digitized.csv';
                    if (contentDisposition) {
                        const match = contentDisposition.match(/filename="?([^"]+)"?/);
                        if (match && match[1]) filename = match[1];
                    }
                    successCallback(blob, filename);
                    showError("Download started!"); // Reuse error msg for success notice broadly
                    errorEl.style.color = 'var(--success)';
                } else {
                    // For JSON response
                    const data = await response.json();
                    successCallback(data);
                    if (resultCard) {
                        uploadCard.style.display = 'none';
                        resultCard.style.display = 'block';
                        resultCard.classList.add('animate-up');
                    }
                }

            } catch (err) {
                errorEl.style.color = 'var(--danger)';
                showError(err.message);
            } finally {
                btn.disabled = false;
                btn.querySelector('span').textContent = originalText;
                spinner.style.display = 'none';
            }
        });
    }

    function resetUI(dropId, inputId, infoId, btnId, errorId, resultId, uploadId) {
        document.getElementById(inputId).value = '';
        document.getElementById(infoId).style.display = 'none';
        document.getElementById(btnId).disabled = true;
        document.getElementById(errorId).textContent = '';
        if (resultId) document.getElementById(resultId).style.display = 'none';
        document.getElementById(uploadId).style.display = 'block';
        document.getElementById(uploadId).classList.add('animate-up');
    }
});
