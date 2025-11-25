 // Form submission handling
        document.getElementById('predictionForm').addEventListener('submit', function (e) {
            e.preventDefault();

            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            loading.style.display = 'block';
            result.style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';

                    if (data.success) {
                        let resultHtml = `
                        <div class="result-card">
                            <h3><i class="fas fa-check-circle me-2"></i>Prediction Complete</h3>
                            <div class="prediction">${data.prediction}</div>
                            <p class="mb-4">Credit Risk Classification</p>
                    `;

                        if (data.probabilities) {
                            resultHtml += '<h5>Prediction Probabilities:</h5>';
                            for (const [class_name, prob] of Object.entries(data.probabilities)) {
                                const percentage = (prob * 100).toFixed(1);
                                resultHtml += `
                                <div class="probability-bar">
                                    <div class="d-flex justify-content-between mb-1">
                                        <span>${class_name}</span>
                                        <span>${percentage}%</span>
                                    </div>
                                    <div class="progress">
                                        <div class="progress-bar" style="width: ${percentage}%"></div>
                                    </div>
                                </div>
                            `;
                            }
                        }

                        resultHtml += '</div>';
                        result.innerHTML = resultHtml;
                    } else {
                        result.innerHTML = `
                        <div class="alert alert-danger">
                            <h5><i class="fas fa-exclamation-triangle me-2"></i>Error</h5>
                            <p>${data.error}</p>
                        </div>
                    `;
                    }

                    result.style.display = 'block';
                })
                .catch(error => {
                    loading.style.display = 'none';
                    result.innerHTML = `
                    <div class="alert alert-danger">
                        <h5><i class="fas fa-exclamation-triangle me-2"></i>Error</h5>
                        <p>An error occurred while processing your request.</p>
                    </div>
                `;
                    result.style.display = 'block';
                });
        });

        // File upload area styling
        const fileUploadArea = document.getElementById('fileUploadArea');
        const batchFile = document.getElementById('batchFile');

        fileUploadArea.addEventListener('click', () => batchFile.click());

        batchFile.addEventListener('change', function () {
            if (this.files.length > 0) {
                const file = this.files[0];
                const fileExtension = file.name.split('.').pop().toLowerCase();
                let iconClass = 'fas fa-file fa-3x text-success mb-3';

                // Set appropriate icon based on file type
                if (fileExtension === 'csv') {
                    iconClass = 'fas fa-file-csv fa-3x text-success mb-3';
                } else if (fileExtension === 'xlsx' || fileExtension === 'xls') {
                    iconClass = 'fas fa-file-excel fa-3x text-success mb-3';
                } else if (fileExtension === 'json') {
                    iconClass = 'fas fa-file-code fa-3x text-success mb-3';
                }

                fileUploadArea.innerHTML = `
                    <i class="${iconClass}"></i>
                    <h5>File Selected: ${file.name}</h5>
                    <p class="text-muted">Ready for batch processing</p>
                    <div class="mt-3">
                        <button type="submit" class="btn btn-success btn-lg">
                            <i class="fas fa-play me-2"></i>Process Batch Prediction
                        </button>
                    </div>
                `;
            }
        });

        // Drag and drop functionality
        fileUploadArea.addEventListener('dragover', function (e) {
            e.preventDefault();
            this.classList.add('dragover');
        });

        fileUploadArea.addEventListener('dragleave', function (e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });

        fileUploadArea.addEventListener('drop', function (e) {
            e.preventDefault();
            this.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const fileExtension = file.name.split('.').pop().toLowerCase();

                // Check if file type is supported
                if (['csv', 'xlsx', 'xls', 'json'].includes(fileExtension)) {
                    batchFile.files = files;
                    batchFile.dispatchEvent(new Event('change'));
                } else {
                    alert('Please select a supported file format (CSV, Excel, or JSON).');
                }
            }
        });

        // Batch form submission
        document.getElementById('batchForm').addEventListener('submit', function (e) {
            if (!batchFile.files.length) {
                e.preventDefault();
                alert('Please select a file first.');
                return;
            }
        });

        // Form validation
        function validateForm() {
            const requiredFields = document.querySelectorAll('input[required], select[required]');
            let isValid = true;

            requiredFields.forEach(field => {
                if (!field.value) {
                    field.classList.add('is-invalid');
                    isValid = false;
                } else {
                    field.classList.remove('is-invalid');
                }
            });

            return isValid;
        }

        // Real-time validation
        document.querySelectorAll('input, select').forEach(field => {
            field.addEventListener('blur', function () {
                if (this.hasAttribute('required') && !this.value) {
                    this.classList.add('is-invalid');
                } else {
                    this.classList.remove('is-invalid');
                }
            });
        });