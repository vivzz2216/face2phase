// Upload page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFile = document.getElementById('removeFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadForm = document.getElementById('uploadForm');

    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            displayFileInfo(file);
        }
    });

    // Drag and drop handlers
    const fileInputLabel = document.querySelector('.file-input-label');
    
    fileInputLabel.addEventListener('dragover', function(e) {
        e.preventDefault();
        fileInputLabel.style.borderColor = '#667eea';
        fileInputLabel.style.backgroundColor = '#f0f2ff';
    });

    fileInputLabel.addEventListener('dragleave', function(e) {
        e.preventDefault();
        fileInputLabel.style.borderColor = '#ddd';
        fileInputLabel.style.backgroundColor = '#f8f9fa';
    });

    fileInputLabel.addEventListener('drop', function(e) {
        e.preventDefault();
        fileInputLabel.style.borderColor = '#ddd';
        fileInputLabel.style.backgroundColor = '#f8f9fa';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            displayFileInfo(files[0]);
        }
    });

    // Remove file handler
    removeFile.addEventListener('click', function() {
        fileInput.value = '';
        fileInfo.style.display = 'none';
        uploadBtn.disabled = true;
    });

    // Form submission handler
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file to upload');
            return;
        }

        uploadFile(file);
    });

    function displayFileInfo(file) {
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'flex';
        uploadBtn.disabled = false;
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        // Disable upload button and show loading state
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="btn-text">Uploading...</span><span class="btn-icon">⏳</span>';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.session_id) {
                // Redirect to processing page
                window.location.href = `/report/${data.session_id}`;
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        })
        .catch(error => {
            console.error('Upload error:', error);
            alert('Upload failed: ' + error.message);
            
            // Reset button state
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<span class="btn-text">Start Analysis</span><span class="btn-icon">🚀</span>';
        });
    }
});
