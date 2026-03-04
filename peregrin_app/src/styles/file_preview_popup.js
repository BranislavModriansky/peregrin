(function() {
    'use strict';

    // Store file data per input
    const fileInputData = new Map();

    // File icon SVG
    const fileIconSVG = `<svg viewBox="0 0 24 24"><path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/></svg>`;

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Create popup HTML
    function createPopup(files, inputLabel) {
        const overlay = document.createElement('div');
        overlay.className = 'file-preview-overlay';
        
        const popup = document.createElement('div');
        popup.className = 'file-preview-popup';
        
        let contentHTML = '';
        
        if (!files || files.length === 0) {
            contentHTML = '<div class="file-preview-empty">No files uploaded yet</div>';
        } else {
            contentHTML = `
                <div class="file-preview-count">${files.length} file${files.length !== 1 ? 's' : ''} uploaded</div>
                <ul class="file-preview-list">
                    ${files.map(file => `
                        <li class="file-preview-item">
                            <div class="file-preview-icon">${fileIconSVG}</div>
                            <div class="file-preview-info">
                                <div class="file-preview-name" title="${file.name}">${file.name}</div>
                                <div class="file-preview-size">${formatFileSize(file.size)}</div>
                            </div>
                        </li>
                    `).join('')}
                </ul>
            `;
        }

        // Do not remove these comments, saved for future
        // <h3>📁 ${inputLabel || 'Uploaded Files'}</h3>
        
        popup.innerHTML = `
            <div class="file-preview-header">
                <h3>${inputLabel || 'Uploaded Files'}</h3>
                <button class="file-preview-close" aria-label="Close">&times;</button>
            </div>
            <div class="file-preview-content">
                ${contentHTML}
            </div>
        `;
        
        document.body.appendChild(overlay);
        document.body.appendChild(popup);
        
        // Trigger animation
        requestAnimationFrame(() => {
            overlay.classList.add('show');
            popup.classList.add('show');
        });
        
        // Close handlers
        function closePopup() {
            overlay.classList.remove('show');
            popup.classList.remove('show');
            setTimeout(() => {
                overlay.remove();
                popup.remove();
            }, 250);
        }
        
        overlay.addEventListener('click', closePopup);
        popup.querySelector('.file-preview-close').addEventListener('click', closePopup);
        
        // Close on Escape key
        function handleEscape(e) {
            if (e.key === 'Escape') {
                closePopup();
                document.removeEventListener('keydown', handleEscape);
            }
        }
        document.addEventListener('keydown', handleEscape);
    }

    // Get label for input
    function getInputLabel(inputElement) {
        const container = inputElement.closest('[id^="input_file_container_"]');
        if (container) {
            const labelInput = container.querySelector('input[id^="condition_label"]');
            if (labelInput && labelInput.value) {
                return labelInput.value;
            }
            // Extract number from container ID
            const match = container.id.match(/input_file_container_(\d+)/);
            if (match) {
                return `Condition ${match[1]}`;
            }
        }
        return 'Uploaded Files';
    }

    // Initialize file preview functionality
    function initFilePreview() {
        document.addEventListener('change', function(e) {
            const input = e.target;
            if (input.type === 'file') {
                const inputId = input.id || input.closest('.shiny-input-container')?.querySelector('input[type="file"]')?.id;
                if (inputId && input.files) {
                    const files = Array.from(input.files).map(f => ({
                        name: f.name,
                        size: f.size,
                        type: f.type
                    }));
                    fileInputData.set(inputId, files);
                }
            }
        });

        document.addEventListener('click', function(e) {
            const formControl = e.target.closest('input.form-control[readonly]');
            if (formControl) {
                const container = formControl.closest('.shiny-input-container');
                if (container) {
                    const fileInput = container.querySelector('input[type="file"]');
                    if (fileInput) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        const inputId = fileInput.id;
                        const files = fileInputData.get(inputId) || [];
                        const label = getInputLabel(formControl);
                        
                        createPopup(files, label);
                    }
                }
            }
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initFilePreview);
    } else {
        initFilePreview();
    }
})();
