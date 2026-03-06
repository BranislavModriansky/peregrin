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

    // Store files from a file input element into our map
    function captureFiles(fileInput) {
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) return;
        const inputId = fileInput.id || fileInput.closest('.shiny-input-container')?.querySelector('input[type="file"]')?.id;
        if (inputId) {
            const files = Array.from(fileInput.files).map(f => ({
                name: f.name,
                size: f.size,
                type: f.type
            }));
            fileInputData.set(inputId, files);
        }
    }

    // Store files directly from a FileList or DataTransfer (for drop events)
    function captureFilesFromList(fileList, inputId) {
        if (!fileList || fileList.length === 0 || !inputId) return;
        const files = Array.from(fileList).map(f => ({
            name: f.name,
            size: f.size,
            type: f.type
        }));
        fileInputData.set(inputId, files);
    }

    // Find the file input id from a container element
    function findFileInputId(container) {
        const fileInput = container.querySelector('input[type="file"]');
        return fileInput ? (fileInput.id || null) : null;
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

        // Capture files from browse (change event on file inputs)
        document.addEventListener('change', function(e) {
            const input = e.target;
            if (input.type === 'file') {
                captureFiles(input);
            }
        });

        // Capture files from drag-and-drop onto shiny file input containers
        document.addEventListener('drop', function(e) {
            // Find the closest shiny input container that accepts file drops
            const shinyContainer = e.target.closest('.shiny-input-container');
            if (!shinyContainer) return;

            const fileInput = shinyContainer.querySelector('input[type="file"]');
            if (!fileInput) return;

            const inputId = fileInput.id;
            if (!inputId) return;

            // The drop event's dataTransfer holds the dropped files
            const droppedFiles = e.dataTransfer && e.dataTransfer.files;
            if (droppedFiles && droppedFiles.length > 0) {
                captureFilesFromList(droppedFiles, inputId);
            } else {
                // Fallback: wait briefly for the file input to update, then read from it
                setTimeout(() => captureFiles(fileInput), 150);
            }
        }, true); // Use capture phase to catch before Shiny processes it

        // Additional fallback: observe the file input's files property via a MutationObserver
        // on the readonly text field that Shiny updates after upload completes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'value') {
                    const target = mutation.target;
                    if (target.matches && target.matches('input.form-control[readonly]')) {
                        const container = target.closest('.shiny-input-container');
                        if (container) {
                            const fileInput = container.querySelector('input[type="file"]');
                            if (fileInput && fileInput.files && fileInput.files.length > 0) {
                                captureFiles(fileInput);
                            }
                        }
                    }
                }
            });
        });

        observer.observe(document.body, {
            attributes: true,
            attributeFilter: ['value'],
            subtree: true
        });

        // Also watch for Shiny's custom progress/complete events on file inputs
        // Shiny triggers shiny:inputchanged after file upload completes
        $(document).on('shiny:inputchanged', function(e) {
            if (e.name && e.value && Array.isArray(e.value)) {
                // Check if this looks like a file input value (array of objects with name, size, etc.)
                const isFileInput = e.value.length > 0 && e.value[0] && e.value[0].name;
                if (isFileInput) {
                    const inputId = e.name;
                    const files = e.value.map(f => ({
                        name: f.name,
                        size: f.size || 0,
                        type: f.type || ''
                    }));
                    fileInputData.set(inputId, files);
                }
            }
        });

        // Click handler for popup
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