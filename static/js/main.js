// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const promptInput = document.getElementById('promptInput');
const processBtn = document.getElementById('processBtn');
const clearBtn = document.getElementById('clearBtn');
const resultSection = document.getElementById('resultSection');
const resultText = document.getElementById('resultText');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const exampleBtns = document.querySelectorAll('.example-btn');

let currentFile = null;

// Event Listeners
uploadBox.addEventListener('click', () => fileInput.click());
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('dragleave', handleDragLeave);
uploadBox.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', removeImage);
processBtn.addEventListener('click', processOCR);
clearBtn.addEventListener('click', clearAll);
copyBtn.addEventListener('click', copyResult);
downloadBtn.addEventListener('click', downloadResult);

// Example buttons
exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const prompt = btn.getAttribute('data-prompt');
        promptInput.value = prompt;
    });
});

// Drag and Drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp', 'application/pdf'];
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|gif|bmp|webp|pdf)$/i)) {
        showError('Định dạng file không được hỗ trợ!');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File quá lớn! Kích thước tối đa là 16MB.');
        return;
    }
    
    currentFile = file;
    
    // Show preview for images
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewSection.style.display = 'block';
            uploadBox.style.display = 'none';
            processBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    } else {
        // For PDF, just show filename
        previewSection.style.display = 'block';
        uploadBox.style.display = 'none';
        previewImage.src = '';
        previewImage.alt = file.name;
        processBtn.disabled = false;
    }
}

function removeImage() {
    currentFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadBox.style.display = 'block';
    processBtn.disabled = true;
    resultSection.style.display = 'none';
}

function clearAll() {
    removeImage();
    promptInput.value = '';
    resultSection.style.display = 'none';
}

async function processOCR() {
    if (!currentFile) {
        showError('Vui lòng chọn file ảnh!');
        return;
    }
    
    loadingOverlay.style.display = 'flex';
    resultSection.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('prompt', promptInput.value || '');
        formData.append('format', document.getElementById('formatSelect').value || 'markdown');
        
        const response = await fetch('/api/ocr', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Hiển thị kết quả với format phù hợp
            const text = data.text || 'Không có kết quả';
            
            // Nếu là markdown, render đẹp hơn
            if (data.format === 'markdown' && text.includes('##') || text.includes('**')) {
                // Render markdown cơ bản
                resultText.innerHTML = formatMarkdown(text);
            } else {
                resultText.textContent = text;
            }
            
            resultSection.style.display = 'block';
            // Scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            showError(data.error || 'Có lỗi xảy ra khi xử lý OCR');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Lỗi kết nối: ' + error.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

function formatMarkdown(text) {
    // Basic markdown rendering
    let html = text
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        .replace(/\n/g, '<br>');
    return html;
}

function copyResult() {
    // Copy text content (nếu là HTML thì lấy textContent)
    const text = resultText.textContent || resultText.innerText || '';
    
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Đã sao chép!';
            copyBtn.classList.add('success');
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fas fa-copy"></i> Sao chép';
                copyBtn.classList.remove('success');
            }, 2000);
        });
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Đã sao chép!';
        copyBtn.classList.add('success');
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Sao chép';
            copyBtn.classList.remove('success');
        }, 2000);
    }
}

function downloadResult() {
    const text = resultText.textContent;
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr_result_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showError(message) {
    // Simple alert for now, can be enhanced with a toast notification
    alert('Lỗi: ' + message);
}

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        if (!data.model_loaded) {
            console.warn('Model đang được tải, vui lòng đợi...');
        }
    } catch (error) {
        console.error('Không thể kết nối đến server:', error);
    }
});

