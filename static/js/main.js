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
            
            // Xóa class cũ
            resultText.classList.remove('text-mode');
            
            // Nếu là markdown hoặc có markdown syntax, render đẹp
            if (data.format === 'markdown' || text.includes('##') || text.includes('**') || text.includes('- ') || text.includes('```')) {
                // Render markdown với HTML
                resultText.innerHTML = formatMarkdown(text);
            } else {
                // Text thuần, dùng text mode
                resultText.textContent = text;
                resultText.classList.add('text-mode');
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
    // Enhanced markdown rendering - Giữ layout như DeepSeek-OCR
    let html = text;
    
    // Code blocks (phải xử lý trước để không bị ảnh hưởng)
    html = html.replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`([^`\n]+)`/gim, '<code>$1</code>');
    
    // Headers (từ lớn đến nhỏ)
    html = html.replace(/^###### (.*$)/gim, '<h6>$1</h6>');
    html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Horizontal rules
    html = html.replace(/^---$/gim, '<hr>');
    html = html.replace(/^\*\*\*$/gim, '<hr>');
    
    // Bold và italic (bold trước)
    html = html.replace(/\*\*\*(.*?)\*\*\*/gim, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^\)]+)\)/gim, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    
    // Images
    html = html.replace(/!\[([^\]]*)\]\(([^\)]+)\)/gim, '<img src="$2" alt="$1">');
    
    // Lists - xử lý từng dòng
    const lines = html.split('\n');
    let result = [];
    let inList = false;
    let listType = 'ul';
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        const trimmed = line.trim();
        
        // Ordered list
        if (/^\d+\.\s/.test(trimmed)) {
            if (!inList || listType !== 'ol') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ol>');
                inList = true;
                listType = 'ol';
            }
            result.push('<li>' + trimmed.replace(/^\d+\.\s/, '') + '</li>');
        }
        // Unordered list
        else if (/^[\-\*\+]\s/.test(trimmed)) {
            if (!inList || listType !== 'ul') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ul>');
                inList = true;
                listType = 'ul';
            }
            result.push('<li>' + trimmed.replace(/^[\-\*\+]\s/, '') + '</li>');
        }
        // Not a list item
        else {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
            }
            if (trimmed) {
                result.push(line);
            } else {
                result.push('');
            }
        }
    }
    
    if (inList) {
        result.push(`</${listType}>`);
    }
    
    html = result.join('\n');
    
    // Blockquotes
    html = html.replace(/^>\s(.*$)/gim, '<blockquote>$1</blockquote>');
    
    // Paragraphs - wrap text không phải block elements
    const finalLines = html.split('\n');
    let finalResult = [];
    let inParagraph = false;
    
    for (let i = 0; i < finalLines.length; i++) {
        const line = finalLines[i].trim();
        
        if (!line) {
            if (inParagraph) {
                finalResult.push('</p>');
                inParagraph = false;
            }
            continue;
        }
        
        // Block elements không wrap trong paragraph
        if (line.match(/^<(h[1-6]|ul|ol|li|pre|blockquote|hr|img|table)/) || 
            line.match(/^<\/?(ul|ol|li|pre|blockquote|hr|table)/)) {
            if (inParagraph) {
                finalResult.push('</p>');
                inParagraph = false;
            }
            finalResult.push(line);
        } else {
            if (!inParagraph) {
                finalResult.push('<p>');
                inParagraph = true;
            } else {
                finalResult.push('<br>');
            }
            finalResult.push(line);
        }
    }
    
    if (inParagraph) {
        finalResult.push('</p>');
    }
    
    return finalResult.join('\n');
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

