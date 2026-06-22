// --- LOGIKA INTERAKSI ANTARMUKA PENGGUNA PREMIUM --- //

document.addEventListener('DOMContentLoaded', () => {
    // Referensi Elemen DOM
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const processBtn = document.getElementById('processBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const processBtnText = processBtn.querySelector('.btn-text');
    
    // Elemen Log Status
    const processingLog = document.getElementById('processingLog');
    const stepUpload = document.getElementById('stepUpload');
    const stepOCR = document.getElementById('stepOCR');
    const stepAI = document.getElementById('stepAI');
    
    // Elemen Presentasi
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsContent = document.getElementById('resultsContent');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Nilai Metrik
    const valStore = document.getElementById('valStore');
    const valDate = document.getElementById('valDate');
    const valTotal = document.getElementById('valTotal');
    const valAddress = document.getElementById('valAddress');
    const addressSection = document.getElementById('addressSection');
    const valItemsCount = document.getElementById('valItemsCount');
    const itemsTableBody = document.getElementById('itemsTableBody');
    
    // Geometri & JSON
    const canvasImage = document.getElementById('canvasImage');
    const boxesContainer = document.getElementById('boxesContainer');
    const valProcessingTime = document.getElementById('valProcessingTime');
    const jsonOutput = document.getElementById('jsonOutput');
    const copyJsonBtn = document.getElementById('copyJsonBtn');

    let currentFile = null;
    let lastExtractionResult = null;

    // --- PENGENDALI DRAG & DROP --- //
    browseBtn.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.add('dragover'));
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length) handleFileSelected(files[0]);
    });

    fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFileSelected(e.target.files[0]);
    });

    removeBtn.addEventListener('click', e => {
        e.stopPropagation();
        resetWorkspace();
    });

    function handleFileSelected(file) {
        if (!file.type.startsWith('image/')) {
            alert('Hanya mendukung berkas gambar.');
            return;
        }
        currentFile = file;
        const objectUrl = URL.createObjectURL(file);
        imagePreview.src = objectUrl;
        canvasImage.src = objectUrl;
        
        previewContainer.hidden = false;
        processBtn.disabled = false;
        resultsEmpty.hidden = false;
        resultsContent.hidden = true;
    }

    function resetWorkspace() {
        currentFile = null;
        fileInput.value = '';
        previewContainer.hidden = true;
        processBtn.disabled = true;
        resultsEmpty.hidden = false;
        resultsContent.hidden = true;
        processingLog.hidden = true;
    }

    // --- PENGENDALI TABS SWITCHER --- //
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.hidden = true);
            
            btn.classList.add('active');
            const targetId = `pane-${btn.dataset.tab}`;
            document.getElementById(targetId).hidden = false;
            
            // Render ulang penanda geometri saat tab aktif demi akurasi dimensi
            if (btn.dataset.tab === 'geometry' && lastExtractionResult) {
                setTimeout(() => renderBoundingBoxes(lastExtractionResult), 50);
            }
        });
    });

    // --- SALIN JSON RESPONS --- //
    copyJsonBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(jsonOutput.textContent);
        copyJsonBtn.textContent = 'Tersalin!';
        copyJsonBtn.style.background = 'var(--c-total)';
        setTimeout(() => {
            copyJsonBtn.textContent = 'Salin JSON';
            copyJsonBtn.style.background = '';
        }, 2000);
    });

    // --- EKSEKUSI PIPELINE ASINKRON --- //
    processBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Atur Status UI
        processBtn.disabled = true;
        btnSpinner.hidden = false;
        processBtnText.textContent = 'Mengekstrak...';
        
        // Mulai Rantai Mikro-Animasi Log
        processingLog.hidden = false;
        [stepUpload, stepOCR, stepAI].forEach(st => {
            st.classList.remove('active', 'done');
        });
        
        stepUpload.classList.add('active');

        const formData = new FormData();
        formData.append('file', currentFile);

        // Simulasi perpindahan log berdasarkan estimasi waktu lokal/GPU
        setTimeout(() => {
            stepUpload.classList.remove('active');
            stepUpload.classList.add('done');
            stepOCR.classList.add('active');
        }, 800);

        setTimeout(() => {
            stepOCR.classList.remove('active');
            stepOCR.classList.add('done');
            stepAI.classList.add('active');
        }, 1800);

        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || 'Kegagalan komunikasi peladen');
            }

            const result = await response.json();
            lastExtractionResult = result;

            // Log Tuntas
            stepAI.classList.remove('active');
            stepAI.classList.add('done');

            // Render Presentasi Data
            populateSummaryPane(result);
            jsonOutput.textContent = JSON.stringify(result, null, 2);
            valProcessingTime.textContent = result.processing_time_sec || '1.50';
            
            // Sembunyikan state kosong, munculkan hasil
            resultsEmpty.hidden = true;
            resultsContent.hidden = false;
            
            // Render geometri jika gambar sudah dimuat
            if (canvasImage.complete) {
                renderBoundingBoxes(result);
            } else {
                canvasImage.onload = () => renderBoundingBoxes(result);
            }

        } catch (err) {
            alert(`Kesalahan Ekstraksi: ${err.message}`);
            resultsEmpty.hidden = false;
            resultsContent.hidden = true;
        } finally {
            processBtn.disabled = false;
            btnSpinner.hidden = true;
            processBtnText.textContent = 'Ekstrak Entitas';
        }
    });

    // --- RENDER POPULASI METRIK --- //
    function populateSummaryPane(result) {
        valStore.textContent = result.store || 'Tidak Terdeteksi';
        valDate.textContent = result.date || '-';
        
        // Format Total Uang
        const totalNum = parseFloat(result.total) || 0;
        valTotal.textContent = totalNum.toLocaleString('id-ID');
        
        if (result.address) {
            valAddress.textContent = result.address;
            addressSection.hidden = false;
        } else {
            addressSection.hidden = true;
        }

        // Render Tabel Item
        const items = result.items || [];
        valItemsCount.textContent = `${items.length} Item`;
        itemsTableBody.innerHTML = '';

        if (!items.length) {
            itemsTableBody.innerHTML = `<tr><td colspan="3" class="text-center" style="color:var(--text-muted)">Tidak ada baris harga/item terklasifikasi mandiri.</td></tr>`;
            return;
        }

        items.forEach(item => {
            const tr = document.createElement('tr');
            const priceStr = (parseFloat(item.price) || 0).toLocaleString('id-ID');
            tr.innerHTML = `
                <td style="font-weight:500">${escapeHtml(item.name || item.raw)}</td>
                <td class="text-center">${item.qty || 1}</td>
                <td class="text-right" style="color:var(--text-accent); font-weight:600">Rp ${priceStr}</td>
            `;
            itemsTableBody.appendChild(tr);
        });
    }

    // --- RENDER VISUAL OVERLAY GEOMETRI --- //
    function renderBoundingBoxes(result) {
        boxesContainer.innerHTML = '';
        const lines = result.raw_lines || [];
        const metadata = result.metadata || {};
        
        if (!metadata.image_size || !lines.length) return;
        
        // Dapatkan resolusi absolut asli citra
        const [origW, origH] = metadata.image_size.split('x').map(Number);
        if (!origW || !origH) return;

        lines.forEach(line => {
            // Hindari merender kelas dekoratif murni kecuali memiliki tingkat keyakinan yang memadai
            if (line.class === 'OTHER' && line.confidence < 0.5) return;
            
            // Gunakan x_min, y_min, width, height dari struktur
            const wRatio = (line.width / origW) * 100;
            const hRatio = (line.height / origH) * 100;
            const leftRatio = (line.x_min / origW) * 100;
            const topRatio = (line.y_min / origH) * 100;

            const box = document.createElement('div');
            box.className = `box-marker b-${getBoxClassSuffix(line.class)}`;
            box.style.width = `${wRatio}%`;
            box.style.height = `${hRatio}%`;
            box.style.left = `${leftRatio}%`;
            box.style.top = `${topRatio}%`;
            
            // Tooltip informasi semantik
            const confPct = Math.round(line.confidence * 100);
            box.title = `[${line.class}] "${line.text}" (${confPct}%)`;
            
            boxesContainer.appendChild(box);
        });
    }

    function getBoxClassSuffix(cls) {
        switch(cls) {
            case 'STORE': return 'store';
            case 'ADDRESS_CONTACT': return 'addr';
            case 'DATE': return 'date';
            case 'ITEM_DESC': 
            case 'ITEM_PRICE/QTY': return 'item';
            case 'TOTAL_PAYMENT': return 'total';
            default: return 'other';
        }
    }

    function escapeHtml(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
        );
    }
});
