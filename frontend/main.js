const API_BASE_URL = 'https://content-fingerprinting-xe7m.onrender.com';

// DOM Elements
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const resultsContainer = document.getElementById('results-container');
const resultsContent = document.getElementById('results-content');

// --- File Upload Logic ---
const setupDropzone = (dropzoneId, inputId, endpoint) => {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);

  // Drag and Drop Effects
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
  });

  // Handle file drop
  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFile(files[0], endpoint);
    }
  }, false);

  // Handle file select via click
  input.addEventListener('change', function() {
    if (this.files.length > 0) {
      handleFile(this.files[0], endpoint);
    }
  });
};

// Initialize both dropzones
setupDropzone('index-dropzone', 'index-input', '/upload-media');
setupDropzone('check-dropzone', 'check-input', '/check-media');

// --- API Integration ---
async function handleFile(file, endpoint) {
  // Show loader
  loader.classList.remove('hidden');
  resultsContainer.classList.add('hidden');
  
  if (endpoint === '/upload-media') {
    loaderText.textContent = 'Extracting Frames & Generating Neural Index...';
  } else {
    loaderText.textContent = 'Computing Signatures & Searching Database...';
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    displayResults(data, endpoint);
  } catch (error) {
    displayError(error);
  } finally {
    // Hide loader
    loader.classList.add('hidden');
    // Reset file inputs so the same file can be selected again
    document.getElementById('index-input').value = '';
    document.getElementById('check-input').value = '';
  }
}

function displayResults(data, endpoint) {
  resultsContainer.classList.remove('hidden');
  
  if (endpoint === '/upload-media') {
    resultsContent.innerHTML = `
      <div class="result-header success">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
        Indexed Successfully
      </div>
      <div class="result-details">
        <p><strong>Media ID:</strong> ${data.id}</p>
        <p><strong>Filename:</strong> ${data.filename}</p>
        <p><strong>Frames Analyzed:</strong> ${data.frames_processed}</p>
      </div>
    `;
  } else {
    let matchesHtml = '';
    if (data.matches && data.matches.length > 0) {
      const bestMatch = data.matches[0]; // Only show the single best match
      matchesHtml = `
        <div class="match-card ${bestMatch.label}">
          <p><strong>Match ID:</strong><br><span style="font-size: 0.9em; word-break: break-all;">${bestMatch.id}</span></p>
          <p><strong>Similarity Score:</strong> ${(bestMatch.score * 100).toFixed(2)}%</p>
          <p><strong>Status:</strong> ${bestMatch.label.toUpperCase().replace('_', ' ')}</p>
        </div>
      `;
    } else {
      matchesHtml = '<p style="color: var(--text-secondary); font-size: 1.1rem;">No matching signatures found in the database.</p>';
    }

    const isMatch = data.match_type === 'exact' || data.match_type === 'similar';
    
    resultsContent.innerHTML = `
      <div class="result-header ${isMatch ? 'success' : 'info'}">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        Search Complete — ${data.match_type.toUpperCase().replace('_', ' ')}
      </div>
      <div class="matches-container">
        ${matchesHtml}
      </div>
    `;
  }
  
  // Smooth scroll to results
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayError(error) {
  resultsContainer.classList.remove('hidden');
  resultsContent.innerHTML = `
    <div style="color: #ef4444; font-weight: bold; margin-bottom: 1rem;">
      ⚠ Error Processing Media
    </div>
    <pre style="color: #fca5a5;">${error.message}</pre>
  `;
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
