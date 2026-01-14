// Global variables
let resumeId = null;
let jobs = [];

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const resumeFile = document.getElementById('resumeFile');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfo = document.getElementById('fileInfo');
const jobDescription = document.getElementById('jobDescription');
const jobTitle = document.getElementById('jobTitle');
const matchBtn = document.getElementById('matchBtn');
const saveJobBtn = document.getElementById('saveJobBtn');
const resultsSection = document.getElementById('resultsSection');
const matchingSkills = document.getElementById('matchingSkills');
const missingSkills = document.getElementById('missingSkills');
const recommendationsList = document.getElementById('recommendationsList');
const jobsList = document.getElementById('jobsList');
const loadingOverlay = document.getElementById('loadingOverlay');
const toast = document.getElementById('toast');

// Event Listeners
uploadArea.addEventListener('click', () => {
    if (!uploadArea.classList.contains('uploaded-success')) {
        resumeFile.click();
    }
});
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
resumeFile.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', handleUpload);
matchBtn.addEventListener('click', handleMatch);
saveJobBtn.addEventListener('click', handleSaveJob);
jobDescription.addEventListener('input', validateInputs);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadJobs();
});

// Drag and Drop Handlers
function handleDragOver(e) {
    if (uploadArea.classList.contains('uploaded-success')) {
        return;
    }
    e.preventDefault();
    uploadArea.style.borderColor = '#6366f1';
    uploadArea.style.background = '#f1f5f9';
}

function handleDrop(e) {
    if (uploadArea.classList.contains('uploaded-success')) {
        return;
    }
    e.preventDefault();
    uploadArea.style.borderColor = '#e2e8f0';
    uploadArea.style.background = '#f8fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    // Prevent new file selection if already uploaded
    if (uploadArea.classList.contains('uploaded-success')) {
        showToast('Please refresh the page to upload a new resume', 'error');
        return;
    }
    
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword'];
    
    if (!allowedTypes.includes(file.type)) {
        showToast('Please upload a PDF or DOCX file', 'error');
        return;
    }
    
    fileInfo.innerHTML = `
        <i class="fas fa-file-alt"></i>
        <div>
            <strong>${file.name}</strong>
            <p>Size: ${(file.size / 1024).toFixed(2)} KB</p>
        </div>
    `;
    fileInfo.classList.remove('hidden');
    uploadBtn.disabled = false;
    resumeFile.file = file;
}

// Upload Handler
async function handleUpload() {
    if (!resumeFile.files[0]) {
        showToast('Please select a file first', 'error');
        return;
    }
    
    showLoading('📄 Processing your resume...');
    
    const formData = new FormData();
    formData.append('resume', resumeFile.files[0]);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            resumeId = data.resume_id;
            
            // Update file info with success indicator
            const fileName = resumeFile.files[0].name;
            fileInfo.innerHTML = `
                <i class="fas fa-check-circle success-icon"></i>
                <div>
                    <strong>${fileName}</strong>
                    <p class="success-text"><i class="fas fa-check"></i> Resume uploaded successfully!</p>
                </div>
            `;
            fileInfo.classList.remove('hidden');
            
            // Disable upload button and change its appearance
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<i class="fas fa-check-circle"></i> Resume Uploaded';
            uploadBtn.classList.add('uploaded');
            
            // Update upload area to show success state
            uploadArea.classList.add('uploaded-success');
            uploadArea.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <p><strong>Resume Uploaded Successfully!</strong></p>
                <p class="file-types">${fileName}</p>
            `;
            uploadArea.style.cursor = 'default';
            uploadArea.style.borderColor = '#10b981';
            uploadArea.style.background = '#ecfdf5';
            
            showToast('Resume uploaded successfully!');
            validateInputs();
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        showToast('Error uploading resume: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Match Handler
async function handleMatch() {
    if (!resumeId) {
        showToast('Please upload a resume first', 'error');
        return;
    }
    
    const description = jobDescription.value.trim();
    if (!description) {
        showToast('Please enter a job description', 'error');
        return;
    }
    
    showLoading('🤖 AI is analyzing your resume...');
    
    try {
        const response = await fetch('/match', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                resume_id: resumeId,
                job_description: description
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showToast(data.error || 'Matching failed', 'error');
        }
    } catch (error) {
        showToast('Error matching resume: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Save Job Handler
async function handleSaveJob() {
    const description = jobDescription.value.trim();
    if (!description) {
        showToast('Please enter a job description', 'error');
        return;
    }
    
    showLoading('💾 Saving job description...');
    
    try {
        const response = await fetch('/jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: jobTitle.value.trim() || 'Untitled Job',
                description: description
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Job saved successfully!');
            jobTitle.value = '';
            jobDescription.value = '';
            loadJobs();
        } else {
            showToast(data.error || 'Failed to save job', 'error');
        }
    } catch (error) {
        showToast('Error saving job: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Display Results
function displayResults(data) {
    resultsSection.classList.remove('hidden');
    
    // Update match score
    const score = data.match_score;
    document.getElementById('matchScore').textContent = score;
    
    // Animate score ring with AI-powered color
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (score / 100) * circumference;
    const scoreRing = document.getElementById('scoreRing');
    scoreRing.style.strokeDashoffset = offset;
    
    // Use AI-determined color or default
    const matchColor = data.match_color || (score >= 70 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444');
    scoreRing.style.stroke = matchColor;
    
    // Update score label with quality
    const scoreLabel = document.querySelector('.score-label');
    if (data.match_quality && scoreLabel) {
        scoreLabel.innerHTML = `${data.match_quality} <span style="color: ${matchColor}">(${score}%)</span>`;
    }
    
    // Display matching skills
    matchingSkills.innerHTML = '';
    if (data.matching_skills && data.matching_skills.length > 0) {
        data.matching_skills.forEach(skill => {
            const tag = document.createElement('span');
            tag.className = 'skill-tag';
            tag.textContent = skill;
            matchingSkills.appendChild(tag);
        });
    } else {
        matchingSkills.innerHTML = '<p class="empty-state">No matching skills found</p>';
    }
    
    // Display missing skills
    missingSkills.innerHTML = '';
    if (data.missing_skills && data.missing_skills.length > 0) {
        data.missing_skills.forEach(skill => {
            const tag = document.createElement('span');
            tag.className = 'skill-tag';
            tag.textContent = skill;
            missingSkills.appendChild(tag);
        });
    } else {
        missingSkills.innerHTML = '<p class="empty-state">No missing skills!</p>';
    }
    
    // Display recommendations
    recommendationsList.innerHTML = '';
    if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'recommendation-item';
            item.innerHTML = `<p>${rec.message}</p>`;
            recommendationsList.appendChild(item);
        });
    } else {
        recommendationsList.innerHTML = '<p class="empty-state">Great job! Your resume looks good.</p>';
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Load Jobs
async function loadJobs() {
    try {
        const response = await fetch('/jobs');
        const data = await response.json();
        jobs = data.jobs || [];
        renderJobs();
    } catch (error) {
        console.error('Error loading jobs:', error);
    }
}

// Render Jobs
function renderJobs() {
    if (jobs.length === 0) {
        jobsList.innerHTML = '<div class="empty-state"><i class="fas fa-briefcase"></i><p>No saved jobs yet</p></div>';
        return;
    }
    
    jobsList.innerHTML = jobs.map(job => `
        <div class="job-item" onclick="loadJob(${job.id})">
            <h4>${job.title || 'Untitled Job'}</h4>
            <p>${job.description.substring(0, 100)}${job.description.length > 100 ? '...' : ''}</p>
            <div class="job-date">${new Date(job.created_date).toLocaleDateString()}</div>
        </div>
    `).join('');
}

// Load Job
async function loadJob(jobId) {
    const job = jobs.find(j => j.id === jobId);
    if (!job) return;
    
    jobTitle.value = job.title || '';
    jobDescription.value = job.description;
    
    if (resumeId) {
        // Auto-match if resume is already uploaded
        handleMatch();
    } else {
        showToast('Please upload a resume to match');
    }
}

// Compare with Job
async function compareWithJob(jobId) {
    if (!resumeId) {
        showToast('Please upload a resume first', 'error');
        return;
    }
    
    showLoading('🤖 AI is comparing your resume...');
    
    try {
        const response = await fetch('/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                resume_id: resumeId,
                job_id: jobId
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            showToast(`Matched with "${data.job_title || 'Job'}"`);
        } else {
            showToast(data.error || 'Comparison failed', 'error');
        }
    } catch (error) {
        showToast('Error comparing: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Validate Inputs
function validateInputs() {
    const hasResume = resumeId !== null;
    const hasJobDesc = jobDescription.value.trim().length > 0;
    matchBtn.disabled = !(hasResume && hasJobDesc);
}

// Show/Hide Loading
function showLoading(message = 'Processing...') {
    loadingOverlay.classList.remove('hidden');
    const loadingText = loadingOverlay.querySelector('p');
    if (loadingText) {
        loadingText.textContent = message;
    }
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// Show Toast
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}
