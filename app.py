from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
from resume_matcher import ResumeMatcher
from database import init_db, add_resume, add_job, get_all_jobs, get_resume_by_id
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize database
init_db()

# Initialize resume matcher
matcher = ResumeMatcher()

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Extract text from resume
        try:
            resume_text = matcher.extract_text_from_resume(filepath)
            resume_id = add_resume(unique_filename, resume_text)
            
            # Store resume_id in session
            session['resume_id'] = resume_id
            
            return jsonify({
                'success': True,
                'resume_id': resume_id,
                'message': 'Resume uploaded successfully'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing resume: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/match', methods=['POST'])
def match_resume():
    data = request.get_json()
    
    resume_id = data.get('resume_id') or session.get('resume_id')
    job_description = data.get('job_description', '')
    
    if not resume_id:
        return jsonify({'error': 'Please upload a resume first'}), 400
    
    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400
    
    try:
        # Get resume text
        resume_data = get_resume_by_id(resume_id)
        if not resume_data:
            return jsonify({'error': 'Resume not found'}), 404
        
        resume_text = resume_data['text']
        
        # Calculate match using AI
        match_score, missing_skills, matching_skills = matcher.calculate_match(
            resume_text, job_description
        )
        
        # Get AI-powered recommendations
        recommendations = matcher.get_recommendations(resume_text, job_description)
        
        # Determine match quality
        if match_score >= 80:
            match_quality = "Excellent Match"
            match_color = "#10b981"
        elif match_score >= 60:
            match_quality = "Good Match"
            match_color = "#3b82f6"
        elif match_score >= 40:
            match_quality = "Fair Match"
            match_color = "#f59e0b"
        else:
            match_quality = "Needs Improvement"
            match_color = "#ef4444"
        
        return jsonify({
            'success': True,
            'match_score': round(match_score, 2),
            'match_quality': match_quality,
            'match_color': match_color,
            'missing_skills': missing_skills,
            'matching_skills': matching_skills,
            'recommendations': recommendations,
            'ai_powered': True
        })
    except Exception as e:
        return jsonify({'error': f'Error matching resume: {str(e)}'}), 500

@app.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = get_all_jobs()
    return jsonify({'jobs': jobs})

@app.route('/jobs', methods=['POST'])
def add_job_route():
    data = request.get_json()
    title = data.get('title', '')
    description = data.get('description', '')
    
    if not description:
        return jsonify({'error': 'Job description is required'}), 400
    
    job_id = add_job(title, description)
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Job added successfully'
    })

@app.route('/compare', methods=['POST'])
def compare_with_job():
    data = request.get_json()
    resume_id = data.get('resume_id') or session.get('resume_id')
    job_id = data.get('job_id')
    
    if not resume_id:
        return jsonify({'error': 'Please upload a resume first'}), 400
    
    if not job_id:
        return jsonify({'error': 'Job ID is required'}), 400
    
    try:
        # Get resume and job data
        resume_data = get_resume_by_id(resume_id)
        jobs = get_all_jobs()
        job = next((j for j in jobs if j['id'] == job_id), None)
        
        if not resume_data or not job:
            return jsonify({'error': 'Resume or job not found'}), 404
        
        resume_text = resume_data['text']
        job_description = job['description']
        
        # Calculate match using AI
        match_score, missing_skills, matching_skills = matcher.calculate_match(
            resume_text, job_description
        )
        
        recommendations = matcher.get_recommendations(resume_text, job_description)
        
        # Determine match quality
        if match_score >= 80:
            match_quality = "Excellent Match"
            match_color = "#10b981"
        elif match_score >= 60:
            match_quality = "Good Match"
            match_color = "#3b82f6"
        elif match_score >= 40:
            match_quality = "Fair Match"
            match_color = "#f59e0b"
        else:
            match_quality = "Needs Improvement"
            match_color = "#ef4444"
        
        return jsonify({
            'success': True,
            'match_score': round(match_score, 2),
            'match_quality': match_quality,
            'match_color': match_color,
            'missing_skills': missing_skills,
            'matching_skills': matching_skills,
            'recommendations': recommendations,
            'job_title': job['title'],
            'ai_powered': True
        })
    except Exception as e:
        return jsonify({'error': f'Error comparing: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
