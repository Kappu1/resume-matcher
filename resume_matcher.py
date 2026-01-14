import re
import PyPDF2
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import os
import numpy as np

# Try to import advanced AI libraries
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using basic TF-IDF only.")

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        SPACY_AVAILABLE = False
        print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ResumeMatcher:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add common resume stopwords
        self.stop_words.update(['email', 'phone', 'address', 'linkedin', 'github', 
                               'www', 'http', 'https', 'com', 'edu', 'org'])
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        # Cache for skill extraction results
        self._skill_cache = {}
        
        # Initialize AI models if available
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ AI Semantic Model Loaded")
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}")
                self.semantic_model = None
        else:
            self.semantic_model = None
        
        # Comprehensive technical skills database with synonyms
        self.skill_keywords = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'web_frameworks': [
                'react', 'angular', 'vue', 'vue.js', 'next.js', 'nuxt', 'svelte',
                'django', 'flask', 'fastapi', 'express', 'spring', 'asp.net', 'laravel'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 
                'oracle', 'sqlite', 'dynamodb', 'neo4j', 'elasticsearch'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
                'kubernetes', 'docker', 'terraform', 'ansible'
            ],
            'ml_ai': [
                'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
                'scikit-learn', 'keras', 'opencv', 'pandas', 'numpy', 'matplotlib'
            ],
            'tools_methodologies': [
                'git', 'jenkins', 'ci/cd', 'agile', 'scrum', 'devops', 'microservices',
                'rest api', 'graphql', 'linux', 'unix', 'bash', 'shell scripting'
            ],
            'frontend': [
                'html', 'css', 'bootstrap', 'tailwind', 'sass', 'less', 'jquery',
                'redux', 'webpack', 'babel', 'npm', 'yarn'
            ]
        }
        
        # Flatten skills for quick lookup
        self.all_skills = []
        for category, skills in self.skill_keywords.items():
            self.all_skills.extend(skills)
        
        # Create lowercase skill lookup set for O(1) lookup
        self._skill_lookup = {skill.lower() for skill in self.all_skills}
        # Create skill patterns once for regex matching
        self._skill_patterns = [re.compile(r'\b' + re.escape(skill.lower()) + r'\b', re.IGNORECASE) 
                                for skill in self.all_skills]
    
    def extract_text_from_resume(self, filepath):
        """Extract text from PDF or DOCX resume"""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_from_pdf(filepath)
        elif file_ext in ['.docx', '.doc']:
            return self._extract_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _extract_from_pdf(self, filepath):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        return text.strip()
    
    def _extract_from_docx(self, filepath):
        """Extract text from DOCX file"""
        try:
            doc = Document(filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")
        return text.strip()
    
    def preprocess_text(self, text):
        """Preprocess and clean text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def extract_skills(self, text):
        """AI-powered skill extraction with context understanding - optimized"""
        # Check cache first
        text_hash = hash(text[:1000])  # Use first 1000 chars as cache key
        if text_hash in self._skill_cache:
            return self._skill_cache[text_hash]
        
        text_lower = text.lower()
        found_skills = set()  # Use set to avoid duplicates during processing
        
        # Extract skills from skill keywords database - optimized with precompiled patterns
        for i, pattern in enumerate(self._skill_patterns):
            if pattern.search(text_lower):
                found_skills.add(self.all_skills[i])
        
        # AI-enhanced pattern matching for skills section
        skill_patterns = [
            r'skills?[:\s]+([^\n]+)',
            r'technolog(?:y|ies)[:\s]+([^\n]+)',
            r'proficient in[:\s]+([^\n]+)',
            r'expertise in[:\s]+([^\n]+)',
            r'competent in[:\s]+([^\n]+)',
            r'experienced with[:\s]+([^\n]+)',
            r'familiar with[:\s]+([^\n]+)',
            r'tools?[:\s]+([^\n]+)',
            r'frameworks?[:\s]+([^\n]+)'
        ]
        
        # Compile patterns once
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in skill_patterns]
        for pattern in compiled_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                # Extract individual skills from the match
                words = re.split(r'[,\-••\n;]', match)
                for word in words:
                    word = word.strip()
                    if len(word) > 2 and not word.isdigit():
                        word_lower = word.lower()
                        # Fast lookup using set
                        skill_found = False
                        # Check if word contains or is contained in any skill
                        for skill_lower in self._skill_lookup:
                            if skill_lower in word_lower or word_lower in skill_lower:
                                # Find original skill name
                                for skill in self.all_skills:
                                    if skill.lower() == skill_lower:
                                        found_skills.add(skill)
                                        skill_found = True
                                        break
                                if skill_found:
                                    break
                        # Add if it looks like a skill (capitalized, short, etc.)
                        if not skill_found and len(word) < 30 and not word.startswith('http'):
                            found_skills.add(word.title())
        
        # Use spaCy for NER if available (extract technical terms) - limit text length
        if SPACY_AVAILABLE and nlp:
            try:
                # Limit to first 3000 chars for better performance
                text_sample = text[:3000] if len(text) > 3000 else text
                doc = nlp(text_sample)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'EVENT']:
                        text_clean = ent.text.lower().strip()
                        if len(text_clean) > 2 and len(text_clean) < 30:
                            found_skills.add(ent.text)
            except Exception:
                pass
        
        result = list(found_skills)
        # Cache the result
        self._skill_cache[text_hash] = result
        # Limit cache size to prevent memory issues
        if len(self._skill_cache) > 100:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._skill_cache))
            del self._skill_cache[oldest_key]
        
        return result
    
    def _truncate_text(self, text, max_length=2000):
        """Truncate text to max_length characters for faster processing"""
        if len(text) <= max_length:
            return text
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If period is in last 20%, use it
            return truncated[:last_period + 1]
        return truncated
    
    def calculate_match(self, resume_text, job_description):
        """AI-powered match calculation using multiple techniques - optimized"""
        # Truncate texts for faster processing (keep full text for skill extraction)
        resume_truncated = self._truncate_text(resume_text, 3000)
        job_truncated = self._truncate_text(job_description, 2000)
        
        # Preprocess both texts
        processed_resume = self.preprocess_text(resume_truncated)
        processed_job = self.preprocess_text(job_truncated)
        
        scores = []
        weights = []
        
        # 1. TF-IDF + Cosine Similarity (40% weight)
        tfidf_matrix = self.vectorizer.fit_transform([processed_resume, processed_job])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        scores.append(tfidf_score)
        weights.append(0.4)
        
        # 2. Semantic Similarity using Sentence Transformers (50% weight) - if available
        if self.semantic_model:
            try:
                # Truncate for semantic encoding (models have token limits)
                semantic_resume = self._truncate_text(resume_text, 512)
                semantic_job = self._truncate_text(job_description, 512)
                
                # Encode both texts
                resume_embedding = self.semantic_model.encode([semantic_resume], convert_to_numpy=True, show_progress_bar=False)
                job_embedding = self.semantic_model.encode([semantic_job], convert_to_numpy=True, show_progress_bar=False)
                
                # Calculate cosine similarity
                semantic_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
                scores.append(semantic_score)
                weights.append(0.5)
            except Exception as e:
                print(f"Semantic similarity error: {e}")
        
        # 3. Skills-based matching (10% weight) - use full text for skills
        resume_skills = set([s.lower() for s in self.extract_skills(resume_text)])
        job_skills = set([s.lower() for s in self.extract_skills(job_description)])
        
        if job_skills:
            skills_match_ratio = len(resume_skills.intersection(job_skills)) / len(job_skills)
            scores.append(skills_match_ratio)
            weights.append(0.1)
        else:
            # If no skills found, don't penalize
            scores.append(1.0)
            weights.append(0.1)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        match_percentage = sum(score * weight for score, weight in zip(scores, weights)) * 100
        
        # Extract skills (normalized for comparison)
        resume_skills_list = list(resume_skills)
        job_skills_list = list(job_skills)
        
        # Find missing and matching skills (using fuzzy matching for similar skills)
        matching_skills = []
        missing_skills = []
        
        for job_skill in job_skills_list:
            matched = False
            for resume_skill in resume_skills_list:
                # Exact match or substring match
                if job_skill in resume_skill or resume_skill in job_skill:
                    matching_skills.append(job_skill.title())
                    matched = True
                    break
            if not matched:
                missing_skills.append(job_skill.title())
        
        return match_percentage, missing_skills[:15], matching_skills[:15]
    
    def get_recommendations(self, resume_text, job_description):
        """AI-powered intelligent recommendations for resume improvement - optimized"""
        recommendations = []
        
        # Extract skills from both (normalized) - reuse from cache if available
        resume_skills = set([s.lower() for s in self.extract_skills(resume_text)])
        job_skills = set([s.lower() for s in self.extract_skills(job_description)])
        
        missing_skills = job_skills - resume_skills
        if missing_skills:
            top_missing = list(missing_skills)[:8]
            recommendations.append({
                'type': 'missing_skills',
                'message': f'🔧 Add these critical skills: {", ".join([s.title() for s in top_missing[:5]])}',
                'priority': 'high'
            })
        
        # AI-powered keyword analysis - use truncated text for faster processing
        resume_truncated = self._truncate_text(resume_text, 2000)
        job_truncated = self._truncate_text(job_description, 1500)
        processed_resume = self.preprocess_text(resume_truncated)
        processed_job = self.preprocess_text(job_truncated)
        
        # Extract important keywords using TF-IDF
        job_keywords = set(processed_job.split())
        resume_keywords = set(processed_resume.split())
        
        # Find important missing keywords
        missing_keywords = []
        for keyword in job_keywords:
            if keyword not in resume_keywords and len(keyword) > 4:
                # Weight by frequency in job description
                frequency = processed_job.count(keyword)
                if frequency > 2:
                    missing_keywords.append((keyword, frequency))
        
        if missing_keywords:
            top_missing_keywords = sorted(missing_keywords, key=lambda x: x[1], reverse=True)[:5]
            recommendations.append({
                'type': 'keywords',
                'message': f'💡 Incorporate these important keywords: {", ".join([k[0].title() for k in top_missing_keywords])}',
                'priority': 'medium'
            })
        
        # Experience analysis
        experience_keywords = ['experience', 'years', 'developed', 'implemented', 'managed', 'led', 'achieved']
        resume_has_experience = any(kw in processed_resume for kw in experience_keywords)
        job_requires_experience = any(kw in processed_job for kw in experience_keywords)
        
        if job_requires_experience and not resume_has_experience:
            recommendations.append({
                'type': 'experience',
                'message': '📈 Add quantifiable achievements and impact statements to highlight your experience.',
                'priority': 'high'
            })
        
        # Length and structure recommendations
        word_count = len(processed_resume.split())
        if word_count < 200:
            recommendations.append({
                'type': 'length',
                'message': '📝 Resume seems too brief. Add more details about projects, achievements, and responsibilities.',
                'priority': 'medium'
            })
        elif word_count > 1200:
            recommendations.append({
                'type': 'length',
                'message': '✂️ Resume is lengthy. Condense to highlight most relevant achievements (ATS systems prefer 1-2 pages).',
                'priority': 'low'
            })
        
        # Education and certification check
        education_keywords = ['degree', 'bachelor', 'master', 'phd', 'certification', 'certified']
        job_education = any(kw in processed_job for kw in education_keywords)
        resume_education = any(kw in processed_resume for kw in education_keywords)
        
        if job_education and not resume_education:
            recommendations.append({
                'type': 'education',
                'message': '🎓 Consider highlighting your educational background and relevant certifications.',
                'priority': 'medium'
            })
        
        # Action verbs recommendation
        action_verbs = ['developed', 'created', 'designed', 'implemented', 'optimized', 'led', 'managed']
        resume_verbs = [verb for verb in action_verbs if verb in processed_resume]
        if len(resume_verbs) < 3:
            recommendations.append({
                'type': 'writing',
                'message': '✍️ Use more action verbs (developed, created, implemented, led) to make your resume more impactful.',
                'priority': 'low'
            })
        
        return recommendations
