# Resume Matcher AI 🤖

An **advanced AI-powered** web application that intelligently analyzes resumes and matches them with relevant job descriptions using state-of-the-art Natural Language Processing (NLP), semantic analysis, and machine learning techniques.

## Features

🔹 **Resume Upload & Parsing**
- Supports PDF and DOCX formats
- Extracts text using NLP techniques
- Removes stopwords and irrelevant data

🔹 **Job Description Analysis**
- Accepts job descriptions as input
- Extracts required skills and keywords
- Normalizes data for comparison

🔹 **Matching Algorithm**
- Uses TF-IDF vectorization and Cosine Similarity
- Calculates resume-job match percentage
- Ranks candidates/jobs based on relevance

🔹 **Skill Gap Analysis**
- Highlights missing skills
- Shows matching skills
- Provides recommendations for resume improvement

🔹 **User-Friendly Interface**
- Simple drag-and-drop resume upload
- Displays match score visually
- Shows recommended jobs
- Saves and manages job descriptions

## Technologies Used

- **Backend**: Python, Flask
- **AI/NLP**: 
  - Sentence Transformers (Semantic Similarity)
  - NLTK (Natural Language Toolkit)
  - spaCy (Advanced NLP)
  - Scikit-learn (TF-IDF, Cosine Similarity)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **File Handling**: PyPDF2, python-docx

## AI Features

🧠 **Advanced Semantic Matching**
- Sentence Transformers for context-aware similarity
- Multi-algorithm approach (TF-IDF + Semantic + Skills-based)
- Weighted scoring for accurate results

🎯 **Intelligent Skill Extraction**
- Context-aware skill detection
- Comprehensive skills database with synonyms
- Pattern recognition for skill sections

💡 **AI-Powered Recommendations**
- Smart keyword suggestions
- Experience and education analysis
- Action verb recommendations
- Resume optimization tips

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd "resume matcher"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy language model (Optional but recommended)**
   ```bash
   python -m spacy download en_core_web_sm
   ```
   Note: The system works without spaCy but will have enhanced features with it.

4. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## Usage

1. **Upload Resume**
   - Click on the upload area or drag and drop a PDF/DOCX resume
   - Click "Upload Resume" button

2. **Enter Job Description**
   - Paste or type the job description in the text area
   - Optionally add a job title
   - Click "Match Resume" to see results

3. **View Results**
   - See your match score (percentage)
   - Review matching skills (green)
   - Check missing skills (yellow)
   - Read recommendations for improvement

4. **Save Jobs**
   - Click "Save Job" to store job descriptions
   - View saved jobs in the "Saved Jobs" section
   - Click on a saved job to auto-fill and match

## Project Structure

```
resume matcher/
├── app.py                 # Main Flask application
├── resume_matcher.py      # Core matching logic and NLP
├── database.py            # Database operations
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   ├── style.css         # CSS styles
│   └── script.js         # JavaScript functionality
└── uploads/              # Uploaded resumes (auto-created)
```

## Algorithm Details

### TF-IDF Vectorization
- Converts text into numerical vectors
- Weights words by importance (frequency vs. rarity)

### Cosine Similarity
- Measures the angle between two vectors
- Returns a similarity score between 0 and 1
- Converted to percentage (0-100%)

### Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Lemmatization
- Special character removal

## API Endpoints

- `GET /` - Main page
- `POST /upload` - Upload resume file
- `POST /match` - Match resume with job description
- `GET /jobs` - Get all saved jobs
- `POST /jobs` - Save a new job
- `POST /compare` - Compare resume with saved job

## Limitations

- Keyword-based matching may miss context
- Requires quality resume formatting
- Complex job roles may need advanced models
- English language only (currently)

## Future Enhancements

- Deep Learning models (BERT/GPT embeddings)
- Resume ranking dashboard
- ATS integration
- Multi-language resume support
- Interview question recommendations
- Batch processing for recruiters

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to fork, modify, and enhance this project. Contributions are welcome!

## Support

For issues or questions, please create an issue in the repository or contact the development team.
