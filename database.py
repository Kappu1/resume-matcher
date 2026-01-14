import sqlite3
import os
from datetime import datetime

DB_NAME = 'resume_matcher.db'

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create resumes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            text TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create matches table (optional - for storing match history)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            job_id INTEGER,
            match_score REAL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resume_id) REFERENCES resumes (id),
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_resume(filename, text):
    """Add a new resume to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO resumes (filename, text) VALUES (?, ?)
    ''', (filename, text))
    
    resume_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return resume_id

def get_resume_by_id(resume_id):
    """Get resume by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def add_job(title, description):
    """Add a new job to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO jobs (title, description) VALUES (?, ?)
    ''', (title or 'Untitled Job', description))
    
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return job_id

def get_all_jobs():
    """Get all jobs from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM jobs ORDER BY created_date DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def save_match(resume_id, job_id, match_score):
    """Save a match result to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO matches (resume_id, job_id, match_score) VALUES (?, ?, ?)
    ''', (resume_id, job_id, match_score))
    
    match_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return match_id
