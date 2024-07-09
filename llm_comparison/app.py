from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import time
from werkzeug.utils import secure_filename
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langdetect import detect

from get_embedding_function import get_embedding_function
from populate_database import main as populate_db_main
from query_data import query_rag, select_prompt_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'

CHROMA_PATH = "chroma"

LANGUAGE_MODELS = {
    'gemma2': 'gemma2',
    'llama3': 'llama3',
    'mistral': 'mistral'
}

@app.route('/')
def index():
    return render_template('index.html', models=LANGUAGE_MODELS.keys())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))

@app.route('/query', methods=['POST'])
def query():
    question = request.form['question']
    model_name = request.form['model']
    language_model = LANGUAGE_MODELS.get(model_name, 'mistral')
    
    # Process and add PDFs to the Chroma database
    populate_db_main()

    # Get response from the language model
    response_text = query_rag(question, language_model)
    
    return render_template('index.html', models=LANGUAGE_MODELS.keys(), response=response_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)