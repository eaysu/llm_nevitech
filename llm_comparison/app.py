from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import os
from populate_database import main as populate_db_main
from query_data import query_rag

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
socketio = SocketIO(app)

CHROMA_PATH = "chroma"

LANGUAGE_MODELS = {
    'gemma2': 'gemma2',
    'llama3': 'llama3',
    'mistral': 'mistral',
    'qwen2:7b': 'qwen2:7b',
    'gemma2:27b': 'gemma2:27b',
    'llama3:70b': 'llama3:70b',
    'qwen2:72b': 'qwen2:72b',
    'mixtral:8x7b': 'mixtral:8x7b',
    'curiositytech/MARS': 'curiositytech/MARS',
    'Orbina/Orbita-v0.1': 'Orbina/Orbita-v0.1',
    'Eurdem/Defne_llama3_2x8B': 'Eurdem/Defne_llama3_2x8B',
    'Metin/LLaMA-3-8B-Instruct-TR-DPO': 'Metin/LLaMA-3-8B-Instruct-TR-DPO',
    'ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1': 'ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1',
    'meta-llama/Meta-Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'Eurdem/megatron_1.1_MoE_2x7B': 'Eurdem/megatron_1.1_MoE_2x7B',
    'mistralai/Mistral-Nemo-Instruct-2407': 'mistralai/Mistral-Nemo-Instruct-2407',
    'mistralai/Mistral-7B-v0.3': 'mistralai/Mistral-7B-v0.3'
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

@socketio.on('query')
def handle_query(data):
    question = data['question']
    model_name = data['model']
    language_model = LANGUAGE_MODELS.get(model_name, 'mistral')
    
    # Process and add PDFs to the Chroma database
    populate_db_main()

    # Get response from the language model
    for response_part in query_rag(question, language_model):
        emit('response', {'data': response_part})
        socketio.sleep(0.1)  # Adjust as necessary for more realistic streaming

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
