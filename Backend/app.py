from flask import Flask, request, jsonify
import os
from flask_cors import CORS  # Импортируйте библиотеку Flask-CORS

app = Flask(__name__)
CORS(app)  # Инициализируйте Flask-приложение для обработки CORS
app.config['UPLOAD_FOLDER'] = 'uploads'  # Папка для сохранения загруженных файлов
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла (16MB)

CORS(app)  # Инициализируйте Flask-приложение для обработки CORS
app.config['UPLOAD_FOLDER'] = '../uploads'  # Папка для сохранения загруженных файлов
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла (16MB)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify({'message': 'File uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True)
