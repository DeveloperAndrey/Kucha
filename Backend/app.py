from flask import Flask, request, jsonify
import os
from flask_cors import CORS  # Импортируйте библиотеку Flask-CORS
import random
from datetime import datetime

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
        # Генерируем уникальное имя файла на основе времени загрузки и расширения файла
        current_time = datetime.now()
        filename = current_time.strftime('%Y%m%d%H%M%S%f') + os.path.splitext(file.filename)[1]

        # Сохраняем файл с уникальным именем
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Генерируем случайное число (замените его на вашу логику)
        generated_number = random.randint(1, 100)

        return jsonify({'message': 'File uploaded successfully', 'generated_number': generated_number})

if __name__ == '__main__':
    app.run(debug=True)
