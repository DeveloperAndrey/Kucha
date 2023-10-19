// Функция для отображения информации и предпросмотра файла
function displayFileInfo() {
    var fileInput = document.getElementById('fileInput');
    var file = fileInput.files[0]; // Получаем выбранный файл

    if (file) {
        // Отображение параметров файла
        var imageInfo = document.getElementById('imageInfo');
        imageInfo.innerHTML = 'Имя файла: ' + file.name + '<br>' +
            'Тип файла: ' + file.type + '<br>' +
            'Размер файла: ' + (file.size / 1024).toFixed(2) + ' KB';

        // Создаем URL для предпросмотра выбранной картинки
        var imageUrl = URL.createObjectURL(file);
        var previewImage = document.getElementById('previewImage');
        previewImage.src = imageUrl;

        // Отображаем предпросмотр
        var imagePreview = document.getElementById('imagePreview');
        imagePreview.style.display = 'block';
    } else {
        // Если файл не выбран, очищаем информацию и предпросмотр
        var imageInfo = document.getElementById('imageInfo');
        imageInfo.innerHTML = '';
        var previewImage = document.getElementById('previewImage');
        previewImage.src = '';
        var imagePreview = document.getElementById('imagePreview');
        imagePreview.style.display = 'none';
    }
}

// Обработчик события изменения файла
document.getElementById('fileInput').addEventListener('change', displayFileInfo);

// Функция для отправки файла на сервер с использованием Fetch API
function uploadFile() {
    var fileInput = document.getElementById('fileInput');
    var file = fileInput.files[0]; // Получаем выбранный файл

    if (file) {
        var formData = new FormData(); // Создаем объект FormData для отправки файла на сервер
        formData.append('file', file); // Добавляем файл в FormData

        // Опции для запроса
        var options = {
            method: 'POST',
            body: formData,
        };

        // Выполняем запрос к серверу
        fetch('http://localhost:5000/upload', options)
            .then(response => {
                if (response.status === 200) {
                    // Файл успешно загружен
                    console.log('Файл успешно загружен.');
                } else {
                    // Произошла ошибка при загрузке файла
                    console.error('Ошибка при загрузке файла.');
                }
            })
            .catch(error => {
                console.error('Ошибка:', error);
            });
    } else {
        alert('Пожалуйста, выберите файл для загрузки.');
    }
}