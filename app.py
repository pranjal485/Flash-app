import os
from configs.config import app
from flask import request
from modules.face_recognition.face_recognition import FaceRecognition

face_recognition = FaceRecognition()

@app.route('/health')
def health():
    return 'OK'

@app.route('/generate-distance-insight-face', methods=['POST'])
def generate_distance_insight_face():
    file1 = request.files['photo1']
    file2 = request.files['photo2']
    
    embedding = face_recognition.calculate_distance(file1, file2)
    return embedding.tolist()

@app.route('/load-vector-images', methods=['POST'])
def load_vector_images():
    vectors = face_recognition.load_vectors_from_images()
    return vectors

@app.route('/register', methods=['POST'])
def register():
    result = face_recognition.register(request.files, request.form)
    return result

@app.route('/face-recognize', methods=['POST'])
def recognize():
    return face_recognition.recognize(request.files, request.form)

@app.route('/delete-users-data', methods=['POST'])
def delete_users_data():
    return face_recognition.delete_users_data()

if __name__ == '__main__':
    PORT = os.getenv('PORT') or app.config['PORT']
    if PORT:
        app.run(port=PORT)
    else:
        app.run()