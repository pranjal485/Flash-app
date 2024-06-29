from flask import jsonify
from pinecone import Pinecone
import numpy as np
import os
import cv2
import json
import base64
import requests
import uuid
from models.user import User
from configs import db
from modules.users.users import UserRepository
from modules.image_process.image_process import ImageProcess
from scipy.spatial.distance import cosine
from modules.s3.video_s3 import VideoS3

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
IMAGE_DIR = "dataset/"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
SERVER_URL = "https://infer.roboflow.com"

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("images")

threshold = 0.4


class FaceRecognition:
    def __init__(self):
        self.users_repository = UserRepository()
        self.image_process = ImageProcess()
        self.video_s3 = VideoS3()
        self.vectors = []

    def calc_cos_sim(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calculate_distance(self, file1, file2):
        image1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        embedding_image1 = self.image_process.get_embedding(image1)
        embedding_image2 = self.image_process.get_embedding(image2)

        distance = self.calc_cos_sim(embedding_image1, embedding_image2)

        return embedding_image1

    def register(self, files, form):
        """Register a user with a face image."""
        if 'video' not in files:
            return jsonify({'error': 'No photo file provided'}), 400

        file = files['video']
        username = form.get('username')
        email = form.get('email')
        
        if not (username or email):
            return jsonify({'success': False, 'error': 'Missing username or email'}), 400

        user = self.users_repository.getByUsernameAndEmail(username, email)

        if user:
            return jsonify({'success': False, 'error': 'Username or email already exists'}), 400

        # Save temporary video file
        temp_video_path = f'temp_{username}.mp4'
        
        file.save(temp_video_path)

        # Extract frames and embeddings
        frames = self.image_process.extract_frames(temp_video_path)
        embeddings = [self.image_process.get_embedding(frame) for frame in frames if frame is not None]

        s3_filename = f"{username}_{uuid.uuid4()}.mp4"
        upload_result = self.video_s3.upload_video(temp_video_path, s3_filename)

        if not upload_result:
            return jsonify({'error': 'AWS couldn\'t be uploaded'}), 400

        # Cleanup the temporary video file
        os.remove(temp_video_path)

        # Filter out None embeddings if face detection fails on some frames
        list_embeddings = [embedding for embedding in embeddings if embedding is not None]

        if not list_embeddings:
            return jsonify({'error': 'None or multiple faces detected'}), 400

        # Create the list of embeddings in one ndarray
        embeddings = np.concatenate(list_embeddings, axis=0)

        pinecone_object = [{"id": str(uuid.uuid4()), "values": embeddings, "metadata": {"username": username}}]

        index.upsert(vectors=pinecone_object)

        new_user = User(
            username=username,
            email=email,
            pinecone_id=pinecone_object[0].get('id'),
            s3_file=s3_filename,
        )

        db.session.add(new_user)
        db.session.commit()

        # Optionally, return any details you want about the detected face, like coordinates
        return jsonify({'success': True, 'user': username}), 200

    def recognize(self, files, form):
        if 'photo' not in files:
            return jsonify({'error': 'No photo file provided'}), 400

        file = files['photo']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        username = form.get('username')
        email = form.get('email')

        if not (username or email):
            return jsonify({'error': 'Missing username or email'}), 400

        user = self.users_repository.getByUsernameAndEmail(username, email)
        
        if not user:
            return jsonify({
                'success': False,
                'details': f'{username} not registered'
            }), 404

        user_vector = index.query(
            id=str(user.pinecone_id),
            top_k=1,
            include_values=True,
        )
        
        vector_embeddings = user_vector.matches[0].values

        image_embedding = self.image_process.get_embedding(image)

        sub_arrays_embeddings = np.array_split(vector_embeddings, 5)

        distance = 0.0
        min_distance = 0.0

        for embedding in sub_arrays_embeddings:
            distance = distance + cosine(image_embedding, embedding)
        
        
        print(distance)

        print(distance)
        min_distance = distance / 5

        threshold = 0.4
        
        if min_distance < threshold:
            return jsonify({'success': True, 'user': user.username}), 200
        else:
            return jsonify({'success': False, 'details': f'You are not {user.username}'}), 200
        


    def load_vectors_from_images(self):
        for i, image in enumerate(os.listdir(IMAGE_DIR)):
            print(f"Processing image {image}")
            infer_clip_payload = {
                "image": {
                    "type": "base64",
                    "value": base64.b64encode(open(IMAGE_DIR + image, "rb").read()).decode("utf-8"),
                },
            }

            res = requests.post(
                f"{SERVER_URL}/clip/embed_image?api_key={ROBOFLOW_API_KEY}",
                json=infer_clip_payload,
            )

            embeddings = res.json()['embeddings']

            self.vectors.append({"id": str(uuid.uuid4()), "values": embeddings[0], "metadata": {"filename": image}})

        index.upsert(vectors=self.vectors)
        
        return json.dumps({ self.vectors.tolist() })

    def delete_users_data(self):
        index.delete(
            delete_all=True
        )
        self.users_repository.delete_all()

        return jsonify({'success': True }), 200

    