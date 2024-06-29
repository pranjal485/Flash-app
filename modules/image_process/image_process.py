import cv2
from configs.config import model


class ImageProcess:
    def extract_frames(self, video_path, interval=40):
        """Extract frames from a video at a given interval."""
        frames = []
        video_capture = cv2.VideoCapture(video_path)
        success, image = video_capture.read()

        image = self.get_resize(image)

        count = 0

        while success:
            if count % interval == 0:
                image = self.get_resize(image)
                frames.append(image)
            success, image = video_capture.read()

            count += 1

        video_capture.release()
        return frames

    def get_new_dimension(self, original_width, original_height):
        new_width = 400
        ratio = new_width / original_width

        new_height = int(original_height * ratio)

        return new_width, new_height
    
    def get_resize(self, image):

        original_width = image.shape[1]
        original_height = image.shape[0]

        new_dim = self.get_new_dimension(original_width, original_height)

        resized_img = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        return resized_img
    
    def get_embedding(self, image):
        faces = model.get(image)
        if faces:
            if len(faces) > 1:
                print("Multiple faces detected")
                return None
            else:
                return faces[0].embedding
        return None
