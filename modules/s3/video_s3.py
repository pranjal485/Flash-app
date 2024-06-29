import os
from configs.config import s3_client
from botocore.exceptions import NoCredentialsError

AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")


class VideoS3:

    def upload_video(self, local_file, s3_filename):
        try:
            s3_client.upload_file(local_file, AWS_S3_BUCKET, s3_filename)
            print("Upload Successful")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False
