import os
import insightface
from flask import Flask as FlaskInit
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import boto3 as b

class Config(object):
    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI")
    AWS_S3_REGION = os.environ.get("AWS_S3_REGION")
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Flask app instance
app = FlaskInit(__name__)
app.config.from_object(Config)

# SQLAlchemy instance
db = SQLAlchemy(app)

# Marshmallow instance
ma = Marshmallow(app)

# model instance
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))

# AWS S3 connection
s3_client = b.client(
    service_name="s3",
    region_name=Config.AWS_S3_REGION,
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
)
