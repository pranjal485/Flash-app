from sqlalchemy.dialects.postgresql import UUID
import uuid
from configs import db, ma
import json


class User(db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pinecone_id = db.Column(UUID(as_uuid=True), unique=True, nullable=False)
    s3_file = db.Column(db.String(300))

    def __repr__(self):
        return '<User %r>' % self.username

    def to_dict(self):
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'pinecone_id': str(self.pinecone_id),
            's3_file': str(self.s3_file)
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class UserSchema(ma.Schema):
    class Meta:
        fields = (
            'id', 'username', 'email', 'pinecone_id', 's3_file'
        )


user_schema = UserSchema()
users_schema = UserSchema(many=True)
