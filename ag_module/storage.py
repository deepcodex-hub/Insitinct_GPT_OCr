import boto3
from botocore.exceptions import NoCredentialsError
import os

class CloudStorage:
    def __init__(self, bucket_name, endpoint_url=None, access_key=None, secret_key=None):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url or os.getenv('S3_ENDPOINT'),
            aws_access_key_id=access_key or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        )

    def upload_file(self, file_path, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_path)
        try:
            self.s3.upload_file(file_path, self.bucket_name, object_name)
            return True
        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def generate_presigned_url(self, object_name, expiration=3600):
        try:
            response = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Could not generate presigned URL: {e}")
            return None

    def generate_presigned_post(self, object_name, expiration=3600):
        try:
            response = self.s3.generate_presigned_post(
                self.bucket_name,
                object_name,
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Could not generate presigned POST: {e}")
            return None
