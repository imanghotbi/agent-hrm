import aioboto3
from botocore.exceptions import ClientError
from app.config.config import config

class MinioHandler:
    def __init__(self):
        self.session = aioboto3.Session()
        self.config = {
            'endpoint_url': config.minio_endpoint,      
            'aws_access_key_id': config.minio_access_key, 
            'aws_secret_access_key': config.minio_secret_key.get_secret_value(), 
            'use_ssl': False
        }

    async def ensure_bucket(self):
        async with self.session.client("s3", **self.config) as s3:
            try:
                await s3.create_bucket(Bucket=config.minio_bucket) 
            except ClientError as e:
                if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
                    raise

    async def upload_file(self, file_path: str, object_name: str):
        async with self.session.client("s3", **self.config) as s3:
            await s3.upload_file(file_path, config.minio_bucket, object_name)
            print(f"✅ Uploaded: {object_name}")

    async def list_files(self) -> list[str]:
        async with self.session.client("s3", **self.config) as s3:
            response = await s3.list_objects_v2(Bucket=config.minio_bucket) 
            return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]

    async def download_file_bytes(self, object_name: str) -> bytes:
        async with self.session.client("s3", **self.config) as s3:
            response = await s3.get_object(Bucket=config.minio_bucket, Key=object_name)
            return await response['Body'].read()