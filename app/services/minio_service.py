import aioboto3
from botocore.exceptions import ClientError
from app.config.config import config
from app.config.logger import logger

class MinioHandler:
    def __init__(self):
        self.session = aioboto3.Session()
        self.config = {
            'endpoint_url': config.minio_endpoint,      
            'aws_access_key_id': config.minio_access_key, 
            'aws_secret_access_key': config.minio_secret_key.get_secret_value(), 
            'use_ssl': False
        }

    async def ensure_bucket(self , bucket_name:str):
        async with self.session.client("s3", **self.config) as s3:
            try:
                await s3.create_bucket(Bucket=bucket_name) 
            except ClientError as e:
                if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
                    raise

    async def upload_file(self, file_path: str, bucket_name:str ,object_name: str):
        async with self.session.client("s3", **self.config) as s3:
            await s3.upload_file(file_path, bucket_name, object_name)
            logger.info(f"✅ Uploaded: {object_name}")

    async def list_files(self , bucket_name:str) -> list[str]:
        async with self.session.client("s3", **self.config) as s3:
            response = await s3.list_objects_v2(Bucket=bucket_name) 
            return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]

    async def download_file_bytes(self, bucket_name:str ,object_name: str) -> bytes:
        async with self.session.client("s3", **self.config) as s3:
            response = await s3.get_object(Bucket=bucket_name, Key=object_name)
            return await response['Body'].read()

    async def empty_bucket(self,bucket_name:str):
        """
        Deletes all objects in the configured bucket.
        """
        async with self.session.client("s3", **self.config) as s3:
            # 1. Use a paginator to handle buckets with more than 1000 files
            paginator = s3.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(Bucket=bucket_name):
                # 2. Check if the page has contents
                if 'Contents' in page:
                    # 3. Prepare list of objects to delete
                    objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                    
                    if objects_to_delete:
                        # 4. Perform batch deletion
                        await s3.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
                        logger.info(f"🗑️ Batch deleted {len(objects_to_delete)} files from {bucket_name}")
            
            logger.info(f"✅ Bucket '{bucket_name}' has been successfully emptied.")