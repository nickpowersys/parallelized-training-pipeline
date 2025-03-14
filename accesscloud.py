import os
from pathlib import PurePath, Path
import shutil
import uuid

_called_from_test = False

import boto3


if not os.path.exists('.env'):
    # Docker environment
    LOCAL_DATA_DIR = ''
    if 'S3_BUCKET' in os.environ:
        S3_BUCKET = os.environ['S3_BUCKET']
    else:
        raise ValueError('.env does not exist')
else:
    # Local environment
    from settings import LOCAL_DATA_DIR, S3_BUCKET
    # raise ValueError('wooho - in local environment!')



#S3_BUCKET = os.environ['S3_BUCKET']
#S3_BUCKET = os.getenv("S3_BUCKET")

# To avoid deterministically named files being on the
# same partition and improve performance, randomize first part
# of file name
def create_random_file_name(file_name, prefix_len=6):
    random_prefix = str(uuid.uuid4().hex[:prefix_len])
    return PurePath(''.join([random_prefix, file_name]))


def copy_to_random_path(file_name, prefix_len=None):
    file_path = PurePath(LOCAL_DATA_DIR).joinpath(file_name)
    random_file_name = create_random_file_name(file_name, prefix_len=prefix_len)
    random_file_path = PurePath(LOCAL_DATA_DIR).joinpath(random_file_name)
    shutil.copy(file_path, random_file_path)
    return file_path, random_file_name


# Bucket name includes randomly generated prefix
# to maximize chances of creating unique bucket name
def create_bucket_name(bucket_prefix):
    return ''.join([bucket_prefix, str(uuid.uuid4())])


def create_bucket(bucket_prefix, s3_connection):
    session = boto3.session.Session()
    current_region = session.region_name
    bucket_name = create_bucket_name(bucket_prefix)
    if current_region == 'us-east-1':
        bucket_response = s3_connection.create_bucket(
            Bucket=bucket_name)
    else:
        bucket_response = s3_connection.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={
                'LocationConstraint': current_region})
    print(bucket_name, current_region)
    return bucket_name, bucket_response


def download_s3_file(file_key, file_name):
    print('in download_s3_file')
    #s3_resource = boto3.resource('s3')
    #s3_resource.Object(S3_BUCKET, file_name).download_file(file_name)
    #s3 = boto3.resource('s3')
    # object key and name are same
    if not Path(file_name).exists():
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, file_key, file_name)
    else:
        print(f"{file_name} exists locally.")
    return file_name
