import boto3
from botocore.exceptions import ClientError

from consts import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY


boto_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)