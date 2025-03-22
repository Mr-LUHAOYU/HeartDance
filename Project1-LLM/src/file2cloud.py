# -*- coding: utf-8 -*-
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OSS_ACCESS_KEY_ID'] = os.getenv('OSS_ACCESS_KEY_ID')
os.environ['OSS_ACCESS_KEY_SECRET'] = os.getenv('OSS_ACCESS_KEY_SECRET')

auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(
    auth,
    'https://oss-cn-shanghai.aliyuncs.com',
    'website-lhy'
)


def download_from_oss(yourObjectName):
    return bucket.get_object(yourObjectName)


def upload_to_oss(yourObjectName, yourLocalFile):
    bucket.put_object_from_file(yourObjectName, yourLocalFile)
    return f"https://website-lhy.oss-cn-shanghai.aliyuncs.com/{yourObjectName}"


def delete_from_oss(yourObjectName):
    bucket.delete_object(yourObjectName)
