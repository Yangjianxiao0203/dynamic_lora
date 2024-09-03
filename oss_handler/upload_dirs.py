from dotenv import load_dotenv
import oss2
import zipfile
import os

load_dotenv()

access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
bucket_name = os.getenv('OSS_BUCKET_NAME')
endpoint = os.getenv('OSS_ENDPOINT')

# 初始化OSS Bucket
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

def zip_folder(folder_path, zip_name):
    """将文件夹压缩为ZIP文件"""
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def upload_to_oss(bucket, file_name, object_name):
    """上传文件到OSS"""
    with open(file_name, 'rb') as file:
        bucket.put_object(object_name, file)

def compress_and_upload(folder_path, zip_name, object_name):
    """压缩文件夹并上传到OSS"""
    # 压缩文件夹
    zip_folder(folder_path, zip_name)
    # 上传到OSS
    upload_to_oss(bucket, zip_name, object_name)
    # 删除本地压缩文件
    os.remove(zip_name)

if __name__ == '__main__':

    # folder_to_upload = '/Users/jianxiaoyang/Documents/deepLearning/qwen_stf/final_models'  # 需要压缩并上传的文件夹路径
    zip_file_name = 'test_models.zip'  # 压缩后的文件名
    oss_object_name = '__jianxiao__/test/test_models.zip'  # OSS中的对象名

    # compress_and_upload(folder_to_upload, zip_file_name, oss_object_name)
    upload_to_oss(bucket, zip_file_name, oss_object_name)

    print("文件夹已成功压缩并上传到OSS。")
