import boto3
# s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('imagesairbnb')
for file in my_bucket.objects.all():
    print(file.key)
# def download_images():
#     pass
