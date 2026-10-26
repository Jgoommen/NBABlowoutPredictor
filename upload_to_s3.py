import boto3

BUCKET_NAME = "student-s3-josh-2026-161012475511-us-east-2-an"
REGION = "us-east-2"

FILES_TO_UPLOAD = {
    "nbaMatchups.csv": "raw/nbaMatchups.csv",
    "nbaTeamFeatures.csv": "raw/nbaTeamFeatures.csv",  
}

s3 = boto3.client("s3", region_name=REGION)

for local_file, s3_key in FILES_TO_UPLOAD.items():
    s3.upload_file(local_file, BUCKET_NAME, s3_key)
    print(f"Uploaded {local_file} to s3://{BUCKET_NAME}/{s3_key}")

response = s3.list_objects_v2(Bucket=BUCKET_NAME)
print("\nCurrent bucket contents:")
for obj in response.get('Contents', []):
    print(f"  {obj['Key']}  ({obj['Size']} bytes)")