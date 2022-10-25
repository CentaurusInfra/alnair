import boto3
import glob
from PIL import Image
import pickle
import multiprocessing
import concurrent.futures


session = boto3.Session()
s3 = session.client("s3")
bucket = 'zhuangwei-bucket'


def preprocess(path):
    key = 'Imagenet-Mini-Obj/{}'.format(path)
    img = Image.open(path)
    img = img.convert("RGB")
    obj = {'size': img.size, 'body': img.tobytes()}
    s3.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(obj))
    return key
 
def upload_objects(folder):
    print(folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        imgs = glob.glob('{}/*/*'.format(folder))
        for path in imgs:
            futures.append(executor.submit(preprocess, path))
        concurrent.futures.wait(futures)
            
            
if __name__=="__main__":
    upload_objects('train')
    upload_objects('val')