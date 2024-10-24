import requests
import json
import base64
from io import BytesIO
from PIL import Image
import time


root = 'test_img/'
count = 0
log = ''

while True:
    try:
        response = requests.get('http://localhost:6262/result/detection')

        img = json.loads(response.text)['img']

        if count != 0:
            if log == img:
                continue

        image_data = base64.b64decode(img)
        image = Image.open(BytesIO(image_data))

        img_name = root+str(int(time.time()))+'.png'
        image.save(img_name)

        count += 1
        log = img

        time.sleep(10)
    except:
        continue


