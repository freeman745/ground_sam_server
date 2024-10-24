import cv2
import requests
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


url = "http://127.0.0.1:4000/"
image = cv2.imread('out.jpg')
raw_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
data = (
        "predict",
        {
            "arr": image,
            "score_thr_multiplier": 1.0,
            "text": "bottle,bag,box",
            "plan": 2
        },
        )
s_request = pickle.dumps(data)
for i in tqdm(range(1)):
    s_response = requests.post(url, data=s_request).content
response = pickle.loads(s_response)

print(response)

print(len(response['bboxes']), len(response['labels']))
'''
plt.figure(figsize=(10, 10))
plt.imshow(raw_img)
for mask in response['segms']:
    show_mask(mask, plt.gca(), random_color=len(response['segms']) > 1)
for box in response['bboxes']:
    show_box(box, plt.gca())
plt.axis("off")
plt.savefig('output.jpg', bbox_inches="tight", dpi=300, pad_inches=0.0)
'''