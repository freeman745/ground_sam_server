from flask import Flask, request, jsonify
import pickle
from configparser import ConfigParser
from gd_pred import GroundingDINO
from sam_onnx import SAM_ONNX
from sam_trt import SAM_TRT
import requests
import cv2
import threading


app = Flask(__name__)

config_path = 'config.ini'
config = ConfigParser()
config.read(config_path)
engine = config.get('SAM', 'engine')
gd_model = GroundingDINO(config_path)
if engine == 'onnx':
    sam_model = SAM_ONNX(config_path)
else:
    sam_model = SAM_TRT(config_path)

def call_internvl(internvl_server, s_req):
    requests.post(internvl_server, data=s_req)

@app.route('/health', methods=['GET'])
async def health():
    res = {'status':'ok', 'code':200}
    res = pickle.dumps(res)
    return res


@app.route('/', methods=['POST'])
async def predict():
    try:
        data = request.data
        method, data = pickle.loads(data)
        image = data['arr']
        try:
            text = data['text']
        except:
            text = 'product'
        try:
            box_thresh = data['box_thresh']
        except:
            box_thresh = 0.2
        try:
            text_thresh = data['text_thresh']
        except:
            text_thresh = 0.2

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            prompt = data['prompt']
        except:
            prompt = config.get('InternVL', 'prompt')

        bbox, label = gd_model.bbox_predict(img, text, box_thresh, text_thresh)

        if not bbox:
            output = {
                    'bboxes': bbox,
                    'segms': [],
                    'labels': label
                }
            res = pickle.dumps(output)
        
            return res

        mask = sam_model.mask_predict(img, bbox)

        internvl_server = "http://"+str(config.get('InternVL', 'ip'))+':'+str(config.get('InternVL', 'port'))+'/'

        req = {
            "arr": image,
            "bbox": bbox,
            "prompt": prompt
        }

        s_req = pickle.dumps(req)
        threading.Thread(target=call_internvl, args=(internvl_server, s_req,)).start()

        label = ['complete' for i in bbox] #for NFC, only output complete tag

        output = {
                    'bboxes': bbox,
                    'segms': mask,
                    'labels': label
        }
        res = pickle.dumps(output)
        
        return res
    except Exception as e:
        res = {'status':str(e), 'code':300}
        res = pickle.dumps(res)

        return res
    

if __name__ == '__main__':
    ip = config.get('Server', 'ip')
    port = config.get('Server', 'port')
    app.config['JSON_AS_ASCII'] = False
    app.run(host=ip, port=port, threaded=False)
