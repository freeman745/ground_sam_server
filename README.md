# ground-sam-server
Using GroundingDINO and SAM for zero-shot detection and segmentation.
Apply TensorRT and efficientSAM.

## Build docker

build once then RPS can automatically start and stop the container

Remember to download 3 weight files in the folder.

```python
docker build -t ground_sam_server .
```

## Manually run

```python
docker run -id --network="host" --gpus "device=1" --shm-size="8g" --name "ground_sam_server" ground_sam_server /bin/bash

docker exec -it ground_sam_server /bin/bash

python app.py
```

## TensorRT
need to re-generate from ONNX when using different GPU/CUDA/Driver etc

```python
# Export Encoder
trtexec --onnx=l2_encoder.onnx --minShapes=input_image:1x3x512x512 --optShapes=input_image:4x3x512x512 --maxShapes=input_image:4x3x512x512 --saveEngine=l2_encoder.engine
```

```python
# Export Decoder
trtexec --onnx=l2_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=l2_decoder.engine
```
