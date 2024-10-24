FROM nvcr.io/nvidia/tensorrt:24.02-py3

WORKDIR /app

COPY . /app

RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

RUN pip install "flask[async]"

RUN pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt

RUN apt-get update

RUN apt-get install libgl1-mesa-glx -y

RUN apt-get install libglib2.0-0 -y

RUN apt-get install git -y

EXPOSE 4000

CMD ["python", "-m", "app"]

