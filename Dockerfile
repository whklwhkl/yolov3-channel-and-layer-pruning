FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

WORKDIR /app

COPY serving.py /app
COPY weights/half_h416_w416.pt /app

RUN pip install --no-cache flask \
    && pip install --no-cache torchvision==0.4.2 \
    && conda install -y opencv

EXPOSE 6670

CMD python serving.py --model half_h416_w416.pt --port 6670 --conf-thres 0.75 --nms-thres 0.5
