# Prepare Data
```bash
python utils/extract_coco_bag_data.py
```

# Decide Anchor Sizes
```bash
make anchor
```

# Train
```bash
make train2
```

# Application
```bash
python jit_trace.py \
  --cfg cfg/yolov3-spp-4cls.cfg \
  --weights weights/best.pt \
  --device cuda \
  --half
docker build -t detector .
```
Then run the docker image to set up the detection service
`docker run -it --rm -p 6670:6670 --runtime nvidia detector`

# API
```python
import requests

ret = requests.post('http://{IP}:6670/det', files={'img':open('{IMAGE_PATH}', 'rb')})
ret.json()
# {'dets':[{'label':'person', 'conf':0.8, 'x1y1x2y2':[32, 64, 64, 128]},...]}
# or
# {} for no detection
```
