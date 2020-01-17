import argparse, cv2, re, torch, numpy as np

from torchvision.ops import nms

from flask import Flask, request, jsonify
app = Flask('yolov3')
from io import BytesIO
from PIL import Image


def preprocess(img:np.array, input_size, device, dtype):
    h, w, _ = img.shape
    if h * input_size[1] > w * input_size[0]:
        img = np.pad(img, [(0, 0), (0, int(h * input_size[1] / input_size[0]) - w), (0, 0)], mode='constant')
    else:
        img = np.pad(img, [(0, int(w * input_size[0] / input_size[1]) - h), (0, 0), (0, 0)], mode='constant')
    new_shape = img.shape[:-1]
    img = cv2.resize(img, input_size[::-1]).transpose([2, 0, 1]) / 255.  # normalize
    img = torch.from_numpy(img).type(dtype).to(device)
    return img, new_shape


def infer(model, image:torch.Tensor, confidence_threshold:float, iou_threshold:float):
    """
        image:
            value range from 0 to 1,
            shape [C, H, W]
    """
    with torch.no_grad():
        # outputs range from 0 to 1
        ret = model(image[None])
        box, obj_prob = ret[:2]  # box:(xc, yc, w, h), obj_score
        box_wh = box[:, 2:] / 2
        box_xy = box[:, :2]
        box = torch.cat([box_xy - box_wh, box_xy + box_wh], 1)  # left, top, right, bottom
        keep = torch.where(obj_prob > confidence_threshold)[0]
        obj_prob = obj_prob[keep]
        keep = keep[nms(box[keep], obj_prob, iou_threshold)]
    return box[keep], ret[2][keep] if len(ret) > 2 else obj_prob  # class_conf or obj_score


def postprocess(box:torch.Tensor, img_shape):
    box[:, [0, 2]] *= img_shape[1]
    box[:, [1, 3]] *= img_shape[0]
    return box.int().tolist()


@app.route('/det', methods=['POST'])
def det():
    f = request.files['img']
    imb = BytesIO(f.read())
    img = Image.open(imb)
    img = np.array(img)

    with torch.no_grad():
        img, new_shape = preprocess(img, img_size, device, torch.float16)
        box, pred = infer(model, img, args.conf_thres, args.nms_thres)
        detections = postprocess(box, new_shape)
    ret = {}
    if detections is not None:
        for p, (l, t, r, b) in zip(pred, box):
            class_prob, class_idx = p.max(0)
            class_idx = int(class_idx)
            if class_idx != 0 and class_idx != 25 and class_idx != 27 and class_idx != 29:
                continue
            ret.setdefault('dets', []).append(
                {'label':classes[class_idx],
                 'conf':class_prob.item(),
                 'x1y1x2y2':[l.item(), t.item(), r.item(), b.item()]
                 })
    return jsonify(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='weights/half_h416_w416.pt', help='path to jit model file')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument("--port", default=6666)
    args = parser.parse_args()
    model = torch.jit.load(args.model).half().cuda().eval()
    img_size = (416, 416)
    device = 'cuda'
    classes = ['person','bicycle','car','motorcycle','airplane','bus','train',
               'truck','boat','traffic light','fire hydrant','stop sign',
               'parking meter','bench','bird','cat','dog','horse','sheep',
               'cow','elephant','bear','zebra','giraffe','backpack','umbrella',
               'handbag','tie','suitcase','frisbee','skis','snowboard',
               'sports ball','kite','baseball bat','baseball glove',
               'skateboard','surfboard','tennis racket','bottle','wine glass',
               'cup','fork','knife','spoon','bowl','banana','apple','sandwich',
               'orange','broccoli','carrot','hot dog','pizza','donut','cake',
               'chair','couch','potted plant','bed','dining table','toilet',
               'tv','laptop','mouse','remote','keyboard','cell phone',
               'microwave','oven','toaster','sink','refrigerator','book',
               'clock','vase','scissors','teddy bear','hair drier','toothbrush']
    app.run(host="0.0.0.0", debug=True, port=args.port)
