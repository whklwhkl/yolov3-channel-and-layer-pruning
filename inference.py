import argparse, cv2, re, torch, numpy as np

from glob import glob
from os import makedirs
from os.path import exists, join, basename
from torchvision.ops import nms


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


## VISUALIZATION
import random
classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]
def visualize(image:np.array, box, pred):
    for p, (l, t, r, b) in zip(pred, box):
        class_prob, class_idx = p.max(0)
        cv2.rectangle(image, (l,t), (r,b), colors[class_idx], 2)
        cv2.putText(image, classes[class_idx] + f':{class_prob:.4f}', (l,t), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[class_idx], 2)


def detect(args):
    """
        Draw bounding boxes on images
    """
    # Parse input image size, device settings
    pattern = re.compile(r'h([\d]+)_w([\d]+)')
    device = basename(args.model).split('_')[0]
    if device == 'half':
        device = 'cuda'
        half = True
    else:
        half = False
    img_size = *map(int, pattern.search(args.model).groups()),
    # Make output folder if necessary
    if not exists(args.output): makedirs(out)
    image_extensions = '.jpg', '.jpeg', '.bmp', '.png'
    # Load model
    model = torch.jit.load(args.model)
    model.to(device).eval()
    # Inference with float16
    if half:
        model.half()

    # Run inference
    for path in glob(join(args.source, '*')):
        if not path.endswith(image_extensions): continue
        image = cv2.imread(path)
        img, new_shape = preprocess(image, img_size, device, torch.float16 if half else torch.float32)
        box, pred = infer(model, img, args.conf_thres, args.nms_thres)
        box = postprocess(box, new_shape)
        visualize(image, box, pred)
        cv2.imwrite(join(args.output, basename(path)), image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='weights/half_h320_w192.pt', help='path to jit model file')
    parser.add_argument('--source', type=str, default='data/samples', help='path to image folder')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    detect(parser.parse_args())
