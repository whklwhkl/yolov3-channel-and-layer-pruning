import argparse, cv2

from sys import platform
from models import create_grids
from models import *  # set ONNX_EXPORT in models.py
import models
from torchvision.ops import nms
models.ONNX_EXPORT=True
ONNX_EXPORT=True
from utils.datasets import LoadImages, LoadStreams, LoadWebcam
from utils.utils import *


def trace(save_txt=False, save_img=False):
    img_size = opt.input_height, opt.input_width
    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    if opt.weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(opt.weights, map_location=opt.device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, opt.weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(opt.device).eval()
    if opt.device != 'cpu':
        for i in model.yolo_layers:
            create_grids(model.module_list[i],
                         img_size,
                         tuple(model.module_list[i].ng.int().tolist()),
                         opt.device, torch.float16 if opt.half else torch.float32)
    img = torch.zeros((1, 3) + img_size).to(opt.device)  # (1, 3, 320, 192)

    # Half precision
    if opt.half:
        assert opt.device != 'cpu', 'half precision only supported on CUDA'
        model.half()
        img = img.half()

    # Export model
    rep = torch.jit.trace(model, img)
    torch.jit.save(rep, 'weights/{}_h{}_w{}.pt'.format('half' if opt.half else opt.device, *img_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--input_width', type=int, default=416, help='inference width (pixels)')
    parser.add_argument('--input_height', type=int, default=416, help='inference height (pixels)')
    parser.add_argument('--device', default='cpu', help='device: cuda or cpu')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        trace()
