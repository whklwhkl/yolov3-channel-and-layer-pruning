def get_args():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('cfg', help='destination model cfg')
    ap.add_argument('src', help='source model weights, pytorch state dict')
    ap.add_argument('dst', help='destination model weights')
    ap.add_argument('class_idx', help='seperate with comma, removing the rest;-1 means delete class predictors')
    return ap.parse_args()


def main(args):
    import torch
    from os.path import join, dirname
    from models import Darknet, save_weights
    model = Darknet(args.cfg)
    na = model.module_list[model.yolo_layers[0]].na  # anchor number per point
    sd0 = model.state_dict()
    sd1 = torch.load(args.src)['model']
    idx = list(range(5))
    if args.class_idx != '-1':
        idx += [int(x) + 5 for x in args.class_idx.split(',')]  # box + obj + cls
    for k in sd0:
        if sd0[k].shape != sd1[k].shape:
            sd1[k] = torch.cat([x[idx] for x in sd1[k].chunk(na)])
    model.load_state_dict(sd1)
    save_weights(model, args.dst)


if __name__ == '__main__':
    main(get_args())
