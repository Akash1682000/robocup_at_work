#!/usr/bin/env python3
"""Train YOLOv8n on the dataset from gen_dataset.py.

Run with the venv python (ultralytics needs a newer opencv):
    ../../yolo_env/bin/python scripts/train_yolo.py

imgsz 640 keeps the small Bearing visible, batch 8 fits a 4 GB GPU.
"""
import argparse
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PKG, 'dataset', 'data.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--model', default='yolov8n.pt')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--device', default='0')
    ap.add_argument('--name', default='robocup')
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit('no %s - run gen_dataset.py first (it writes data.yaml at the '
                 'END of the run, so an interrupted run will not have one)'
                 % args.data)

    import torch
    from ultralytics import YOLO

    print('torch %s, cuda available: %s' % (torch.__version__,
                                            torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('device: %s, %.1f GB'
              % (torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9))
    else:
        print('no GPU visible, falling back to CPU - this will be slow')
        args.device = 'cpu'

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        project=os.path.join(PKG, 'runs'),
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print('\nmAP50-95 %.4f   mAP50 %.4f' % (metrics.box.map, metrics.box.map50))
    print('per class:')
    for i, name in enumerate(model.names.values()):
        print('  %-10s mAP50 %.4f' % (name, metrics.box.ap50[i]))
    print('\nweights: %s' % os.path.join(PKG, 'runs', args.name, 'weights', 'best.pt'))


if __name__ == '__main__':
    main()
