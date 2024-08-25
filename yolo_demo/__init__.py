from samsara import Variable
from .build import build_yolov1
import numpy as np


def build_model(
        args,
        model_cfg,
        num_classes=80,
        trainable=False,
        deploy=False
):
    model, criterion = build_yolov1(
        args, model_cfg, num_classes, trainable, deploy)

    if trainable:
        # Load pretrained weight
        if args.pretrained is not None:
            print('Loading pretrained weight ...')
            checkpoint = np.load(args.pretrained, allow_pickle=True).item()
            checkpoint_state_dict = checkpoint.pop("model")
            model_state_dict = model.state_dict()
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = model_state_dict[k].shape
                    shape_checkpoint = checkpoint_state_dict[k].shape
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict)

        # keep training
        if args.resume is not None:
            print('keep training: ', args.resume)
            checkpoint = np.load(args.resume, allow_pickle=True).item()
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)
            del checkpoint, checkpoint_state_dict

        return model, criterion

    else:
        return model