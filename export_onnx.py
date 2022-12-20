import argparse
from loftr import LoFTR, default_cfg
import torch
import torch.nn.utils.prune as prune
import onnx
from utils import make_student_config

def main():
    parser = argparse.ArgumentParser(description='LoFTR demo.')
    parser.add_argument('--out_file', type=str, default='weights/LoFTR_teacher.onnx',
                        help='Path for the output ONNX model.')
    parser.add_argument('--weights', type=str, default='weights/LoFTR_teacher.pt',  # weights/outdoor_ds.ckpt
                        help='Path to network weights.')
    parser.add_argument('--original', action='store_true',
                        help='If specified the original LoFTR model will be used.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
    parser.add_argument('--prune', default=False, help='Do unstructured pruning')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    if opt.original:
        model_cfg = default_cfg
    else:
        model_cfg = make_student_config(default_cfg)

    print('Loading pre-trained network...')
    model = LoFTR(config=model_cfg)
    checkpoint = torch.load(opt.weights)
    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
        missed_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missed_keys) > 0:
            print('Checkpoint is broken')
            return 1
        print('Successfully loaded pre-trained weights.')
    else:
        print('Failed to load checkpoint')
        return 1

    if opt.prune:
        print('Model pruning')
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.5)
                prune.remove(module, 'weight')
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.5)
                prune.remove(module, 'weight')
        weight_total_sum = 0
        weight_total_num = 0
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                weight_total_sum += torch.sum(module.weight == 0)
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                weight_total_num += module.weight.nelement()

        print(f'Global sparsity: {100. * weight_total_sum / weight_total_num:.2f}')

    print(f'Moving model to device: {device}')
    model = model.eval().to(device=device)
    
    img_size = (model_cfg['input_width'], model_cfg['input_height'])
    input_names = ['image0', "image1"]
    output_names = ["conf_matrix", "sim_matrix"]
    with torch.no_grad():
        dummy_image = torch.randn(1, 1, img_size[1], img_size[0], device=device)
        torch.onnx.export(model, 
                          (dummy_image, dummy_image), 
                          opt.out_file, 
                          verbose=False, 
                          opset_version=12,
                          input_names=input_names,
                          output_names=output_names)

    model = onnx.load(opt.out_file)
    onnx.checker.check_model(model)


if __name__ == "__main__":
    main()
