import argparse
import torch
import os
from apex.contrib.sparsity import ASP

import torch.onnx as onnx

from models import resnet, vgg
from utils.pruner import get_prune_mask, applyMask


def get_args():
    parser = argparse.ArgumentParser(description='Convert a model to ONNX format')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output ONNX file')
    parser.add_argument('--prune-rate', type=float, default=0.0, help='Prune rate for the model')
    parser.add_argument('--model-arch', type=str, default="resnet", help='Model architecture')
    parser.add_argument('--model-depth', type=int, default=32, help='Model depth')
    parser.add_argument('--model-num-classes', type=int, default=10, help='Model classes that are predicted')

    args = parser.parse_args()

    assert os.path.exists(args.model), "Model file does not exist"
    assert args.prune_rate >= 0.0 and args.prune_rate <= 1.0, "Prune rate must be between 0 and 1"
    assert args.model_arch in ["resnet", "vgg"], "Model architecture must be either resnet or vgg"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    return args


def load_model(
        model_path,
        model_arch,
        model_depth,
        num_classes,
):
    if model_arch == "resnet":
        model = resnet(
            num_classes=num_classes,
            depth=model_depth,
            planes=[32, 64, 128],
        )
    elif model_arch == "vgg":
        model = vgg.__dict__[model_arch + str(model_depth) + "_bn"](
            num_classes=num_classes
        )
    else:
        assert False

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model


def convert_to_onnx(model, output_path,device):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Beispiel für die Eingabe
    onnx.export(model, dummy_input, output_path)


def prune_model(model, prune_rate):
    mask = get_prune_mask(model, prune_rate)
    model, keep_ratio = applyMask(model, mask)
    print(f"Keep ratio: {keep_ratio}")
    return model

def ASP_prune_model(model):
    optimizer_sparse = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    ASP.prune_trained_model(model,optimizer_sparse)
    return model



def main():
    args = get_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        args.model,
        args.model_arch,
        args.model_depth,
        args.model_num_classes
    ).to(device)

    # Save pseudo-pruned model with values close to zero
    output_path = os.path.join(
        args.output,
        "pseudo-pruned_" + os.path.basename(args.model) + ".onnx"
    )
    convert_to_onnx(model, output_path,device)

    # Prune Values down to zero
    pruned_model = prune_model(model, args.prune_rate)

    # Save pruned model
    output_path = os.path.join(
        args.output,
        "pruned_" + os.path.basename(args.model) + ".onnx"
    )
    
    convert_to_onnx(pruned_model, output_path,device)

    ASP_pruned_model = ASP_prune_model(model)
    # Save pruned model
    output_path = os.path.join(
        args.output,
        "ASP_pruned_" + os.path.basename(args.model) + ".onnx"
    )
    convert_to_onnx(ASP_pruned_model, output_path,device)



if __name__ == '__main__':
    main()