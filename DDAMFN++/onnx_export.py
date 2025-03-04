import argparse
import torch
from networks.DDAM import DDAMNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--workers", default=8, type=int, help="Number of data loading workers."
    )
    parser.add_argument(
        "--num_head", type=int, default=2, help="Number of attention head."
    )
    parser.add_argument("--num_class", type=int, default=8, help="Number of class.")
    parser.add_argument(
        "--model_path", default="./checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.pth"
    )
    parser.add_argument(
        "--output_path", default="./checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.onnx"
    )
    return parser.parse_args()


def export_onnx():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Define input example
    dummy_input = torch.randn(1, 3, 112, 112, device=device)

    # Perform inference to capture dynamic computation graph
    with torch.no_grad():
        output, _, _ = model(dummy_input)

    # Export the model to ONNX
    torch.onnx.export(
        model, dummy_input, args.output_path, verbose=True, opset_version=10
    )

    print(f"ONNX model exported to {args.output_path}")


if __name__ == "__main__":
    export_onnx()
