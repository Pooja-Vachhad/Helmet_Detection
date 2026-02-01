import torch
import onnx
import onnxruntime as ort
import os
import string
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CRNN

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
char_to_int = {char: idx + 1 for idx, char in enumerate(CHARS)}
int_to_char = {idx: char for char, idx in char_to_int.items()}
num_classes = len(CHARS) + 1

# Decode CTC output logits to readable text
def ctc_decode(logits, int_to_char):
    max_probs = torch.argmax(logits, dim=2)
    decoded_strings = []
    for seq in max_probs:
        prev = -1
        decoded = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                decoded.append(int_to_char[idx])
            prev = idx
        decoded_strings.append("".join(decoded))
    return decoded_strings

test_transforms = transforms.Compose([
    transforms.Resize((100, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.42, 0.43, 0.41),
        std=(0.32, 0.32, 0.32)
    )
])

# Export PyTorch model to ONNX format
def export_to_onnx_legacy(model, model_path, onnx_path="crnn_model.onnx", input_shape=(1, 3, 100, 200), device="cuda"):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Export to ONNX using legacy exporter
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False
        )
    print(f"Model exported to {onnx_path}")
    return onnx_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(H=100, W=200, num_classes=num_classes).to(device)

# Export model to ONNX
onnx_model_path = export_to_onnx_legacy(model=model, model_path="best_model.pth", onnx_path="crnn_model.onnx", input_shape=(1, 3, 100, 200), device=device)

# Validate the exported ONNX model
onnx_model = onnx.load("crnn_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Run inference on a test image using the ONNX model
def test_onnx_inference(onnx_path, test_image_path, test_transforms, int_to_char):
    ort_session = ort.InferenceSession(onnx_path)
    image = Image.open(test_image_path).convert("RGB")
    img_tensor = test_transforms(image).unsqueeze(0).cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    output_tensor = torch.tensor(ort_outputs[0])
    pred = ctc_decode(output_tensor, int_to_char)[0]
    print(f"ONNX Prediction: {pred}")
    plt.imshow(image)
    plt.title(f"Prediction: {pred}")
    plt.axis('off')
    plt.show()
    return pred

test_folder = ""
test_image = os.path.join(test_folder, os.listdir(test_folder)[1])
test_onnx_inference("crnn_model.onnx", test_image, test_transforms, int_to_char)