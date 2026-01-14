# scripts/check_inference.py
import os
import json
import torch
import pickle
from torchvision import transforms
from PIL import Image
import argparse

def load_classes(classes_path):
    with open(classes_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def load_model(model_path, device):
    # Assumes a resnet18-like model with fc attribute
    from torchvision import models
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Try to load a PyTorch state_dict; if the file is a text placeholder (e.g. "No model trained for Orange...")
    # handle gracefully by creating a dummy model that predicts the single class.
    try:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        # Common failure: UnpicklingError because the .pth is not a binary torch file
        print(f"Warning: torch.load failed for '{model_path}': {e}")
        try:
            with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(1024)
        except Exception:
            sample = ''

        # If the file contains an explanatory placeholder, create a dummy model
        if 'No model' in sample or 'No best model' in sample or 'Single class' in sample:
            print(f"Detected placeholder model file. Creating dummy model for '{model_path}'.")
            # We don't know classes here; caller should pass classes list and verify sizes.
            # Create a dummy model that strongly predicts class 0 for any input.
            class DummyModel(torch.nn.Module):
                def __init__(self, num_classes=1):
                    super().__init__()
                    self.num_classes = num_classes
                    # simple linear just to expose fc.out_features
                    self.fc = torch.nn.Linear(1, num_classes)

                def forward(self, x):
                    batch = x.shape[0]
                    logits = torch.full((batch, self.num_classes), -10.0, device=x.device)
                    logits[:, 0] = 10.0
                    return logits

            dummy = DummyModel(num_classes=1).to(device).eval()
            return dummy
        else:
            raise

    # If state is loaded, try to infer number of classes from fc.weight
    fc_weight = None
    if isinstance(state, dict):
        for k in state.keys():
            if 'fc.weight' in k:
                fc_weight = state[k]
                break
    # If we found fc weights, adjust model.fc
    if fc_weight is not None:
        out_features = fc_weight.shape[0]
        model.fc = torch.nn.Linear(model.fc.in_features, out_features)

    # Try loading state_dict (may raise if shapes don't match)
    try:
        model.load_state_dict(state)
    except Exception as e:
        print(f"Warning: load_state_dict failed: {e}")
        # If loading failed but we inferred out_features, still continue with adjusted fc
    model.to(device).eval()
    return model

def predict(image_path, model, classes, device, topk=3):
    transform = make_transform()
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    topk_idx = probs.argsort()[::-1][:topk]
    return [(int(i), classes[i], float(probs[i])) for i in topk_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=".\\plant_health_app\\notebook\\plant_disease\\models\\best\\pepper_model.pth", help="path to .pth model")
    parser.add_argument("--classes", default=".\\plant_health_app\\notebook\\plant_disease\\group_classes.json", help="path to classes json (list ordered)")
    parser.add_argument("--image", default=".\\media\\disease_images\\bacterial_spot_pepper.JPG", help="path to test image")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    classes_raw = load_classes(args.classes)
    model = load_model(args.model, device)

    # Resolve classes list: the JSON may be a dict (group_classes) mapping plant -> [diseases]
    if isinstance(classes_raw, dict):
        # try to infer plant name from image path or model filename
        plant_name = None
        img_lower = args.image.replace('\\', '/').lower()
        model_lower = args.model.lower()
        for key in classes_raw.keys():
            key_lower = key.lower()
            if key_lower in img_lower or key_lower in model_lower:
                plant_name = key
                break
        if plant_name is None:
            plant_name = next(iter(classes_raw.keys()))
            print(f"Warning: could not infer plant from image/model path; defaulting to '{plant_name}' from classes dict.")
        classes = classes_raw[plant_name]
        print(f"Using classes for plant: {plant_name} (num={len(classes)})")
    else:
        classes = classes_raw

    print("Model out_features (fc):", getattr(model.fc, 'out_features', None))
    print("Num classes (json):", len(classes))
    if isinstance(classes, list):
        print("Classes preview:", classes[:10])
    else:
        print("Classes data is not a list; type:", type(classes))

    preds = predict(args.image, model, classes, device, topk=5)
    print("Top predictions:")
    for idx, name, p in preds:
        print(f"  idx={idx} name={name} prob={p:.4f}")