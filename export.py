import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import cv2

import onnx
import onnxruntime as ort
import onnxsim
# ========================
# RegNetX Model (replaces CBAM model)
# ========================
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models import DualHeadRegNetX
from regnet import get_regnet, REGNET_MODEL_ZOO



class UniversalModelWrapper(nn.Module):
    def __init__(self, backbone, target_size=(224, 224)):
        super().__init__()
        self.backbone = backbone.eval()
        self.target_size = target_size
        self.register_buffer('norm_mean', torch.tensor([0.498, 0.498, 0.498], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.498, 0.498, 0.498], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('scale_factor', torch.tensor(255.0, dtype=torch.float32))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)  # HWC -> NHWC
        
        normalized_x = x / self.scale_factor
        rgb_x = normalized_x[..., [2, 1, 0]]  # BGR -> RGB
        nchw_x = rgb_x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        normalized = (nchw_x - self.norm_mean) / self.norm_std
        outputs = self.backbone(normalized)
        return outputs['age'], outputs['gender']


# -------------------------
# Export + Compare
# -------------------------
def export_and_compare(checkpoint_path, onnx_path, img_path, num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    age_final_keys = [k for k in state_dict.keys() if 'age_head' in k and 'weight' in k and k.endswith('.9.weight')]
    gender_final_keys = [k for k in state_dict.keys() if 'gender_head' in k and 'weight' in k and k.endswith('.9.weight')]
    age_classes = state_dict[age_final_keys[0]].shape[0] if age_final_keys else 4
    gender_classes = state_dict[gender_final_keys[0]].shape[0] if gender_final_keys else 2
    
    print(f"Detected classes: Age={age_classes}, Gender={gender_classes}")
    
    import os
    ckpt_base = os.path.basename(checkpoint_path)
    import re
    match = re.search(r'(regnet_\d+m)', ckpt_base)
    if match:
        model_name = match.group(1)
    print(f"Using model_name: {model_name}")

    model = DualHeadRegNetX(
        model_name=model_name,
        num_age_classes=age_classes,
        num_gender_classes=gender_classes,
        pretrained=False
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    
    img_width = 224
    img_height = 224

    ########################################################### torch inference #############################
    # create a random np array of 224x224x3
    original_img = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8).astype(np.float32)
    img = original_img / 255.0
    img = img[..., [2, 1, 0]]
    mean = np.array([0.498, 0.498, 0.498], dtype=np.float32)
    std  = np.array([0.498, 0.498, 0.498], dtype=np.float32)
    img = (img - mean) / std

    transpose_img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW
    input_batch = transpose_img.unsqueeze(0).to(device)  # BCHW
    torch_out = model(input_batch)
    
    if isinstance(torch_out, dict):
        age_out = torch_out['age'].detach().cpu().numpy()
        gender_out = torch_out['gender'].detach().cpu().numpy()
        print("Torch out - Age:", age_out)
        print("Torch out - Gender:", gender_out)
    else:
        print("Torch out:", torch_out.detach().cpu().numpy())

    ########################################################### onnx ###########################################

    # wrapped_model = UniversalModelWrapper(model, target_size=(img_height, img_width)).to(device).eval()

    tensor = torch.from_numpy(original_img).unsqueeze(0).to(device)  # [1,H,W,C], float32

    # wrapped_model_out = wrapped_model(tensor)
    # tensor = wrapped_model_out[2]  # get normalized tensor output
    print("shape of input_batch:", input_batch.shape)
    # print("shape of tensor:", tensor.shape)
    print("input_batch values (flattened, first 10):", input_batch.cpu().numpy().flatten()[:10])
    # print("tensor values (flattened, first 10):", tensor.cpu().numpy().flatten()[:10])

    # diff = input_batch.cpu().numpy() - tensor.cpu().numpy()
    # print("Mean absolute difference:", np.abs(diff).mean())
    # print("Max absolute difference:", np.abs(diff).max())

    # sim = np.dot(input_batch.cpu().numpy().flatten(), tensor.cpu().numpy().flatten()) / (np.linalg.norm(input_batch.cpu().numpy().flatten()) * np.linalg.norm(tensor.cpu().numpy().flatten()))
    # print("Cosine similarity between input_batch and tensor:", sim)
    
    # Handle dual-head output
    # if isinstance(wrapped_model_out, tuple) and len(wrapped_model_out) == 2:
    #     age_out, gender_out = wrapped_model_out
    #     print("wrapped_model_out - Age:", age_out.detach().cpu().numpy())
    #     print("wrapped_model_out - Gender:", gender_out.detach().cpu().numpy())
    # else:
    #     print("wrapped_model_out :", wrapped_model_out.detach().cpu().numpy() if hasattr(wrapped_model_out, 'detach') else wrapped_model_out)

    #########################################################################################################

    dummy_input = torch.from_numpy(original_img).unsqueeze(0).to(device)  # Use actual image size
    
    torch.onnx.export(
        # wrapped_model,
        model,
        # dummy_input,
        input_batch,
        onnx_path,
        input_names=["input"],
        output_names=["age_output", "gender_output"],
        # dynamic_axes={
        #     "input": {0: "batch"}, 
        #     "age_output": {0: "batch"},
        #     "gender_output": {0: "batch"},
        #     },
        opset_version=18
    )
    print(f"ONNX model saved to {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    try:
        model_simplified, check = onnxsim.simplify(onnx_model)
        if not check:
            print("[WARNING] ONNX simplification did not pass all checks. Using unsimplified model.")
        else:
            onnx.save(model_simplified, onnx_path)
            print("ONNX model simplified and saved.")
    except Exception as e:
        print(f"[WARNING] ONNX simplification failed: {e}\nThe exported ONNX model is still usable, but may not be simplified.")

    
    ############################################### onnx inference ###########################################

    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # img = cv2.imread(img_path)  # BGR - raw image, no preprocessing
    # No resize needed - the ONNX model (UniversalModelWrapper) handles all preprocessing internally
    # ort_inputs = {"input": original_img.astype(np.float32)[np.newaxis, ...]}  # NHWC float32, raw image
    ort_inputs = {"input": input_batch.cpu().numpy().astype(np.float32)}
    ort_outputs = ort_sess.run(["age_output", "gender_output"], ort_inputs)
    age_onnx_out, gender_onnx_out = ort_outputs
    print("ONNX out - Age:", age_onnx_out)
    print("ONNX out - Gender:", gender_onnx_out)

    onnx_input_tensor = ort_inputs["input"]  # This is what ONNX receives

# Compare with PyTorch input_batch
    pt_tensor = input_batch.cpu().numpy()

    print("PyTorch input_batch shape:", pt_tensor.shape)
    print("ONNX input tensor shape:", onnx_input_tensor.shape)
    print("First 10 values (PyTorch):", pt_tensor.flatten()[:10])
    print("First 10 values (ONNX):", onnx_input_tensor.flatten()[:10])

# Compute difference and cosine similarity
    diff = pt_tensor - onnx_input_tensor
    print("Mean absolute difference:", np.abs(diff).mean())
    print("Max absolute difference:", np.abs(diff).max())

    cos_sim = np.dot(pt_tensor.flatten(), onnx_input_tensor.flatten()) / (
        np.linalg.norm(pt_tensor.flatten()) * np.linalg.norm(onnx_input_tensor.flatten())
    )
    print("Cosine similarity between PyTorch input_batch and ONNX input tensor:", cos_sim)

    # print("ONNX out - Normalized:", onnx_normalized.shape)
    # print("onnx flattened, first 10):", onnx_normalized.flatten()[:10])

    # cos_sim = np.dot(input_batch.cpu().numpy().flatten(), onnx_normalized.flatten()) / (
    #     np.linalg.norm(input_batch.cpu().numpy().flatten()) * np.linalg.norm(onnx_normalized.flatten())
    # )
    # print("Cosine similarity between PyTorch input_batch and ONNX normalized tensor:", cos_sim)

if __name__ == "__main__":
    export_and_compare(
        checkpoint_path="/app/outputs/models/scratch_2/regnet_1600m_best.pth",
        onnx_path="/app/outputs/models/scratch_2/regnet_1600m.onnx",
        img_path="/app/batbox_dataset/img_00001.jpg",   # <-- test image
        num_classes=4  # This parameter is now auto-detected from model
    )
