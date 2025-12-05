import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import cv2
import onnx
import onnxsim
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models import DualHeadRegNetX

class FixedBatchNorm(nn.Module):
    """Replaces BatchNorm with equivalent fixed computation using running statistics"""
    def __init__(self, bn_layer):
        super().__init__()
        self.register_buffer('weight', bn_layer.weight.data)
        self.register_buffer('bias', bn_layer.bias.data)
        self.register_buffer('running_mean', bn_layer.running_mean.data)
        self.register_buffer('running_var', bn_layer.running_var.data)
        self.eps = bn_layer.eps
    
    def forward(self, x):
        # Apply batch norm using running statistics (equivalent to eval mode)
        return (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

def replace_batchnorm_for_onnx(module):
    """Replace all BatchNorm2d layers with FixedBatchNorm for ONNX compatibility"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FixedBatchNorm(child))
        else:
            replace_batchnorm_for_onnx(child)



class UniversalModelWrapper(nn.Module):
    def __init__(self, backbone, target_size=(224, 224)):
        super().__init__()
        self.backbone = backbone.eval()
        self.target_size = target_size
        self.register_buffer('norm_mean', torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('scale_factor', torch.tensor(255.0, dtype=torch.float32))

    def forward(self, x):
        # Input: [B, H, W, C] BGR image (can be uint8 0-255 or float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # HWC -> NHWC
        
        # Convert to float32 FIRST before any operations
        if x.dtype != torch.float32:
            x = x.float()
        
        # Always resize to target size - ensure float32 before interpolation
        # Convert to NCHW for interpolation
        temp_x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW  
        # Ensure we're working with float32 for interpolation
        if temp_x.dtype != torch.float32:
            temp_x = temp_x.float()
        temp_x = F.interpolate(temp_x, size=self.target_size, mode='bilinear', align_corners=False)
        x = temp_x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        # Normalize to [0,1] range
        normalized_x = x / self.scale_factor
        
        # Convert BGR -> RGB
        rgb_x = normalized_x[..., [2, 1, 0]]  # BGR -> RGB
        
        # Convert to NCHW format for model
        nchw_x = rgb_x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Apply normalization (same as training)
        normalized = (nchw_x - self.norm_mean) / self.norm_std
        
        # Forward through backbone
        outputs = self.backbone(normalized)
        return outputs['age'], outputs['gender']


# -------------------------
# Export + Compare
# -------------------------
def export_and_compare(checkpoint_path, onnx_path, img_path, use_wrapper=True, 
                      dynamic_batch=True, dynamic_size=False, static_export=False,
                      simplify_model=True, verify_model=True, opset_version=17):
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
    else:
        # Fallback to a default model if regex fails
        model_name = 'regnet_1600m'
        print(f"Warning: Could not detect model name from {ckpt_base}, using default: {model_name}")
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
    
    # Fix BatchNorm layers for proper ONNX export
    print("Replacing BatchNorm layers with fixed computations for ONNX compatibility...")
    replace_batchnorm_for_onnx(model)

    
    img_width = 224
    img_height = 224

    ########################################################### torch inference #############################
    # Load actual image and preprocess exactly like training
    img = cv2.imread(img_path)
    if img is None:
        original_img = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
        print("Using random image (no image found)")
    else:
        original_img = cv2.resize(img, (img_width, img_height))
        print("Using real image")
    
    # Convert BGR -> RGB (like PIL Image.open() which ToTensor() expects)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Apply same preprocessing as training: ToTensor() + Normalize()
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # ToTensor() equivalent
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std  # Normalize() equivalent
    
    input_batch = img_normalized.unsqueeze(0).to(device)  # BCHW
    torch_out = model(input_batch)
    
    if isinstance(torch_out, dict):
        age_out = torch_out['age'].detach().cpu().numpy()
        gender_out = torch_out['gender'].detach().cpu().numpy()
        print("Torch out - Age:", age_out)
        print("Torch out - Gender:", gender_out)
    else:
        print("Torch out:", torch_out.detach().cpu().numpy())

    ########################################################### onnx ###########################################
    
    # Determine which model to export and what input to use
    if use_wrapper:
        print("Using wrapper model with preprocessing...")
        # Create wrapper model with preprocessing
        export_model = UniversalModelWrapper(model, target_size=(img_height, img_width)).to(device).eval()
        # Test wrapper with BGR image (NHWC format) - this is what the final ONNX model will expect
        export_tensor = torch.from_numpy(original_img).unsqueeze(0).to(device)  # [1,H,W,C], uint8 BGR image
        
        with torch.no_grad():
            wrapped_model_out = export_model(export_tensor)
        
        # Handle dual-head output and validate wrapper works
        if isinstance(wrapped_model_out, tuple) and len(wrapped_model_out) == 2:
            age_out, gender_out = wrapped_model_out
            print("wrapped_model_out - Age:", age_out.detach().cpu().numpy())
            print("wrapped_model_out - Gender:", gender_out.detach().cpu().numpy())
        else:
            print("wrapped_model_out:", wrapped_model_out.detach().cpu().numpy() if hasattr(wrapped_model_out, 'detach') else wrapped_model_out)
        
        # Verify wrapper output matches PyTorch output (they should be very close)
        if isinstance(wrapped_model_out, tuple) and len(wrapped_model_out) == 2:
            age_diff = np.abs(torch_out['age'].detach().cpu().numpy() - wrapped_model_out[0].detach().cpu().numpy())
            gender_diff = np.abs(torch_out['gender'].detach().cpu().numpy() - wrapped_model_out[1].detach().cpu().numpy())
            print(f"Wrapper vs PyTorch - Age max diff: {age_diff.max():.6f}, Gender max diff: {gender_diff.max():.6f}")
        
        input_description = "raw BGR image [B,H,W,C] uint8 0-255"
        
    else:
        print("Using base model without preprocessing wrapper...")
        export_model = model
        export_tensor = input_batch  # Pre-normalized tensor [B,C,H,W] float32
        wrapped_model_out = None
        input_description = "pre-normalized tensor [B,C,H,W] float32"
    
    #########################################################################################################
    
    # Determine dynamic axes based on options
    if static_export:
        print("Exporting with static dimensions...")
        dynamic_axes = None
    else:
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes["input"] = {0: "batch_size"}
            dynamic_axes["age_output"] = {0: "batch_size"}
            dynamic_axes["gender_output"] = {0: "batch_size"}
        
        if dynamic_size and use_wrapper:
            # Add height and width dimensions for wrapper (input format is [B,H,W,C])
            if "input" not in dynamic_axes:
                dynamic_axes["input"] = {}
            dynamic_axes["input"][1] = "height"
            dynamic_axes["input"][2] = "width"
    
    # Export the model
    export_desc = f"{'wrapper' if use_wrapper else 'base'} model"
    if dynamic_axes:
        dynamic_desc = ", ".join([f"{k}: {v}" for k, v in dynamic_axes.items()])
        print(f"Exporting {export_desc} with dynamic axes: {dynamic_desc}")
    else:
        print(f"Exporting {export_desc} with static dimensions")
    
    print(f"Input format: {input_description}")
    
    torch.onnx.export(
        export_model,
        export_tensor,
        onnx_path,
        input_names=["input"],
        output_names=["age_output", "gender_output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version
    )
    print(f"ONNX model saved to {onnx_path}")
    
    # Simplify model (always enabled)
    if simplify_model:
        print("Simplifying ONNX model...")
        try:
            onnx_model = onnx.load(onnx_path)
            model_simplified, check = onnxsim.simplify(onnx_model)
            if not check:
                print("[WARNING] ONNX simplification did not pass all checks. Using unsimplified model.")
            else:
                onnx.save(model_simplified, onnx_path)
                print("ONNX model simplified and saved.")
        except Exception as e:
            print(f"[WARNING] ONNX simplification failed: {e}\nThe exported ONNX model is still usable, but may not be simplified.")

    # ONNX model verification
    if verify_model:
        ############################################### onnx inference ###########################################

        ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Use appropriate input format based on export type
        if use_wrapper:
            # Raw BGR image as input (same as wrapper expects) - keep as uint8 as ONNX expects
            ort_inputs = {"input": original_img[np.newaxis, ...].astype(np.uint8)}  # [1,H,W,C] BGR image
        else:
            # Pre-normalized tensor as input (same as base model expects) - float32
            ort_inputs = {"input": input_batch.cpu().numpy()}  # [1,C,H,W] normalized float32
        
        ort_outputs = ort_sess.run(["age_output", "gender_output"], ort_inputs)
        age_onnx_out, gender_onnx_out = ort_outputs
        
        print("ONNX out - Age:", age_onnx_out)
        print("ONNX out - Gender:", gender_onnx_out)

        # Compare PyTorch vs ONNX inputs and outputs 
        pt_wrapped_tensor = export_tensor.cpu().numpy()  # Input tensor used for export
        onnx_tensor = ort_inputs["input"]

        print("PyTorch wrapper input shape:", pt_wrapped_tensor.shape)
        print("ONNX input shape:", onnx_tensor.shape)
        print("First 10 values (PyTorch wrapper):", pt_wrapped_tensor.flatten()[:10])
        print("First 10 values (ONNX):", onnx_tensor.flatten()[:10])

        # Compute difference and cosine similarity for inputs
        input_diff = pt_wrapped_tensor - onnx_tensor
        print("Input Mean absolute difference:", np.abs(input_diff).mean())
        print("Input Max absolute difference:", np.abs(input_diff).max())

        input_cos_sim = np.dot(pt_wrapped_tensor.flatten(), onnx_tensor.flatten()) / (
            np.linalg.norm(pt_wrapped_tensor.flatten()) * np.linalg.norm(onnx_tensor.flatten())
        )
        print("Input cosine similarity:", input_cos_sim)
        
        # Compare outputs 
        def cosine_similarity(a, b):
            a_flat = a.flatten()
            b_flat = b.flatten()
            return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        
        if use_wrapper:
            # Compare wrapper outputs with ONNX
            if isinstance(wrapped_model_out, tuple) and len(wrapped_model_out) == 2:
                age_cos_sim = cosine_similarity(wrapped_model_out[0].detach().cpu().numpy(), age_onnx_out)
                gender_cos_sim = cosine_similarity(wrapped_model_out[1].detach().cpu().numpy(), gender_onnx_out)
                
                print(f"Age output cosine similarity (Wrapper vs ONNX): {age_cos_sim:.6f}")
                print(f"Gender output cosine similarity (Wrapper vs ONNX): {gender_cos_sim:.6f}")
                
                # Also compare base PyTorch model with ONNX for reference
                base_age_cos_sim = cosine_similarity(torch_out['age'].detach().cpu().numpy(), age_onnx_out)
                base_gender_cos_sim = cosine_similarity(torch_out['gender'].detach().cpu().numpy(), gender_onnx_out)
                
                print(f"Age output cosine similarity (Base PyTorch vs ONNX): {base_age_cos_sim:.6f}")
                print(f"Gender output cosine similarity (Base PyTorch vs ONNX): {base_gender_cos_sim:.6f}")
            else:
                print("Warning: Could not compare wrapper outputs - output format unexpected")
        else:
            # Compare base model outputs directly with ONNX
            base_age_cos_sim = cosine_similarity(torch_out['age'].detach().cpu().numpy(), age_onnx_out)
            base_gender_cos_sim = cosine_similarity(torch_out['gender'].detach().cpu().numpy(), gender_onnx_out)
            
            print(f"Age output cosine similarity (PyTorch vs ONNX): {base_age_cos_sim:.6f}")
            print(f"Gender output cosine similarity (PyTorch vs ONNX): {base_gender_cos_sim:.6f}")
    else:
        print("Skipping verification (--no-verify flag used)")
        print("ONNX model exported successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export RegNet model to ONNX with flexible options')
    parser.add_argument('--checkpoint', required=True, help='Path to PyTorch checkpoint file')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--test-image', default="/app/mixed_data/test/drama_person_024_img_1.jpg", help='Test image path for verification')
    
    # Wrapper options
    parser.add_argument('--no-wrapper', action='store_true', help='Export model without preprocessing wrapper (requires pre-normalized input)')
    parser.add_argument('--wrapper', action='store_true', help='Export model with preprocessing wrapper (accepts raw BGR images)')
    
    # Dynamic axis options
    parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch size')
    parser.add_argument('--dynamic-size', action='store_true', help='Enable dynamic image size (only with wrapper)')
    parser.add_argument('--static', action='store_true', help='Export with completely static dimensions')
    
    # Additional options
    parser.add_argument('--no-verify', action='store_true', help='Skip verification against PyTorch model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (default: 17)')
    
    args = parser.parse_args()
    
    # Set defaults if no wrapper option specified
    if not args.no_wrapper and not args.wrapper:
        args.wrapper = True  # Default to wrapper
        print("No wrapper option specified, defaulting to --wrapper")
    
    # Set defaults if no dynamic option specified
    if not args.dynamic_batch and not args.dynamic_size and not args.static:
        args.dynamic_batch = True  # Default to dynamic batch
        print("No dynamic option specified, defaulting to --dynamic-batch")
    
    # Validate combinations
    if args.no_wrapper and args.dynamic_size:
        print("Warning: --dynamic-size only works with wrapper, ignoring dynamic-size")
        args.dynamic_size = False
    
    if args.no_wrapper and args.wrapper:
        parser.error("Cannot specify both --no-wrapper and --wrapper")
    
    if args.static and (args.dynamic_batch or args.dynamic_size):
        parser.error("Cannot specify --static with dynamic options")
    
    export_and_compare(
        checkpoint_path=args.checkpoint,
        onnx_path=args.output,
        img_path=args.test_image,
        use_wrapper=args.wrapper,
        dynamic_batch=args.dynamic_batch,
        dynamic_size=args.dynamic_size,
        static_export=args.static,
        simplify_model=True,  # Always simplify ONNX models
        verify_model=not args.no_verify,
        opset_version=args.opset
    )
