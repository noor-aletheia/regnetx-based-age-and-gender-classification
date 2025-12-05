# RegNetX Age and Gender Classification Quantization

## Model Source
This example uses a custom trained RegNetX-800M model for dual-head age and gender classification.
The ONNX model is generated from PyTorch training pipeline in this repository.

## Script Usage

### Basic Quantization (INT8)
```bash
python test.py \
    --onnx /app/outputs/models/regnet800_utk_finetune/regnet_800m_final.onnx \
    --name regnet_800m_utk \
    --platform rk3576 \
    --do_quant \
    --rknn ./rknn_models \
    --datasets /app/quantize/dataset.txt \
    --labels /app/quantize/labels.txt
```

### FP16 Quantization (No Quantization)
```bash
python test.py \
    --onnx /app/outputs/models/regnet800_utk_finetune/regnet_800m_final.onnx \
    --name regnet_800m_utk \
    --platform rk3576 \
    --no_quant \
    --rknn ./rknn_models
```

### With Cosine Similarity Analysis
```bash
python test.py \
    --onnx /app/outputs/models/regnet800_utk_finetune/regnet_800m_final.onnx \
    --name regnet_800m_utk \
    --platform rk3576 \
    --do_quant \
    --rknn ./rknn_models \
    --accuracy_image /app/quantize/img_00001.jpg
```

### Docker Usage
```bash
docker run --rm -it \
  -v /path/to/regnetx:/app \
  -v /path/to/output:/app/rknn_models \
  aletheiaai/deployments:rknntoolkit_v2.3.2 \
  python /app/quantize/test.py \
  --onnx /app/outputs/models/regnet800_utk_finetune/regnet_800m_final.onnx \
  --name regnet_800m_utk \
  --platform rk3576 \
  --do_quant \
  --rknn /app/rknn_models
```

### Alternative: RKNN Convert CLI
```bash
python3 -m rknn.api.rknn_convert -t rk3576 -i ./model_config.yml -o ./rknn_models
```

### Parameters Description
- `--onnx`: Path to ONNX model file
- `--name`: Output model name prefix
- `--platform`: Target platform (rk3576, rk3588, etc.)
- `--do_quant/--no_quant`: Enable/disable INT8 quantization (default: enabled)
- `--rknn`: Output directory for RKNN models
- `--datasets`: Text file listing calibration images for quantization
- `--labels`: Age classification labels file
- `--accuracy_image`: Single image for cosine similarity analysis
- `--img_size`: Input image size [height, width] (default: [224, 224])
- `--mean_values`: Normalization mean values (default: [0.5, 0.5, 0.5])
- `--std_values`: Normalization std values (default: [0.5, 0.5, 0.5])

## Expected Results

### Quantization Output
The script will generate RKNN models in the specified output directory:
- **FP16**: `regnet_800m_utk-224-224_rk3576_fp16.rknn` (~19MB)
- **INT8**: `regnet_800m_utk-224-224_rk3576_int8.rknn` (~12MB)

### Cosine Similarity Analysis
When accuracy analysis is enabled, you'll see layer-by-layer cosine similarity:
```
layer_name                     simulator_error                    
                           entire        single             
                        cos    euc     cos    euc          
----------------------------------------------------------------
[Conv] age_output_mm       0.99870 | 3.9418   0.99998 | 0.3413        
[Conv] gender_output_mm    0.99866 | 3.5479   0.99999 | 0.2288
```

### Inference Results
The script will show dual-head predictions for age and gender:
```
Image: img_00001.jpg
-----Head 0 TOP 4-----
[  1] score:22.949987 class:"10-29"
[  2] score:13.969558 class:"30-49"
[  0] score:-17.960859 class:"0-9"
[  3] score:-54.880405 class:"50-70+"
-----Head 1 TOP 2-----
[  1] score:4.791762 class:"male"
[  0] score:-25.023647 class:"female"
Speed: 42.49 ms/image | 23.53 images/sec
```

## Files Required
- `dataset.txt`: List of calibration images (1000 images recommended)
- `labels.txt`: Age class labels
- `gender_labels.txt`: Gender class labels  
- `model_config.yml`: Configuration for rknn_convert CLI
- Input ONNX model from training pipeline

**Note**: Different platforms, different versions of tools and drivers may have slightly different results.