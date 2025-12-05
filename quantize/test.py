import os
import sys
import cv2
import numpy as np
from rknn.api import RKNN
import argparse
import time
import urllib
import traceback

def show_outputs(outputs, labels_path=None, topn=5):
    """
    Display top-N predictions for each output head, using the correct label file for each head.
    labels_path: list of label file paths (e.g., [age_labels, gender_labels]) or a single path.
    """
    def safe_topn(output, labels, topn, head_name=None):
        n = min(topn, len(labels), len(output))
        if n < topn:
            print(f"[WARN] Output/classes mismatch: output={len(output)}, labels={len(labels)}. Showing top {n}.")
        if head_name:
            print(f'-----{head_name} TOP {n}-----')
        else:
            print(f'-----TOP {n}-----')
        index = sorted(range(len(output)), key=lambda k: output[k], reverse=True)
        for i in range(n):
            value = output[index[i]]
            print('[{:>3d}] score:{:.6f} class:"{}"'.format(index[i], value, labels[index[i]]))

    # Multi-head: use list of label files
    if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        # If labels_path is not a list, make it a list with the same path for all heads
        if not isinstance(labels_path, (list, tuple)):
            labels_path = [labels_path] * len(outputs)
        head_names = [f'Head {i}' for i in range(len(outputs))]
        for idx, out in enumerate(outputs):
            output = np.array(out).flatten()
            label_file = labels_path[idx] if idx < len(labels_path) else None
            if label_file and os.path.exists(label_file):
                with open(label_file, 'r') as fp:
                    labels = [l.strip().split(':')[-1] for l in fp.readlines()]
            else:
                labels = [str(i) for i in range(len(output))]
            safe_topn(output, labels, topn, head_names[idx])
    else:
        # Single head
        output = np.array(outputs[0]).flatten() if isinstance(outputs, (list, tuple)) else np.array(outputs).flatten()
        label_file = labels_path[0] if isinstance(labels_path, (list, tuple)) else labels_path
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as fp:
                labels = [l.strip().split(':')[-1] for l in fp.readlines()]
        else:
            labels = [str(i) for i in range(len(output))]
        safe_topn(output, labels, topn)

def readable_speed(value, mode='inference', count=1):
    """
    If mode == 'inference', value is total time in seconds, count is number of images.
    If mode == 'transfer', value is bytes transferred, count is total seconds.
    """
    if mode == 'inference':
        if count == 0 or value <= 0:
            return "N/A"
        ms_per_image = (value / count) * 1000
        images_per_sec = count / value
        return f"{ms_per_image:.2f} ms/image | {images_per_sec:.2f} images/sec"
    elif mode == 'transfer':
        if count <= 0:
            return "N/A"
        speed_bytes = value / count
        speed_kbytes = speed_bytes / 1024
        if speed_kbytes > 1024:
            speed_mbytes = speed_kbytes / 1024
            if speed_mbytes > 1024:
                speed_gbytes = speed_mbytes / 1024
                return "{:.2f} GB/s".format(speed_gbytes)
            else:
                return "{:.2f} MB/s".format(speed_mbytes)
        else:
            return "{:.2f} KB/s".format(speed_kbytes)
    else:
        return "N/A"

def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize
    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--onnx', type=str, default='/app/outputs/models/regnet800/regnet_800m_final.onnx', help='ONNX model path')
    parser.add_argument('-n', '--name', type=str, default='regnet_800m', help='Model name')
    parser.add_argument('--img', type=str, default='/app/quantize/img_00001.jpg', help='Image for inference')
    parser.add_argument('--labels', type=str, default='/app/quantize/labels.txt', help='Labels file for age head (output 0)')
    parser.add_argument('--img_size', nargs='+', type=int, default=[224,224], help='Inference size h,w')
    parser.add_argument('--rknn', type=str, default='./rknn_models', help='RKNN output directory')
    parser.add_argument('--platform', type=str, default='rk3576', help='Target platform')
    parser.add_argument('--mean_values', nargs='+', type=float, default=[0.5, 0.5, 0.5], help='Mean values for normalization')
    parser.add_argument('--std_values', nargs='+', type=float, default=[0.5, 0.5, 0.5], help='Std values for normalization')
    parser.add_argument('--datasets', type=str, default='/app/quantize/dataset.txt', help='Datasets path file for calibration')
    parser.add_argument('--do_quant', action='store_true', help='Enable quantization (int8)')
    parser.add_argument('--no_quant', dest='do_quant', action='store_false', help='Disable quantization (fp16)')
    parser.set_defaults(do_quant=True)
    parser.add_argument('--accuracy_image', type=str, default=None, help='Image for accuracy analysis (optional)')
    args = parser.parse_args()

    rknn = RKNN(verbose=True)
    OUT_DIR = args.rknn
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    RKNN_MODEL_PATH = os.path.join(OUT_DIR, '{}_{}_{}.rknn'.format(
        args.name+'-'+str(args.img_size[1])+'-'+str(args.img_size[0]), args.platform, 'int8' if args.do_quant else 'fp16'))

    print('--> Config model')
    rknn.config(
        mean_values=args.mean_values, 
        std_values=args.std_values,
        quant_img_RGB2BGR=True,
        target_platform=args.platform,
        quantized_algorithm='normal',
        quantized_method='channel')
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx, inputs=['input'], input_size_list=[[1, 3, 224, 224]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=args.do_quant, dataset=args.datasets)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Accuracy analysis with cosine similarity
    print('--> Running accuracy analysis to compute cosine similarity')
    if args.accuracy_image and os.path.exists(args.accuracy_image):
        accuracy_input = args.accuracy_image
    else:
        # Use first image from dataset for accuracy analysis
        accuracy_input = None
        with open(dataset_file, 'r') as f:
            for line in f:
                img_name = line.strip()
                img_path = img_name if os.path.exists(img_name) else os.path.join('/app/images', img_name)
                if os.path.exists(img_path):
                    accuracy_input = img_path
                    break
    
    if accuracy_input:
        print(f'--> Accuracy analysis on {accuracy_input}')
        # This will print cosine similarity for each layer
        ret = rknn.accuracy_analysis(inputs=[accuracy_input])
        if ret != 0:
            print('Accuracy analysis failed!')
        else:
            print('Cosine similarity analysis completed - check output above')
    else:
        print('No suitable image found for accuracy analysis')


    # Batch inference over dataset.txt
    print('--> Running batch inference')
    dataset_file = args.datasets if os.path.exists(args.datasets) else './dataset.txt'
    with open(dataset_file, 'r') as f:
        image_list = [line.strip() for line in f if line.strip()]

    img_dir = '/app/images'
    total_time = 0.0
    total_bytes = 0
    count = 0

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    for img_name in image_list:
        img_path = img_name if os.path.exists(img_name) else os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            if count < 3:
                print(f"Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            if count < 3:
                print(f"Failed to read image: {img_path}")
            continue
        # Resize image to 224x224 for model input
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0) 
        img_bytes = os.path.getsize(img_path)
        start_time = time.time()
        outputs = rknn.inference(inputs=[img], data_format='nchw')
        elapsed = time.time() - start_time
        if outputs is None:
            if count < 3:
                print(f'Inference failed for {img_path}!')
            continue
        if count < 3:
            print(f'Image: {img_name}')
            show_outputs(outputs, [args.labels, '/app/quantize/gender_labels.txt'])
            print('Speed:', readable_speed(elapsed, mode='inference', count=1), '| Data:', readable_speed(img_bytes, mode='transfer', count=elapsed))
            print('-' * 40)
        total_time += elapsed
        total_bytes += img_bytes
        count += 1

    if count > 0:
        print(f'Batch inference completed for {count} images.')
        print('Average Inference Speed:', readable_speed(total_time, mode='inference', count=count))
        print('Average Data Transfer Rate:', readable_speed(total_bytes, mode='transfer', count=total_time))
    else:
        print('No images processed.')

    rknn.release()


if __name__ == '__main__':
    main()
