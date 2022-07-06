## 1. PP-YOLOE模型TRT导出教程
环境安装：
CUDA 10.2 + [cudnn 8.2.1](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) + [TensorRT 8.2](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/install-guide/index.htm)

python -m pip install 'pycuda<2021.1'

### Paddle模型导出
```commandline
python3.7 tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams trt=True exclude_nms=True
```

### ONNX模型转换 + 定制化修改EfficientNMS_TRT
```commandline
python3.7 deploy/third_engine/demo_ppyoloe/onnx_custom.py --onnx_file=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --model_dir=output_inference/ppyoloe_crn_l_300e_coco/ --opset_version=11
```

### TensorRT模型导出
```commandline
trtexec --onnx=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --saveEngine=ppyoloe_crn_l_300e_coco.engine --tacticSources=-cublasLt,+cublas
```

### 运行TRT推理
```commandline
python3.7 deploy/third_engine/demo_ppyoloe/trt_infer.py --infer_cfg=output_inference/ppyoloe_crn_l_300e_coco/infer_cfg.yml --trt_engine=ppyoloe_crn_l_300e_coco.engine --image_file=demo/000000014439.jpg
```


## 2. PP-YOLOE模型TRT INT8量化教程
### Paddle模型导出
```commandline
python3.7 tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams trt=True exclude_nms=True
```
如果只导出单输入`image`的模型，执行如下命令：
```commandline
python3.7 tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams trt=True exclude_post_process=True
```

### ONNX模型转换 + 定制化修改EfficientNMS_TRT
```commandline
python3.7 deploy/third_engine/demo_ppyoloe/onnx_custom.py --onnx_file=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --model_dir=output_inference/ppyoloe_crn_l_300e_coco/ --opset_version=11
```
如果只需要转换ONNX，执行如下命令：
```commandline
paddle2onnx --model_dir=output_inference/ppyoloe_crn_l_300e_coco/ --save_file=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --model_filename=model.pdmodel --params_filename=model.pdiparams --opset_version=11 --enable_onnx_checker=True
```

### TensorRT模型 INT8量化
```commandline
python3.7 deploy/third_engine/demo_ppyoloe/post_quant_trt.py --onnx_file=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --output=ppyoloe_crn_l_300e_coco_ptq.engine --calibration_data=/paddle/coco/val2017/ --int8
```
如果模型只有`image`单输入，执行如下命令：
```commandline
python3.7 deploy/third_engine/demo_ppyoloe/post_quant_trt.py --onnx_file=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --output=ppyoloe_crn_l_300e_coco_ptq.engine --calibration_data=/paddle/coco/val2017/ --int8 --just_image
```

### 运行TRT推理 + coco eval
```commandline
python3.7 deploy/third_engine/demo_ppyoloe/trt_infer.py --infer_cfg=output_inference/ppyoloe_crn_l_300e_coco/infer_cfg.yml --trt_engine=ppyoloe_crn_l_300e_coco_ptq.engine --image_dir=/paddle/coco/val2017 --save_coco
```

```commandline
python3.7 deploy/third_engine/demo_ppyoloe/coco_eval.py --anno_file=/paddle/coco/annotations/instances_val2017.json --json_file=results.json
```

```
# coco mAP
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.655
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.537
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.320
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.538
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.376
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.632
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.693
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.510
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.756
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.854
```
