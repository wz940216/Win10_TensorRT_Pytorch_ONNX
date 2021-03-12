# TensorRT部署

**Tensorrt版本**：TensorRT-7.2.3.4.Windows10.x86_64.cuda-11.1.cudnn8.1

## Win10 Tensorrt  安装

<!--配置前保证CUDA11.1、cudnn8.1、opencv3.x安装成功并配置好环境变量-->

1. 去这个地方下载对应的版本 https://developer.nvidia.com/nvidia-tensorrt-7x-download
2. 下载完成后，解压。
3. 将 TensorRT-7.2.3.4\include中头文件 copy 到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include
4. 将TensorRT-7.2.3.4\lib 中所有lib文件 copy 到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64
5. 将TensorRT-7.2.3.4\lib 中所有dll文件copy 到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin
6. 用VS2019 打开 TensorRT-7.2.3.4\samples\sampleMNIST\sample_mnist.sln
7. 实测Release版本可直接用，Debug版本配置有些问题。
8. 用anaconda虚拟环境 进入TensorRT-7.2.3.4\data\mnist 目录，执行python download_pgms.py
9. 进入TensorRT-7.2.3.4\bin，用cmd执行，sample_mnist.exe --datadir = d:\path\to\TensorRT-7.2.3.4\data\mnist\
10. 执行成功则说明tensorRT 配置成功 

参考链接：https://arleyzhang.github.io/articles/7f4b25ce/

## onnx模型的转换

**测试模型** 

`torchvision`中的`resnet18`，输入`[1,3,224,224]`输出FC层换成`[1,5]`，五个结果值便于直观对比输出结果的一致性。

**主要代码** 

```python
torch.onnx.export(model, input, ONNX_FILE_PATH,input_names=["input"], output_names=["output"], export_params=True)
```

## onnx模型的调用

**pytorch转onnx**

```powershell
cd torch_to_onnx
python torch_to_onnx.py
```

**TensorRT中的模式：**

**INT8** 和 **fp16**模式

INT8推理仅在具有6.1或7.x计算能力的GPU上可用，并支持在诸如ResNet-50、VGG19和MobileNet等NX模型上进行图像分类。 

**DLA**模式 

DLA是NVIDIA推出的用于专做视觉的部件，一般用于开发板 Jetson AGX Xavier ，Xavier板子上有两个DLA，定位是专做常用计算(Conv+激活函数+Pooling+Normalization+Reshape)，然后复杂的计算交给Volta GPU做。DLA功耗很低，性能很好。参考https://zhuanlan.zhihu.com/p/71984335

## VS2019工程的配置

首推Release版本，可直接用，Debug版本配置有些问题。

属性->调试->命令参数->--fp16(根据需求选择--int8模式还是--fp16)

属性->VC++目录->包含目录->

```powershell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include;
D:\TensorRT-7.2.3.4\include;
D:\TensorRT-7.2.3.4\samples\common;
D:\TensorRT-7.2.3.4\samples\common\windows;
D:\opencv\build\include;
D:\opencv\build\include\opencv;
D:\opencv\build\include\opencv2;$(IncludePath)
```

属性->VC++目录->库目录->

```powershell
D:\opencv\build\x64\vc15\lib;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64;$(LibraryPath)
```

属性->C/C++->附加包含目录->

```powershell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include;%(AdditionalIncludeDirectories)
```

属性->链接器->输入->(自行删减)

```powershell
opencv_world342.lib;OpenCL.lib;cudnn_adv_infer64_8.lib;cudnn_ops_train64_8.lib;nppicc.lib;nvinfer.lib;cublas.lib;cudnn_adv_train.lib;cufft.lib;nppidei.lib;nvinfer_plugin.lib;cublasLt.lib;cudnn_adv_train64_8.lib;cufftw.lib;nppif.lib;nvjpeg.lib;cuda.lib;cudnn_cnn_infer.lib;curand.lib;nppig.lib;nvml.lib;cudadevrt.lib;cudnn_cnn_infer64_8.lib;cusolver.lib;nppim.lib;nvonnxparser.lib;cudart.lib;cudnn_cnn_train.lib;cusolverMg.lib;nppist.lib;nvparsers.lib;cudart_static.lib;cudnn_cnn_train64_8.lib;cusparse.lib;nppisu.lib;nvptxcompiler_static.lib;cudnn.lib;cudnn_ops_infer.lib;myelin64_1.lib;nppitc.lib;nvrtc.lib;cudnn64_8.lib;cudnn_ops_infer64_8.lib;nppc.lib;npps.lib;cudnn_adv_infer.lib;cudnn_ops_train.lib;nppial.lib;nvblas.lib;%(AdditionalDependencies)
```

## 验证输出结果

<!--说明：TensorRT每次构建引擎时会根据当前GPU以及使用情况进行网络优化。每次构建的新引擎可能存在输出有小数点后三位不同。但整体精度变化不大，一般不会影响模型推理结果。构建引擎序列化plan文件，再次加载后输出不变化。-->

**Pytorch 输出**

module output:

tensor([[  4.7960,  -1.9805,   7.9566,   2.4818, -13.3275]], device='cuda:0', grad_fn=<AddmmBackward>)

onnx output:

output [[  4.796035   -1.9805057   7.9566107   2.4817739 -13.327486 ]]

**TensorRT输出 --fp16**

|      | 名称     | 值                                                           |   类型   |
| ---- | -------- | ------------------------------------------------------------ | :------: |
| ◢    | output,6 | 0x00000275858da9d0 {4.53114319, -1.66334152, 7.31781292, 2.51477814, -12.7687111, 1.401e-45#DEN} | float[6] |
|      | [0]      | 4.53114319                                                   |  float   |
|      | [1]      | -1.66334152                                                  |  float   |
|      | [2]      | 7.31781292                                                   |  float   |
|      | [3]      | 2.51477814                                                   |  float   |
|      | [4]      | -12.7687111                                                  |  float   |

**TensorRT输出 --int8** 

这里输出一个warning 提示未使用int8校准

`[03/11/2021-15:16:58] [W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32.`

原因是未指定每一个tensor的量化范围，需要传入`--ranges=per_tensor_dynamic_range_file.txt`

类似于

`gpu_0/data_0: 1.00024`
`gpu_0/conv1_1: 5.43116`
`gpu_0/res_conv1_bn_1: 8.69736`
`gpu_0/res_conv1_bn_2: 8.69736`
`gpu_0/pool1_1: 8.69736`
`gpu_0/res2_0_branch2a_1: 12.819`
`gpu_0/res2_0_branch2a_bn_1: 5.47741`
`gpu_0/res2_0_branch2a_bn_2: 5.58704`
`gpu_0/res2_0_branch2b_1: 5.27718`
`gpu_0/res2_0_branch2b_bn_1: 5.08003`
`gpu_0/res2_0_branch2b_bn_2: 5.08003`
`gpu_0/res2_0_branch2c_1: 2.33625`
`gpu_0/res2_0_branch2c_bn_1: 3.17859`
`gpu_0/res2_0_branch1_1: 6.10492`

未使用校准的int8量化精度损失很大

|      | 名称     | 值                                                           | 类型     |
| ---- | -------- | ------------------------------------------------------------ | -------- |
| ◢    | output,6 | 0x00000229086eee00 {21.9405861, -2.17396450, 15.4984961, -12.8306799, -22.7700291, 5.86610616e-24} | float[6] |
|      | [0]      | 21.9405861                                                   | float    |
|      | [1]      | -2.17396450                                                  | float    |
|      | [2]      | 15.4984961                                                   | float    |
|      | [3]      | -12.8306799                                                  | float    |
|      | [4]      | -22.7700291                                                  | float    |

## 验证速度的提升

**pytorch** 

```python
torch.cuda.synchronize()
start = time.time()
for i in range(1000):    
    output = model(input)
torch.cuda.synchronize()
print("pytorch time used ",(time.time()-start)/1000)
```



time used  4.887054920196534 ms

**TensorRT**

```c++
CSpendTime time;
time.Start();
for (int i = 0; i < 1000; i++)
{
    bool status = context->executeV2(buffers.getDeviceBindings().data());
}
double dTime = time.End();
printf("time used %.8f\n", dTime/1000);
```



**TensorRT int8**

第一次 time used 0.99424440 ms

第二次 time used 0.98159920 ms

**TensorRT fp16**

第一次 time used 0.97576060 ms

第二次 time used 0.97672040 ms



