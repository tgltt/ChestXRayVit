# ChestXRayVit X光胸片肺部疾病检测项目
<div>
本工程是AI<b>深度学习</b>在计算机视觉<b><i>物体分类</i></b>方面的应用，基于<b>Vison Transformer</b>架构(Vit)，采用的数据集是Kaggle X光胸片数据集，共<b>4</b>个分类，<b>7,100</b>多张图片，数据集压缩包占用<b>1.8GB</b>的磁盘空间，该数据集Kaggle持续在更新中。<br>
</div>
<br>

<li><b>4种肺部疾病类别</b><br>
<div>
    Normal(正常)/Pneumonia(一般性肺炎)/Covid-19(新冠肺炎)/Tuberculosis(肺结核)
</div>
<br>

<li><b>Kaggle X光胸片数据集</b><br>
<div>
数据集地址：https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis <br>
数据集页面：
<img width="1079" alt="image" src="https://user-images.githubusercontent.com/36066270/226508741-836be57a-27c1-4963-9c27-fa5da29d3d6d.png">
</div>
<br>
<br>

<li><b>工程结构说明</b><br>

|    文件/文件夹         |                   功能描述                         |
|       ----            |                    ----                           |
| kaggle_jupyter        | 存放本工程运行在kaggle网站jupyter notebook的实现文件 |
| paper                 | 存放模型相关的论文(Transformer、Vision Transformer) |
| save_weights          | 存放模型每个迭代训练完成后的模型参数                 |
| test_res              | 存放测试模型用的图片或视频                          |
| data.py               | 定义数据集定义、数据集加载器等                       |
| model.py              | 定义模型                                           |
| main.py               | 工程主函数，定义命令行参数、构建模型并进行训练等       |
| transforms.py         | 预处理图片的功能类                                  |
| util.py               | 辅助模型定义的工具类                                |
| predict_test.py       | 验证模型的推理功能类                                |
| calculate_dataset_mean_std.py | 计算训练集图像RGB三个通道的均值和方差，考虑X光片基本是灰度图，最后会将3通道的均值、方差 求平均，作为模型预处理的输入。由于X光片数据集并非ImageNet、CoCo、Voc这类国际大型数据集，故数据集和标准差需自行计算 |
<br>
<br>
 
<li><b>环境配置</b><br>
<div>环境要求 Python>=3.6，且 PyTorch>=1.7
</div>
    
<li><b>工程算法说明</b><br>
<div>
    本工程基于<b>Transformer框架</b>，根据任务要求和X光片数据集的特点，进行<b>算法模型改造</b>，改造点为裁减Transformer模型并只保留其Encoder、使用Vit将图片切碎(Patch)并将每个碎片Flatten后导入Transformer的Encoder，经Encoder提取特征后，最后经由一个Linear作分类。<br>
</div>
