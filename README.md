# 图像分类

## 一、项目概述

### 本项目已上传至Github：https://github.com/night-stars-hzw/image_process

### 1.1 提取特征方法

此项目基于多种图像特征提取方法和经典机器学习分类器，构建一个通用的图像分类系统。主要包含以下特征提取模块：

（1）**GLCM（灰度共生矩阵）** ：用于提取图像纹理特征。

（2）**SIFT + BOW（词袋模型）** ：通过 SIFT 提取关键点描述子，然后使用 K-Means 聚类构建视觉词典，生成词频直方图。

（3）**LBP（局部二值模式）** ：提取局部纹理模式特征。

（4）**颜色直方图（HSV 空间）** ：对图像在 HSV 空间中三个通道分别计算直方图并归一化。

（5）**CNN（ResNet50）** ：调用预训练的 ResNet50 模型（去掉最后全连接层）提取深度特征。

### 1.2 使用的分类器

特征提取完成后，对所有特征进行拼接并标准化处理。如果使用了 PCA，可以将高维特征降维到指定维度。最后，训练并评估以下分类器：

（1）**支持向量机（SVM）**

（2）**k 近邻（KNN）**

（3）**随机森林（RandomForest）**

（4）**朴素贝叶斯（GaussianNB）**

### 1.3 输出

代码根据宏 F1 分数选出最佳模型，并输出该模型在测试集上的预测结果、混淆矩阵及各类别准确率。

## 二、目录结构

```
project_root/
│
├── dataset/
│   ├── train/          # 训练集图片，文件名为整数（例如 0.jpg、1.jpg ...），类别由文件名 //100 确定
│   └── test/           # 测试集图片，命名规则同上
│
├── code/
│   ├── main.py         # 主脚本，包含数据加载、特征提取、降维、分类器训练评估等完整流程
│   ├── dataloader.py   # Dataloader 类，用于加载图片路径与对应标签
│   └── feature_extractor.py  # FeatureExtractor 类，封装所有特征提取方法（GLCM、SIFT+BOW、LBP、颜色直方图、CNN）
│
├── requirements.txt    # Python 依赖列表
├── README.md           # 项目说明文档
└── <其他输出文件>  # 训练完成后生成的最佳模型预测文件，如：svm_best_preds.txt
```

## 三、环境依赖

**requirements.txt** ：

```
numpy==1.24.3
Image==9.3.0
skimage==0.21.0
torch==2.3.0
torchvision==0.18.1a0
sklearn==1.3.0
matplotlib==3.5.3
```

## 四、数据集准备

按照以下结构组织文件夹与图像：

dataset/

├── train/

│ ├── 0.jpg

│ ├── 1.jpg

│ ├── … (以整数命名)

│

└── test/

├── 1000.jpg

├── 1001.jpg

├── … (以整数命名)

图片文件名务必为整数且不含扩展名之外的多余字符。

每个类别包含 100 张图片，类别编号 = 文件名 // 100。

## 五、代码结构与核心流程

### 5.1  Dataloader 类 (code/dataloader.py)

#### 5.1.1 初始化参数

  * `root_dir`：数据集根目录路径（Path 对象），包含 train/ 与 test/ 子文件夹。
  * `num_classes`：总类别数（例如 20）。
  * `ext`：图像文件扩展名（例如 .jpg）。

#### 5.1.2 方法

  * **load_paths_and_labels()** ：
    * 扫描 train/ 与 test/ 文件夹下所有符合扩展名的文件，将文件名（整数）属性划分为类别：`category = idx // 100`。
    * 仅保留 `0 ≤ category < num_classes` 的图片，分别返回 train_paths, train_labels, test_paths, test_labels。

### 5.2  FeatureExtractor 类 (code/feature_extractor.py)

主类封装了五种不同类型的特征提取方法，以及相应参数配置与拼接流程。

#### 5.2.1 初始化

##### 5.2.1.1 GLCM 参数

  * `glcm_distances`：距离列表（如 [1]）
  * `glcm_angles`：角度列表（如 [0, π/4, π/2, 3π/4]）
  * `glcm_levels`：灰度量化级别（如 256）

##### 5.2.1.2 LBP 参数

  * `lbp_p`：邻域采样点数（如 8）
  * `lbp_r`：半径（如 1）
  * `lbp_method`：LBP 方法（如 "uniform"）
  * `lbp_hist_size`：直方图长度（如 59，对于 uniform 模式）

##### 5.2.1.3 颜色直方图参数

  * `color_hist_bins`：HSV 三通道直方图的 bin 数量，如 (16,16,16)。

##### 5.2.1.4 SIFT + BOW 参数

  * 自动初始化 SIFT() 对象。
  * `vocab_size`：视觉词典大小（KMeans 聚类中心个数）。
  * `kmeans_model`：后续由 build_sift_vocab() 训练得到的 KMeans 模型。

##### 5.2.1.5 CNN 参数

  * `use_cnn`：是否启用 CNN（ResNet50）提取深度特征。
  * `device`：PyTorch 设备（CPU 或 GPU）。
  * 内部调用 `_build_cnn_model()`：载入预训练的 ResNet50，去掉最后一层 FC，得到一个输出 2048 维特征的模型。
  * 配置图像预处理（Resize(224,224)、ToTensor()、Normalize()）。

##### 5.2.1.6 特征维度统计

  * dim_glcm = 4 × len(distances) × len(angles)
  * dim_bow = vocab_size
  * dim_lbp = lbp_hist_size
  * dim_color = sum(color_hist_bins)
  * dim_cnn = 2048

#### 5.2.2 词典构建

循环遍历所有 train_paths，对于每张图片：

（1）转为灰度数组，归一化到 [0,1]。

（2）调用 `self.sift.detect_and_extract(gray_float)` 提取 SIFT 描述子。

（3）如果成功提取，将 descriptors（形状 (n_keypoints, 128)）加入集合 all_descriptors。

（4）将所有图片的 descriptors 按行拼接为 (总关键点数, 128) 的大数组。

（5）使用 KMeans(n_clusters=vocab_size) 对 descriptors 进行聚类，得到视觉词典（聚类中心）。

（6）将训练好的 kmeans_model 保存在类属性中，供后续 BOW 提取使用。

#### 5.2.3 单张图像特征提取

##### 5.2.3.1 extract_glcm

（1）将 PIL 图像转为量化灰度数组。

（2）调用 skimage.feature.graycomatrix，计算 GLCM 矩阵（形状 (levels, levels, len(distances), len(angles))）。

（3）依次提取四个属性 contrast, correlation, energy, homogeneity 并展平为一维向量。

##### 5.2.3.2 extract_bow

（1）将 PIL 图像转为灰度并归一化。

（2）使用 `self.sift.detect_and_extract` 提取 descriptors（若失败，返回全零向量）。

（3）使用训练好的 `self.kmeans_model.predict(descriptors)` 将每个特征映射到视觉单词索引。

（4）统计词频直方图，长度为 vocab_size，并进行 L2 归一化。

##### 5.2.3.3 extract_lbp

（1）转换为灰度数组后，调用 skimage.feature.local_binary_pattern 计算 LBP map（值范围 [0, lbp_hist_size)）。

（2）统计直方图并 L2 归一化，输出维度 = lbp_hist_size。

##### 5.2.3.4 extract_color_hist

（1）将 PIL 图像转为 RGB 数组并归一化。

（2）使用 matplotlib.colors.rgb_to_hsv 转到 HSV 空间。

（3）对每个通道（H、S、V）映射到 [0,255]，分别统计直方图，维度依次为 color_hist_bins 中的数值，归一化后拼接，输出维度 = sum(color_hist_bins)。

##### 5.2.3.5 extract_cnn(pil_img)

（1）预处理成 (3,224,224) 的张量并归一化。

（2）使用 ResNet50（去掉最后 FC）前向推理，得到 shape (1,2048,1,1)。

（3）展平成 (2048,) 的向量。

#### 5.2.4 批量提取并拼接（extract_all）

输入：图像路径列表 paths。

（1）对每张图片依次调用上述五种提取函数（若 use_cnn=False，则不调用 CNN）。

（2）将各特征向量按顺序拼接得到一个总维度特征 feat。

（3）若启用 CNN，总维度 = dim_glcm + dim_bow + dim_lbp + dim_color + dim_cnn。

（4）否则 = dim_glcm + dim_bow + dim_lbp + dim_color。

（5）将每张图像的特征按行堆叠为一个二维数组 (n_samples, total_dim)，作为输出。

### 5.3  main 函数的参数说明

| 参数            | 说明                                                   | 示例值               |
| --------------- | ------------------------------------------------------ | -------------------- |
| root_dir        | 数据集根目录路径，需包含 train/ 与 test/ 子文件夹      | Path("..")/"dataset" |
| num_classes     | 类别总数，用于根据文件名整数生成标签                   | 20                   |
| ext             | 图片文件扩展名                                         | ".jpg"               |
| vocab_size      | 视觉词典大小（K-Means 聚类中心数），决定 BOW 特征维度  | 200                  |
| use_cnn         | 是否启用 ResNet50 提取深度特征                         | True / False         |
| pca_components  | PCA 降维后特征维度，若为 None 则跳过 PCA               | 70 / None            |
| glcm_distances  | GLCM 距离列表                                          | [1]                  |
| glcm_angles     | GLCM 角度列表（单位为弧度）                            | [0, π/4, π/2, 3π/4]  |
| glcm_levels     | 灰度量化级别                                           | 256                  |
| lbp_p           | LBP 邻域采样点数                                       | 8                    |
| lbp_r           | LBP 半径                                               | 1                    |
| lbp_method      | LBP 计算方法                                           | "uniform"            |
| lbp_hist_size   | LBP 直方图长度，对于 uniform 模式为 P*(P-1)+3（如 59） | 59                   |
| color_hist_bins | HSV 三通道直方图的 bin 数量（元组）                    | (16, 16, 16)         |
| device          | PyTorch 设备，可为 "cpu" 或 "cuda"                     | torch.device("cpu")  |

注：经实验测试，特征维度降到 70 左右效果最佳，故 pca_components 取 70。

### 5.4  查看输出

​	5.4.1**输出** ：

* 打印各个模型的 Macro-Precision、Macro-Recall、Macro-F1 以及最优模型名称。
* 最优模型的预测结果保存在根目录下，如：svm_best_preds.txt。
* 代码会绘制归一化混淆矩阵并弹出 Matplotlib 窗口。

​	5.4.2**输出说明** ：
* 最优模型的各项宏平均指标，如图一所示：

  <img src="D:\image_process\result\评估指标.png" alt="评估指标" style="zoom: 50%;" />

  ​																				图一：最优模型的各项宏平均指标

* `<model_name>_best_preds.txt`：例如 svm_best_preds.txt，包含三列（以空格分隔），如图二所示：
  
  * filename：测试集中图片文件名。
  
  * true_label：图片真实类别编号。
  
  * pred_label：最佳模型预测类别编号。
  
    <img src="D:\image_process\result\部分预测结果与真实结果对比.png" alt="部分预测结果与真实结果对比" style="zoom: 50%;" />

​																									图二：部分预测结果与真实结果对比							

**混淆矩阵** ：

* 各类图像的分类精确率打印结果，如图三、四所示：

  <img src="D:\image_process\result\各类准确率.png" alt="各类准确率" style="zoom: 33%;" />

  ​																					图三：各类图像各类图像的分类精确率（1）

  ​										<img src="D:\image_process\result\各类准确率1.png" alt="各类准确率1" style="zoom: 33%;" />	

  ​																					图四：各类图像各类图像的分类精确率（2）

* 可视化：训练结束时会弹出一个 Matplotlib 窗口，展示归一化后的混淆矩阵，并在图中标注各元素概率值。如图五所示						  	<img src="D:\image_process\result\混淆矩阵.png" alt="混淆矩阵" style="zoom: 50%;" />

## 六、注意事项与优化空间

​	6.1**数据量与训练时间** ：目前由于硬件方面原因，运行较慢。若有 GPU，可将 `device = torch.device("cuda")` 以加速 ResNet50 	特征提取。

​	6.2**算法替换与扩展** ：

* 可替换或增加更强的深度模型（如 ResNet101）。
* 在特征拼接后可尝试使用更强的集成方法（如 StackingClassifier、XGBoost 等）。但目前 svm 模型显著高于其他模型，故不用集成结果（集成效果低于 svm 结果）。

​	6.3**可视化与报告** ：

* 可以将混淆矩阵保存为文件，`plt.savefig("confusion_matrix.png")`。
* 可在评估阶段输出更多指标（如分类报告、ROC 曲线等）。

## 七、常见问题与排查

​	**7.1 找不到数据集路径或命名错误** ：确认 dataset/train/ 与 dataset/test/ 中文件名是否均为整数且后缀正确。例如，如果使用 .png，需在 main.py 中将 `image_ext = ".png"` 改为对应后缀。

​	**7.2 SIFT 提取失败报错** ：如果某张图像无法提取 SIFT 特征，代码会自动跳过并在 BOW 提取时返回零向量，但请检查是否大部分图像均无法提取，可能是图像内容过于平坦。

## 八、参考链接

  1. GitHub 项目仓库：<https://github.com/night-stars-hzw/image_process>
  2. ResNet50 预训练模型（PyTorch 官方文档）：<https://pytorch.org/vision/stable/models.html>
  3. scikit-image GLCM 文档：<https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycomatrix>
  4. scikit-learn PCA 文档：<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>