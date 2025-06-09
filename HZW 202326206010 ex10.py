from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, SIFT
from skimage.feature import graycomatrix, graycoprops
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


class Dataloader:
    def __init__(self, root_dir, num_classes, ext):
        self.train_dir = root_dir / "train"
        self.test_dir = root_dir / "test"
        self.num_classes = num_classes
        self.ext = ext  # 图片文件扩展名".jpg"

    def load_paths_and_labels(self):
        train_paths, train_labels = [], []
        test_paths, test_labels = [], []
        # 训练集
        for img_path in sorted(self.train_dir.glob(f"*{self.ext}")):
            try:
                idx = int(img_path.stem)  # 检查文件名是不是整数，不含后缀
            except ValueError:
                continue
            category = idx // 100  # 每100张图片属于一个类别
            if 0 <= category < self.num_classes:
                train_paths.append(img_path)
                train_labels.append(category)

        # 测试集
        for img_path in sorted(self.test_dir.glob(f"*{self.ext}")):
            try:
                idx = int(img_path.stem)
            except ValueError:
                continue
            category = idx // 100
            if 0 <= category < self.num_classes:
                test_paths.append(img_path)
                test_labels.append(category)
        # 打印读取结果
        # print("训练集:", train_paths)
        # print("测试集:", test_labels)
        print(f"[Dataloader] 训练集路径数量: {len(train_paths)}, 标签数量: {len(train_labels)}")
        print(f"[Dataloader] 测试集路径数量: {len(test_paths)}, 标签数量: {len(test_labels)}")
        return train_paths, train_labels, test_paths, test_labels


class FeatureExtractor:
    def __init__(self, vocab_size, glcm_distances, glcm_angles, glcm_levels, lbp_p, lbp_r, lbp_method, lbp_hist_size,
                 color_hist_bins, use_cnn, device):
        # GLCM 参数
        self.glcm_distances = glcm_distances  # GLCM中灰度共生矩阵的距离列表
        self.glcm_angles = glcm_angles  # GLCM中灰度共生矩阵的角度列表
        self.glcm_levels = glcm_levels  # 灰度量化级别
        # LBP 参数
        self.lbp_p = lbp_p  # 邻域点数
        self.lbp_r = lbp_r  # 半径
        self.lbp_method = lbp_method  # LBP方法，"uniform"
        self.lbp_hist_size = lbp_hist_size  # LBP直方图长度
        # 颜色直方图参数
        self.color_hist_bins = color_hist_bins  # HSV三通道直方图的bin数量，例如 (16,16,16)
        # SIFT + 视觉词典参数
        self.sift = SIFT()
        self.vocab_size = vocab_size  # 视觉词典大小
        self.kmeans_model = None
        self.device = device
        # 是否使用 CNN（ResNet50）特征
        self.use_cnn = use_cnn
        if self.use_cnn:
            self._build_cnn_model()
        # ResNet50 输入预处理：Resize -> ToTensor -> Normalize
        self.cnn_transform = T.Compose([
            T.Resize((224, 224)),  # ResNet50 要求输入尺寸为 224x224
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # 各特征维度
        self.dim_glcm = 4 * len(self.glcm_distances) * len(self.glcm_angles)  # GLCM 属性维度
        self.dim_bow = self.vocab_size  # BOW 维度
        self.dim_lbp = self.lbp_hist_size  # LBP 维度
        self.dim_color = sum(self.color_hist_bins)  # 颜色直方图维度
        self.dim_cnn = 2048  # ResNet50 去掉 FC 层后输出特征维度

    def _build_cnn_model(self):
        # 载入预训练权重
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 去掉最后的全连接层, 用于特征提取
        modules = list(model.children())[:-1]
        self.cnn_model = nn.Sequential(*modules).to(self.device)
        self.cnn_model.eval()  # 设置为评估模式，关闭 dropout、batchnorm 更新等

    @staticmethod
    def _pil_to_gray_array(pil_img, levels=256):
        gray = pil_img.convert("L")  # 转为灰度图
        arr = np.array(gray, dtype=np.uint8)
        return arr.copy()

    @staticmethod
    def _pil_to_rgb_array(pil_img):
        rgb = pil_img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)

    def build_sift_vocab(self, train_paths: List[Path]):
        all_descriptors = []
        for path in tqdm(train_paths, desc="Building SIFT vocab"):
            pil_img = Image.open(path)
            # 转为量化灰度图，取值范围 [0, glcm_levels-1]
            gray_arr = self._pil_to_gray_array(pil_img, levels=self.glcm_levels)
            # 归一化到 [0,1]
            gray_float = gray_arr.astype(np.float32) / 255.0
            try:
                # 提取 SIFT keypoints 和 descriptors
                self.sift.detect_and_extract(gray_float)
            except RuntimeError:
                # 跳过无法提取SIFT特征的图像
                continue

            des = self.sift.descriptors  # descriptors: shape=(n_keypoints, 128)
            if des is not None and des.size > 0:
                all_descriptors.append(des)

        if not all_descriptors:
            raise ValueError("SIFT描述子为空，无法构建视觉词典。")

        # 将所有图片的 descriptors 堆叠，得到形状 (总关键点数, 128)
        all_descriptors = np.vstack(all_descriptors)
        # print(f"[build_sift_vocab] 总 descriptors 数量: {all_descriptors.shape}")  # (n_descriptors, 128)

        # 使用 KMeans 聚类，词典大小为 vocab_size
        self.kmeans_model = KMeans(n_clusters=self.vocab_size, n_init=10, random_state=42)
        self.kmeans_model.fit(all_descriptors)
        # print(f"[build_sift_vocab] 已训练 KMeans 词典，聚类中心形状: {self.kmeans_model.cluster_centers_.shape}")  # (vocab_size, 128)

    def extract_glcm(self, pil_img):
        gray = self._pil_to_gray_array(pil_img, levels=self.glcm_levels)  # shape=(H, W)
        # 计算灰度共生矩阵，返回形状 (levels, levels, len(distances), len(angles))
        glcm = graycomatrix(gray, distances=self.glcm_distances, angles=self.glcm_angles, levels=self.glcm_levels,
                            symmetric=True, normed=True)
        # 需要计算的四种属性
        props = ["contrast", "correlation", "energy", "homogeneity"]
        feats = []
        for prop in props:
            tmp = graycoprops(glcm, prop)  # 返回 shape: (len(distances), len(angles))
            feats.extend(tmp.flatten())
        feats = np.array(feats, dtype=np.float32)  # shape=(dim_glcm,)
        # print(f"[extract_glcm] GLCM 特征维度: {feats.shape}")  # (dim_glcm,)
        return feats

    def extract_bow(self, pil_img):
        if self.kmeans_model is None:
            raise RuntimeError("请先调用 build_sift_vocab() 构建视觉词典。")

        gray_arr = self._pil_to_gray_array(pil_img, levels=self.glcm_levels)  # shape=(H, W)
        gray_float = gray_arr.astype(np.float32) / 255.0

        try:
            # 提取 SIFT
            self.sift.detect_and_extract(gray_float)
        except RuntimeError:
            # 如果提取失败，返回全零向量
            hist_zero = np.zeros(self.vocab_size, dtype=np.float32)
            print(f"[extract_bow] SIFT 提取失败，返回零向量，维度: {hist_zero.shape}")
            return hist_zero

        des = self.sift.descriptors  # shape=(n_keypoints, 128)
        if des is None or des.size == 0:
            # 如果没有描述符，返回零向量
            hist_zero = np.zeros(self.vocab_size, dtype=np.float32)
            print(f"[extract_bow] 未检测到描述符，返回零向量，维度: {hist_zero.shape}")
            return hist_zero

        # 使用 KMeans 对每个 descriptor 进行聚类，返回单词索引列表
        words = self.kmeans_model.predict(des)  # shape=(n_keypoints,)
        # 计算词频直方图
        hist, _ = np.histogram(words, bins=np.arange(self.vocab_size + 1))
        hist = hist.astype(np.float32)  # shape=(vocab_size,)
        # L2 归一化
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        # print(f"[extract_bow] BOW 特征维度: {hist.shape}")  # (vocab_size,)
        return hist

    def extract_lbp(self, pil_img):
        gray = self._pil_to_gray_array(pil_img, levels=self.glcm_levels)  # shape=(H, W)
        # 计算 LBP map，返回 shape=(H, W)，元素值在 [0, lbp_hist_size)
        lbp_map = local_binary_pattern(
            gray,
            P=self.lbp_p,
            R=self.lbp_r,
            method=self.lbp_method
        )
        # 计算 LBP 直方图
        hist, _ = np.histogram(
            lbp_map.ravel(),
            bins=np.arange(self.lbp_hist_size + 1),
            range=(0, self.lbp_hist_size)
        )
        hist = hist.astype(np.float32)  # shape=(lbp_hist_size,)
        # L2 归一化
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        # print(f"[extract_lbp] LBP 特征维度: {hist.shape}")  # (lbp_hist_size,)
        return hist

    def extract_color_hist(self, pil_img):
        rgb_arr = self._pil_to_rgb_array(pil_img)  # shape=(H, W, 3)

        rgb_norm = rgb_arr.astype(np.float32) / 255.0  # 归一化到 [0,1]
        hsv = mcolors.rgb_to_hsv(rgb_norm)  # shape=(H, W, 3)，值在 [0,1]

        # 将 HSV 三个通道映射回 [0,255] 的整数
        h_chan = (hsv[:, :, 0] * 255).astype(np.uint8)  # shape=(H, W)
        s_chan = (hsv[:, :, 1] * 255).astype(np.uint8)
        v_chan = (hsv[:, :, 2] * 255).astype(np.uint8)

        feats = []
        # 对 H、S、V 三个通道分别计算直方图
        for chan, bins in zip([h_chan, s_chan, v_chan], self.color_hist_bins):
            hist, _ = np.histogram(chan, bins=bins, range=(0, 256))  # shape=(bins,)
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            feats.append(hist)
        feats_concat = np.concatenate(feats)  # shape=(sum(color_hist_bins),)
        # print(f"[extract_color_hist] 颜色直方图特征维度: {feats_concat.shape}")  # (dim_color,)
        return feats_concat

    def extract_cnn(self, pil_img):
        if not self.use_cnn:
            raise RuntimeError("当前未启用 CNN 特征提取。")
        # 预处理：Resize、ToTensor、Normalize，返回 shape=(3,224,224)
        tensor = self.cnn_transform(pil_img).unsqueeze(0).to(self.device)  # shape=(1,3,224,224)
        # print(f"[extract_cnn] 输入给 ResNet50 的张量形状: {tensor.shape}")  # (1,3,224,224)
        with torch.no_grad():
            feats = self.cnn_model(tensor)  # 输出 shape=(1,2048,1,1)
        feats_flat = feats.view(-1).cpu().numpy().astype(np.float32)  # 转为 shape=(2048,)
        # print(f"[extract_cnn] ResNet50 提取后特征维度: {feats_flat.shape}")  # (2048,)
        return feats_flat

    def extract_all(self, paths):
        feats_list = []

        for path in tqdm(paths, desc="Extracting features"):
            pil_img = Image.open(path)

            f_glcm = self.extract_glcm(pil_img)  # (dim_glcm,)
            f_bow = self.extract_bow(pil_img)  # (dim_bow,)
            f_lbp = self.extract_lbp(pil_img)  # (dim_lbp,)
            f_color = self.extract_color_hist(pil_img)  # (dim_color,)

            if self.use_cnn:
                f_cnn = self.extract_cnn(pil_img)  # (dim_cnn,)
                # 将所有特征拼接：dim = dim_glcm + dim_bow + dim_lbp + dim_color + dim_cnn
                feat = np.concatenate([f_glcm, f_bow, f_lbp, f_color, f_cnn])
                # print(f"[extract_all] 图像 {path.name} 综合特征维度: {feat.shape}")  # (total_dim,)
            else:
                # 不使用 CNN 时，dim = dim_glcm + dim_bow + dim_lbp + dim_color
                feat = np.concatenate([f_glcm, f_bow, f_lbp, f_color])
                # print(f"[extract_all] 图像 {path.name} 综合特征维度: {feat.shape}")  # (total_dim,)

            feats_list.append(feat)

        # 将 list 转为 ndarray，形状 (n_samples, total_dim)
        feats_matrix = np.vstack(feats_list)
        # print(f"[extract_all] 最终特征矩阵维度: {feats_matrix.shape}")  # (n_samples, total_dim)
        return feats_matrix


def main(vocab_size, use_cnn, pca_components):
    root_dir = Path("..") / "dataset"
    num_classes = 20
    image_ext = ".jpg"
    # GLCM 参数
    default_glcm_distances = [1]
    default_glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm_levels = 256  # 灰度量化级别

    # LBP 参数
    lbp_p = 8
    lbp_r = 1
    lbp_method = "uniform"
    lbp_hist_size = 59

    # 颜色直方图参数
    color_hist_bins = (16, 16, 16)
    device = torch.device("cpu")
    # 数据加载，读取图像路径及标签
    data_loader = Dataloader(root_dir, num_classes=num_classes, ext=image_ext)
    train_paths, train_labels, test_paths, test_labels = data_loader.load_paths_and_labels()
    print(f"[main] 训练集数量: {len(train_paths)}, 测试集数量: {len(test_paths)}")

    # 初始化特征提取器, 包括 ResNet50, 并基于训练集构建 SIFT 视觉词典
    extractor = FeatureExtractor(vocab_size=vocab_size, glcm_distances=default_glcm_distances,
                                 glcm_angles=default_glcm_angles, glcm_levels=glcm_levels, lbp_p=lbp_p, lbp_r=lbp_r,
                                 lbp_method=lbp_method, lbp_hist_size=lbp_hist_size, color_hist_bins=color_hist_bins,
                                 use_cnn=use_cnn, device=device)
    extractor.build_sift_vocab(train_paths)

    # 提取训练集和测试集特征
    print("[main] 正在为训练集提取特征 …")
    x_train = extractor.extract_all(train_paths)
    print("[main] 正在为测试集提取特征 …")
    x_test = extractor.extract_all(test_paths)
    y_train = np.array(train_labels, dtype=np.int32)
    y_test = np.array(test_labels, dtype=np.int32)
    print(f"[main] x_train 形状: {x_train.shape}")
    print(f"[main] x_test 形状: {x_test.shape}")

    # 特征标准化,均值为0,方差为1
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # PCA降维
    if pca_components is not None:
        print(f"[PCA] 将特征从 {x_train_scaled.shape[1]} 维降到 {pca_components} 维")
        pca = PCA(n_components=pca_components, random_state=42)
        x_train_final = pca.fit_transform(x_train_scaled)  # shape=(n_train, pca_components)
        x_test_final = pca.transform(x_test_scaled)  # shape=(n_test, pca_components)
        print(f"[PCA] 完成: x_train_final 形状 = {x_train_final.shape}")
        print(f"[PCA] 完成: x_test_final 形状 = {x_test_final.shape}")
    else:
        x_train_final = x_train_scaled
        x_test_final = x_test_scaled

    # 定义多个分类器:SVM、KNN、RandomForest、Naive Bayes
    classifiers = {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "NaiveBayes": GaussianNB()
    }
    metrics_dict = {}

    # 训练并评估各单模型
    for name, clf in classifiers.items():
        print(f"\n[{name}] 开始训练 …")
        clf.fit(x_train_final, y_train)
        print(f"[{name}] 训练完成，开始预测测试集 …")
        preds = clf.predict(x_test_final)  # preds 形状=(n_test,)

        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")
        metrics_dict[name] = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Preds": preds
        }
        print(f"[{name}] Macro-Precision: {prec:.4f}  Macro-Recall: {rec:.4f}  Macro-F1: {f1:.4f}")

    # # 构建 VotingClassifier，并评估
    # voting_estimators = [
    #     ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
    #     ("knn", KNeighborsClassifier(n_neighbors=5)),
    #     ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    #     ("nb", GaussianNB())
    # ]
    # voting_clf = VotingClassifier(
    #     estimators=voting_estimators,
    #     voting="soft"
    # )
    # print("\n[Voting] 开始训练 …")
    # voting_clf.fit(x_train_final, y_train)
    # print("[Voting] 训练完成，开始预测测试集 …")
    # preds_vote = voting_clf.predict(x_test_final)  # shape=(n_test,)
    #
    # prec_v = precision_score(y_test, preds_vote, average="macro")
    # rec_v = recall_score(y_test, preds_vote, average="macro")
    # f1_v = f1_score(y_test, preds_vote, average="macro")
    # metrics_dict["Voting"] = {
    #     "Precision": prec_v,
    #     "Recall": rec_v,
    #     "F1": f1_v,
    #     "Preds": preds_vote
    # }
    # print(f"[Voting] Macro-Precision: {prec_v:.4f}  Macro-Recall: {rec_v:.4f}  Macro-F1: {f1_v:.4f}")

    # 选出 F1 分数最高的最佳模型
    best_model_name = max(metrics_dict, key=lambda name: metrics_dict[name]["F1"])
    best_metrics = metrics_dict[best_model_name]
    best_preds = best_metrics["Preds"]

    print("\n===== 最优模型与对应宏平均指标 =====")
    print(f"模型名称：{best_model_name}")
    print(f"Macro-Precision: {best_metrics['Precision']:.4f}")
    print(f"Macro-Recall   : {best_metrics['Recall']:.4f}")
    print(f"Macro-F1       : {best_metrics['F1']:.4f}")

    # 将最佳模型的预测结果保存到文件，保留原始图片文件名
    out_txt = f"{best_model_name.lower()}_best_preds.txt"
    with open(out_txt, "w") as f:
        # 如果需要表头，可取消下面一行注释：
        f.write("filename true_label pred_label\n")
        for path, true_label, pred_label in zip(test_paths, test_labels, best_preds):
            f.write(f"{path.name} {true_label} {pred_label}\n")
    print(f"[Output] 最优模型 ({best_model_name}) 的预测结果已保存到：{out_txt}")

    # 计算混淆矩阵
    cm_counts = confusion_matrix(y_test, best_preds, labels=list(range(num_classes)))
    cm_prob = cm_counts.astype(np.float32) / cm_counts.sum(axis=1, keepdims=True)

    # 计算每个类别的准确率A_i及平均准确率A
    class_accuracies = np.diag(cm_prob)  # 取对角线，shape=(num_classes,)
    average_accuracy = class_accuracies.mean()
    print("\n===== 每个类别的分类准确率 ai =====")
    for i, acc in enumerate(class_accuracies):
        print(f"类别 {i:2d} 准确率: {acc:.4f}")
    print(f"\n===== 平均分类准确率 a =====\nA = {average_accuracy:.4f}")
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_prob, interpolation="nearest", cmap='Blues', aspect="auto")
    plt.title(f"{best_model_name} 归一化混淆矩阵\n平均准确率 a = {average_accuracy:.4f}", fontsize=14)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [str(i) for i in range(num_classes)], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [str(i) for i in range(num_classes)], fontsize=8)
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    thresh = cm_prob.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm_prob[i, j]:.2f}", horizontalalignment="center", verticalalignment="center",
                     color="white" if cm_prob[i, j] > thresh else "black", fontsize=6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vocab_size = 200
    use_cnn = True
    pca_components = 70
    main(vocab_size, use_cnn, pca_components)
