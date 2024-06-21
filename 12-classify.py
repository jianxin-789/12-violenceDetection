import torch
from model import ViolenceClassifier
from torchvision import transforms
from PIL import Image

class ViolenceClass:
    def __init__(self, checkpoint_path):
        # 加载模型
        self.model = ViolenceClassifier.load_from_checkpoint(checkpoint_path)
        self.model.eval()  # 设置为评估模式

    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
        # 确保输入是Tensor并且设备正确
        if not isinstance(imgs, torch.Tensor):
            raise TypeError("Input images should be a PyTorch Tensor")
        if imgs.ndim != 4 or imgs.size(1) != 3 or imgs.size(2) != 224 or imgs.size(3) != 224:
            raise ValueError("Input images should have shape [n, 3, 224, 224]")
            # 如果有必要，确保Tensor在正确的设备上（如GPU）
        if next(self.model.parameters()).device != imgs.device:
            images = imgs.to(next(self.model.parameters()).device)

        with torch.no_grad():  # 不需要计算梯度
            # 进行预测
            outputs = self.model(imgs)
            _, predicted_classes = torch.max(outputs, 1)

            # 将预测类别转换为Python列表
        predicted_classes_list = predicted_classes.tolist()
        return predicted_classes_list

    #def misc(self,checkpoint_path):
    # 其他处理函数

'''
一个示例：
checkpoint_path = 'checkpoints/best_model.ckpt'  # 修改为您的检查点文件路径
classifier = ViolenceClass(checkpoint_path)

# 假设您有一个形状为[n, 3, 224, 224]的Tensor，名为test_images
# test_images = ... # 这里应该是您的测试图像Tensor

# 进行分类
predicted_classes = classifier.classify(test_images)
print(predicted_classes)  # 输出预测类别列表
'''