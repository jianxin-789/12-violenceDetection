if __name__ == '__main__':
 from pytorch_lightning import Trainer
 from pytorch_lightning.loggers import TensorBoardLogger

 from model import ViolenceClassifier
 from dataset import CustomDataModule

 cpu_id = 1
 batch_size = 128
 log_name = "resnet18_pretrain"

 data_module = CustomDataModule(batch_size=batch_size)
 ckpt_root = "C:/Users/lenovo/Desktop/violenceDetection/train_logs/"
 ckpt_path = ckpt_root + "resnet18_pretrain_test/version_6/checkpoints/resnet18_pretrain_test-epoch=01-val_loss=0.06.ckpt"
 logger = TensorBoardLogger("test_logs", name=log_name)

 model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
 trainer = Trainer(accelerator='cpu', devices=cpu_id)
 trainer.test(model,data_module)