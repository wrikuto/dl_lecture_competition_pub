import torch.nn as nn
import torch
from .resnet import ResNet18

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()

        # テキストエンコーダーはBERTのエンベディングを受け取るので512次元
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 512),  # BERTの出力次元は768
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question_embedding):
        image_feature = self.resnet(image)  # 画像の特徴量

        # 画像特徴と質問文のエンベディングを結合
        x = torch.cat([image_feature, question_embedding], dim=1)
        x = self.fc(x)
        return x
