import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import torch
from statistics import mode
from utils.text_processing import process_text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer

        # BERTのトークナイザーとモデルの初期化
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.answer2idx = {}
        self.idx2answer = {}
        self.question2idx = {}
        self.idx2question = {}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        # 質問文の前処理とトークナイズ
        question_text = process_text(self.df["question"][idx])
        question_tokens = self.tokenizer(question_text, padding='max_length', max_length=32, return_tensors="pt")

        # BERTを用いてエンベディングを取得
        with torch.no_grad():
            question_embedding = self.bert_model(**question_tokens).last_hidden_state.mean(dim=1).squeeze()

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, question_embedding, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, question_embedding

    def __len__(self):
        return len(self.df)
