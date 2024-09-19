import torch
import torch.nn as nn
from typing import List, Union, Dict, Tuple, Optional
from transformers.configuration_utils import PretrainedConfig

class VoTConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=320,#640, 768,
        num_hidden_layers=6,#12,
        num_attention_heads=8,#12,
        intermediate_size=2560,#3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        voxel_size=200,
        block_size=20,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride

class ModelConfig():
    def __init__(
        self,
        voxel_encode: str = "simple",           # 选择体素编码器类型
        seq_pretrain: bool = False,        # 是否在训练中使用预训练模型
        seq_encode: str = "esm_2",           # 选择氨基酸序列编码器类型
        combine: str = "attention",           # 选择输入数据结合类型
        classify: str = "fusion",            # 选择体素编码器类型
        loc_predict: str = "unet_add",       # 选择体素编码器类型

        hid_size: int = 512,                 # 隐向量维度
        seq_pad_id: int = 1,                 # 氨基酸序列编码器的pad序号
        voxel_size: int = 200,               # 体素的尺寸
        block_size: int = 20,                # 块的尺寸
        vox_channels: int = 3,               # 体素数据通道数
        num_labels: int = 8,                 # 多标签分类的标签数量
        dropout: float = 0.1,                # dropout的比例
        init_factor: float = 1.0,            # 正太分布参数初始化的方差调整因子
        predict_label: bool = True,         # 是否预测多标签
        in_channels: int = 1,                # 输入图像的通道数
        out_channels: int = 1,               # 输出位置分布图的通道数
        label_loss: str = "bce",             # 多标签分类的损失函数设置
        image_loss: str = "mse",             # 图像重塑的损失函数设置
    ) -> None:
        # 模型各模块选择
        self.voxel_encode = voxel_encode
        self.seq_encode = seq_encode
        self.seq_pretrain = seq_pretrain
        self.combine = combine
        self.classify = classify
        self.loc_predict = loc_predict
        
        # 其他参数
        self.hid_size = hid_size
        self.seq_pad_id = seq_pad_id
        self.num_labels = num_labels
        self.dropout = dropout
        self.init_factor = init_factor
        self.predict_label = predict_label
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_loss = label_loss
        self.image_loss = image_loss