"""
this is the main model about all componets
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# from model_config import ModelConfig

sys.path.append("/home/bingxing2/ailab/scxlab0065/Predict_Protein_Subcellular_Localization/exp2/model")
from seq_encode_models.esm_2 import ESM2
from structure_encode_models.pointnet import PointNetfeat
from structure_encode_models.pointnet2 import PointNet2Extract
from image_creat_models.unet_add import UnetAdd


class SeqEncode(nn.Module):
    def __init__(self, config=None) -> None:
        super(SeqEncode, self).__init__()
        # can change the sequence model
        self.seq_encode = ESM2()
    
    def forward(self, x_seq=None, seq_mask=None):
        """
        x_seq.shape : batch, length (N L)
        seq_mask.shape : batch, length (N L)

        seq_hid_states.shape : batch, length, dimension (N L d)
        """
        seq_hid_states = self.seq_encode(x_seq)
        return seq_hid_states


class StructureEncode(nn.Module):
    def __init__(self, config) -> None:
        super(StructureEncode, self).__init__()
        if config.voxel_encode == "simple":
            self.voxel_encode = PointNetfeat()
        elif config.voxel_encode == "pointnet2":
            self.voxel_encode = PointNet2Extract(normal_channel=True)
    def forward(self, x_vox):
        x_vox = x_vox.permute(0, 2, 1)
        vox_hid_states, _, _ = self.voxel_encode(x_vox)
        # torch.Size([16, 1024]) point net
        return vox_hid_states


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.multi_attn = nn.MultiheadAttention(embed_dim=320, num_heads=16, batch_first=True)
        self.linear1 = nn.Linear(320, 320)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(320, 320)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(320)
        self.norm2 = nn.LayerNorm(320)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, k_v, q):
        k_v = self.norm1(k_v)
        q = self.norm1(q)
        attn_out, _ = self.multi_attn(q, k_v, k_v)
        attn_out = self.dropout(attn_out)
        attn_out = q + attn_out
        x = attn_out
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        attn_out = attn_out + x
        return attn_out


class Combine(nn.Module):
    def __init__(self, config=None) -> None:
        super(Combine, self).__init__()
        self.combine = CrossAttention()

    def forward(self, seq_hid, structer_hid):
        # hid_states = torch.cat((seq_hid, structer_hid), dim=1)
        hid_states = self.combine(seq_hid, structer_hid)
        return hid_states


class EasyClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size) -> None:
        super(EasyClassifier, self).__init__()
        self.layers = nn.ModuleList()
        # 添加第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # 添加更多隐藏层
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        # 添加输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        # print(x.shape)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        x = F.softmax(x, dim=1)
        return x


class Classify(nn.Module):
    def __init__(self, config=None) -> None:
        super(Classify, self).__init__()
        if config.classify == "easy_mlp":
            # 修改类别需要注意修改
            self.classify = EasyClassifier(input_size=1024, hidden_sizes=[512, 256], output_size=2)

        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, hid_states, y_lab):
        res = self.classify(hid_states)
        y_lab = y_lab.float()
        # res = res.squeeze(dim=0)
        loss = self.loss_func(res, y_lab)
        return loss, res
    

class ImageCreat(nn.Module):
    def __init__(self, config=None) -> None:
        super(ImageCreat, self).__init__()
        if config.loc_predict == "unet_add":
            self.loc_predict = UnetAdd(config)

    def forward(self, hid_states, x_img, y_img, point_hid=None):
        loss, logits = self.loc_predict(hid_states, x_img, y_img)
        return loss, logits


class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        if config.seq_pretrain == True:
            self.seq_encode = SeqEncode(config)
        self.structer_encode = StructureEncode(config)
        self.combine = Combine(config)
        self.classify = Classify(config)
        self.image_creater = ImageCreat(config)

        self.predict_label = config.predict_label
        self.seq_pad_id = config.seq_pad_id
        self.hid_size = config.hid_size
        self.seq_pretrain = config.seq_pretrain

        self.linear = nn.Linear(in_features=1280, out_features=1024)

    def forward(self, x_seq, x_img, y_lab, y_img):
        # config.seq_pretrain == True时，需要这行代码
        # x_seq = x_seq.type(torch.long)
        # x_structer = x_structer.type(torch.float)
        # y_lab = y_lab.type(torch.long)
        seq_mask = (x_seq != 1.0).type(torch.float).unsqueeze(-1)
        # if self.seq_pretrain:
        #     seq_hid_features = self.seq_encode(x_seq)
        #     seq_hid_features = seq_hid_features * seq_mask
        # elif not self.seq_pretrain:
        #     seq_hid_features = x_seq.squeeze(dim = 1) # N 1 L D -> N L D

        # structer_hid_states = self.structer_encode(x_structer)
        # hid_states = self.combine(seq_hid_features, structer_hid_states)
        x_seq = F.relu(self.linear(x_seq))
        res_img = self.image_creater(x_seq, x_img, y_img)
        loss = res_img[0]

        if self.predict_label:
            loss_cl, res_cl = self.classify(x_seq, y_lab)
            loss += loss_cl
            return loss, res_cl, res_img[1]
    
        return loss, None, res_img[1]
    

class AllModel(nn.Module):
    def __init__(self, config):
        super(AllModel, self).__init__()
        if config.seq_pretrain == True:
            self.seq_encode = SeqEncode(config)
        self.structer_encode = StructureEncode(config)
        self.combine = Combine(config)
        self.classify = Classify(config)
        self.image_creater = ImageCreat(config)

        self.predict_label = config.predict_label
        self.seq_pad_id = config.seq_pad_id
        self.hid_size = config.hid_size
        self.seq_pretrain = config.seq_pretrain

        self.linear = nn.Linear(in_features=1280, out_features=1024)

    def forward(self, x_seq, x_img, y_lab=None, y_img=None, p=None):
        # config.seq_pretrain == True时，需要这行代码
        # x_seq = x_seq.type(torch.long)
        # x_structer = x_structer.type(torch.float)
        # y_lab = y_lab.type(torch.long)
        # seq_mask = (x_seq != 1.0).type(torch.float).unsqueeze(-1)
        # if self.seq_pretrain:
        #     seq_hid_features = self.seq_encode(x_seq)
        #     seq_hid_features = seq_hid_features * seq_mask
        # elif not self.seq_pretrain:
        #     seq_hid_features = x_seq.squeeze(dim = 1) # N 1 L D -> N L D

        # structer_hid_states = self.structer_encode(x_structer)
        # print(structer_hid_states.shape, seq_hid_features.shape)
        # hid_states = self.combine(seq_hid_features, structer_hid_states)
        
        x_seq = F.relu(self.linear(x_seq)).unsqueeze(dim=1)
        x_point = self.structer_encode(p).unsqueeze(dim=1)
        # torch.Size([16, 1024]) x_seq
        # torch.Size([16, 1024]) x_point
        x_seq = torch.cat((x_seq, x_point), dim=1)
        # print(x_seq.shape, "x_seq")
        res_img = self.image_creater(x_seq, x_img, y_img)
        loss = res_img[0]
        # print(res_img[1].shape, "out")

        if self.predict_label:
            loss_cl, res_cl = self.classify(x_seq, y_lab)
            loss += loss_cl
            return loss, res_cl, res_img[1]
    
        return loss, None, res_img[1]
    
    
class PointModel(nn.Module):
    def __init__(self, config):
        super(PointModel, self).__init__()
        # if config.seq_pretrain == True:
        #     self.seq_encode = SeqEncode(config)
        self.structer_encode = StructureEncode(config)
        self.combine = Combine(config)
        self.classify = Classify(config)
        self.image_creater = ImageCreat(config)

        self.predict_label = config.predict_label
        self.seq_pad_id = config.seq_pad_id
        self.hid_size = config.hid_size
        self.seq_pretrain = config.seq_pretrain

        self.linear = nn.Linear(in_features=1280, out_features=1024)

    def forward(self, x_img, y_lab=None, y_img=None, p=None):
        # x_seq = F.relu(self.linear(x_seq)).unsqueeze(dim=1)
        # y_lab = y_lab.type(torch.long)
        x_point = self.structer_encode(p).unsqueeze(dim=1)
        # x_seq = torch.cat((x_seq, x_point), dim=1)
        # print(x_seq.shape, "x_seq")
        res_img = self.image_creater(x_point, x_img, y_img)
        loss = res_img[0]
        # print(res_img[1].shape, "out")

        if self.predict_label:
            loss_cl, res_cl = self.classify(x_point.squeeze(dim=1), y_lab)
            loss += loss_cl
            return loss, res_cl, res_img[1]
    
        return loss, None, res_img[1]
