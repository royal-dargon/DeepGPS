import re
import pandas as pd
import torch
import torch.nn as nn
import sys
import pickle
from typing import List, Union, Dict, Tuple, Optional

import esm

'''
ESM-2 model parameters at different scales:
                          8M      35M      150M      650M     3B      15B
Number of layers          6       12       30        33       36      48
Embedding dim             320     480      640       1280     2560    5120
Attention heads           20      20       20        20       40      40
Training steps            500K    500K     500K      500K     500K    270K
Learning rate             4e-4    4e-4     4e-4      4e-4     4e-4    1.6e-4
Weight decay              0.01    0.01     0.01      0.01     0.01    0.1
Clip norm                 0       0        0         0        1.0     1.0
Distributed backend       DDP     DDP      DDP       DDP      FSDP    FSDP

8M: esm2_t6_8M_UR50D()
35M: esm2_t12_35M_UR50D()
150M: esm2_t30_150M_UR50D()
650M: esm2_t33_650M_UR50D()
3B: esm2_t36_3B_UR50D()
15B: esm2_t48_15B_UR50D()
'''

class ESM2(nn.Module):
    def __init__(self):
        super(ESM2, self).__init__()
        self.model, _ = esm.pretrained.esm2_t6_8M_UR50D()

    '''
    x_seq: tokenized后的氨基酸序列
    '''
    def forward(self, x_seq) -> torch.Tensor:
        self.model.eval()

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(x_seq, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]

        return token_representations

if __name__ == "__main__":
    test_model = ESM2()

    # 将模型移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = test_model.to(device)

    # 多个GPU并行处理
    if torch.cuda.device_count() > 1:
        print("使用多个GPU进行并行处理...")
        test_model = nn.DataParallel(test_model)
    
    with open("ESM2_id_to_seq_list_se.pkl", 'rb') as f:
        data = pickle.load(f)

    # 将输入数据移动到GPU上
    # input = torch.tensor([[0,  5, 25, 23, 2, 1, 1]])    # "ABC"
    input = torch.zeros((1, 2700))
    input[0] = data[0].squeeze()
    # input[1] = data[1].squeeze()
    
    input = input.to(torch.long)
    input = input.to(device)
    
    print(input)
    print(input.shape) #(2, 2700)
    
    res = test_model(input)
    
    print(res)
    print(res.shape)
