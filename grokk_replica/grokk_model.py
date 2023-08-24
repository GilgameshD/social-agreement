'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2023-08-23 00:39:47
LastEditTime: 2023-08-23 13:49:02
Description: 
'''

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from grokk_replica.transformer import Transformer
from grokk_replica.utils import causal_attn_mask, parameter_norm


class GrokkModel(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device):
        super(GrokkModel, self).__init__()
        num_model = 4

        self.transformers = nn.ModuleList()
        for m_i in range(num_model):
            self.transformers.append(Transformer(**transformer_config, vocab_size=vocab_size, output_size=output_size))
        self.device = device
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    def single_model_forward(self, x, m_i):
        attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)
        prediction, _, _ = self.transformers[m_i](x, attn_mask)
        return prediction

    def forward(self, x):
        predictions = []
        for m_i in range(len(self.transformers)):
            attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)
            prediction, _, _ = self.transformers[m_i](x, attn_mask)
            predictions.append(prediction)
        return predictions
    
    def get_loss(self, x, y, match_data):
        predictions = self(x)
        loss_list = []
        accuracy_list = []
        for p_i in range(len(predictions)):
            loss = F.cross_entropy(predictions[p_i][:, -1, :], y)
            accuracy = (torch.argmax(predictions[p_i][:, -1, :], dim=-1) == y).float().mean()
            loss_list.append(loss)
            accuracy_list.append(accuracy)
        ce_loss = sum(loss_list) / len(loss_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)

        num_match = 1
        dataloader = DataLoader(match_data, num_workers=0, batch_size=num_match)
        data = next(iter(dataloader))

        # sample two models
        models_idx = np.random.choice(len(self.transformers), 2, replace=False)
        pred_1 = self.single_model_forward(x, models_idx[0])
        pred_2 = self.single_model_forward(x, models_idx[1])

        # convert to log space
        pred_1 = F.log_softmax(pred_1, dim=1)
        pred_2 = F.log_softmax(pred_2, dim=1)
        kld = self.kl_loss(pred_1, pred_2) + self.kl_loss(pred_2, pred_1)
        loss = ce_loss + kld * 0.1

        log_dict = {
            'kld': (kld.item(), x.shape[0]),
            'loss': (loss.item(), x.shape[0]), 
            'ce_loss': (ce_loss.item(), x.shape[0]),
            'accuracy': (accuracy.item(), x.shape[0]), 
        }
        return loss, log_dict
