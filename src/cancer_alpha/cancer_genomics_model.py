# cancer_genomics_model.py

import torch
import torch.nn as nn
from transformers import BertModel

class CancerGenomicsTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(CancerGenomicsTransformer, self).__init__()
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # CLS token
        pooled_output = self.dropout(pooled_output)
        return self.linear(pooled_output)

