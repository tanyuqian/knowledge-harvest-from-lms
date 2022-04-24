
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch

class LAMAKnowledgeScorer(torch.nn.Module):
    def __init__(self, plm) -> None:
        super().__init__()
        if "roberta" in plm:
            self.encoder = RobertaModel.from_pretrained("roberta-large").to("cuda")
            self.classifier = torch.nn.Linear(1024, 1).to("cuda")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        else:
            self.encoder = BertModel.from_pretrained("bert-base-uncased").to("cuda")
            self.classifier = torch.nn.Linear(768, 1).to("cuda")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        '''
        self.eval()
    def forward(self, inputs):
        outputs = self.encoder(**inputs)
        logits = self.classifier(outputs[0][:, 0, :])  # (batch_size, emb_size)
        # outputs[0] 32, 24, 1024
        return logits
        # return outputs
    def score(self, h, r, t):
        return self.score_batch([(h, r, t)])[0]
    
    def score_batch(self, triples):
        inputs = self.tokenizer([h + " | " + r + " | " + t for h, r, t in triples], return_tensors='pt', padding=True)
        # labels = torch.tensor([int(_) for (r, h, t), _ in batch])
        logits = self.forward(inputs.to("cuda"))  # (n)
        scores = torch.sigmoid(logits) # (n)
        return scores
