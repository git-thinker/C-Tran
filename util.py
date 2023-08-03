import torch
from sklearn.metrics import average_precision_score

class MetricsCentre():
    def __init__(self):
        self.y = []
        self.y_ = []
        
    def log(self, y: torch.Tensor, y_: torch.Tensor):
        self.y.append(y.detach().cpu())
        self.y_.append(y_.detach().cpu())
    
    def wrap_up(self):
        with torch.no_grad():
            self.y = torch.cat(self.y, dim=0).int()
            self.y_ = torch.cat(self.y_, dim=0).float()
    
    def F1s(self, criteria: float = 0.5):
        # OF1
        # OP
        y = self.y.int()
        pre = (self.y_ > criteria).int()
        OP = (y & pre).sum() / pre.sum()
        OR = (y & pre).sum() / y.sum()
        OF1 = 2 * OP * OR / (OP + OR)

        # CF1
        CP = ((y & pre).sum(dim=0) / pre.sum(dim=0)).mean()
        CR = ((y & pre).sum(dim=0) / y.sum(dim=0)).mean()
        CF1 = 2 * CP * CR / (CP + CR)
        return {
            'OP': OP,
            'OR': OR,
            'OF1': OF1,
            'CP': CP,
            'CR': CR,
            'CF1': CF1
        }
    
    def F1sTopk(self, criteria: float = 0.5, k: int = None):
        topk_indices = torch.topk(self.y_, k=k, dim=-1).indices
        mask = torch.zeros_like(self.y_).scatter_(-1, topk_indices, 1)
        pre = torch.where(mask > 0, self.y_, torch.tensor(-1, dtype=self.y_.dtype))
        y = torch.where(mask > 0, self.y, torch.tensor(0, dtype=self.y.dtype)).int()
        pre = (pre > criteria).int()
        
        OP = (y & pre).sum() / pre.sum()
        OR = (y & pre).sum() / y.sum()
        OF1 = 2 * OP * OR / (OP + OR)

        # CF1
        CP = ((y & pre).sum(dim=0) / pre.sum(dim=0)).mean()
        CR = ((y & pre).sum(dim=0) / y.sum(dim=0)).mean()
        CF1 = 2 * CP * CR / (CP + CR)
        return {
            f'top{k}_OP': OP,
            f'top{k}_OR': OR,
            f'top{k}_OF1': OF1,
            f'top{k}_CP': CP,
            f'top{k}_CR': CR,
            f'top{k}_CF1': CF1
        }

    def ap(self):
        ap_scores = []
        for i in range(self.y.shape[1]):
            ap = average_precision_score(self.y[:, i], self.y_[:, i])
            ap_scores.append(ap)
        return ap_scores

    def mAP(self):
        ap_scores = self.ap()
        mAP = torch.tensor(ap_scores).mean().item()
        return mAP

    def evaluate(self, criteria: float = 0.5, k: int = 3, ap=False):
        self.wrap_up()
        ret = {'k': k}
        ret |= self.F1s(criteria=criteria)
        ret |= self.F1sTopk(criteria=criteria, k=k)
        if ap:
            ret['AP'] = self.ap()
        ret['mAP'] = self.mAP()
        return ret
    
    def clear(self):
        self.y = []
        self.y_ = []

