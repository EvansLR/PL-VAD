import torch
import torch.nn as nn

def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
    def forward(self, result, _label):

        loss = {}

        _label = _label.float()

     
        att = result['frame']
       
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        cos_score=result["cos_score"]
        Mem_loss=result["Mem_loss"]
        

        b = _label.size(0)//2
        t = att.size(1)      
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        mem_score=((1-N_Aatt)+A_att)/2
        att_abn=att[att.shape[0]//2:]

        p_loss = self.bce(att_abn, mem_score.detach())
        anomaly_loss = self.bce(anomaly, _label)
        cos_anomaly = torch.topk(cos_score, t//16 + 1, dim=-1)[0].mean(-1)
        cos_loss = self.bce(cos_anomaly,_label)
        cost = anomaly_loss+1e-2*p_loss+1e-3*Mem_loss+1e-3*cos_loss
      
        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['Mem_loss'] = Mem_loss
        loss["p_loss"] = p_loss
        loss["cos_loss"] =cos_loss
       
        return cost, loss



def train(net, normal_loader, abnormal_loader, optimizer, criterion,index):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    cost, loss = criterion(predict, _label)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()