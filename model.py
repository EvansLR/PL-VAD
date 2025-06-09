import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from Local_Transformer import Local_Transformer
from Global_Transformer import Global_Transformer
import torch.nn.functional as F
# 引入记忆模块
from MemNet import Memory
from Classifiers import ADCLS_head,Temporal


class WSAD(Module):
    def __init__(self, input_size, flag):
        super().__init__()
        self.flag = flag

        # 使用简单的全连接层
        self.linear=nn.Linear(input_size,512)
        self.embedding = Temporal(input_size,512)

        self.cls_head = ADCLS_head(512, 1)


        self.local_selfatt = Local_Transformer(512, 2,4,128,512, dropout = 0.5)
        self.global_selfatt = Global_Transformer(512,2,4,128,512,dropout=0.5)

        self.sig=nn.Sigmoid()
        self.bce_loss=nn.BCELoss()
        self.dropout_rate=0.4
        self.softmax=nn.Softmax(dim=-1)
    
        self.memory_size=60
        self.mem_dim=512
        self.memory = Memory(self.memory_size, self.mem_dim)

        self.Nor_m_items =F.normalize(torch.rand((self.memory_size, self.mem_dim), dtype=torch.float), dim=1).cuda()
        self.Abn_m_items= F.normalize(torch.rand((self.memory_size, self.mem_dim), dtype=torch.float), dim=1).cuda()

        self.twoStageCount = 0
        self.twoStageThreshold = 2000



       
    def caculate_Nor_Abn_Attn(self,fea):
        T = self.memory_size    
     
        top_k_n = T // 16 + 1 
    
        attention1 = self.sig(torch.einsum('btd,kd->btk', fea, self.Nor_m_items) / (self.mem_dim ** 0.5))  # (B, T, K)
       
        N_att = torch.topk(attention1, top_k_n, dim=-1)[0].mean(-1)  # (B, T)
        attention2 = self.sig(torch.einsum('btd,kd->btk', fea, self.Abn_m_items) / (self.mem_dim ** 0.5))  # (B, T, K)
     
        A_att = torch.topk(attention2, top_k_n, dim=-1)[0].mean(-1)  # (B, T)
        if self.flag == "Train" and self.twoStageCount<self.twoStageThreshold:
            return N_att,1-N_att
        return N_att,A_att
    


    

    def get_pseudo_label_and_loss(self, x,N_att,A_att):
        self.twoStageCount+=10
        

        batch_size = x.shape[0]
        Mem_score=A_att
        Nor_x = x[:batch_size//2]
    
        Nor_sim_score_N=self.memory.get_similarity(Nor_x,self.Nor_m_items)
        Normal_loss=self.bce_loss(Nor_sim_score_N,torch.ones_like(Nor_sim_score_N))

        new_Nor_mem = self.memory(Nor_x, self.Nor_m_items)
     
        Abn_x = x[batch_size//2:]
       
        sim_score=self.memory.get_similarity(Abn_x,self.Nor_m_items)
        t_k=Abn_x.size(1)//16 + 1
        tmp_value, tmp_index = torch.topk(sim_score, k=t_k*2, dim = -1, largest=False)
        Abnormal_loss=self.bce_loss(tmp_value,torch.zeros_like(tmp_value))
        self.Nor_m_items = new_Nor_mem


        if self.twoStageCount<self.twoStageThreshold and self.flag == "Train":
            return  Normal_loss+Abnormal_loss,Mem_score
        

        Nor_sim_score_A=self.memory.get_similarity(Nor_x,self.Abn_m_items)
        Normal_loss+=self.bce_loss(Nor_sim_score_A,torch.zeros_like(Nor_sim_score_A))
        sim_score2=self.memory.get_similarity(Abn_x,self.Abn_m_items)
        tmp_value_A, tmp_index_A = torch.topk(sim_score2, k=t_k, dim = -1, largest=True)
        Abnormal_loss+=self.bce_loss(tmp_value_A,torch.ones_like(tmp_value_A))
        new_abn_index=[[i for j in range(0,tmp_index.shape[1])] for i in range(tmp_index.shape[0])]
        tmp_fea=Abn_x[new_abn_index,tmp_index,:]
        new_Abn_mem= self.memory(tmp_fea, self.Abn_m_items)
        self.Abn_m_items = new_Abn_mem
        Mem_loss=(Abnormal_loss+Normal_loss)

        return  Mem_loss,Mem_score




    def diversity_regularization(self,mem_matrix, reg_weight=0.1):
        similarity = F.cosine_similarity(mem_matrix.unsqueeze(1), mem_matrix.unsqueeze(0), dim=-1)
        mask = ~torch.eye(self.memory_size, dtype=torch.bool, device=mem_matrix.device)
        div_loss = torch.mean(similarity[mask])
        return reg_weight * div_loss




    def cos_sim(self, x):
   
        x_padded = torch.cat([x[:, :1, :], x, x[:, -1:, :]], dim=1)

        cos_sim_left = F.cosine_similarity(x, x_padded[:, :-2, :], dim=2)
        cos_sim_right = F.cosine_similarity(x, x_padded[:, 2:, :], dim=2)

        cos_similarities = (cos_sim_left + cos_sim_right)
        
        cos_score=(2-cos_similarities)/4
        return cos_score

    def forward(self, x):



        b= x.size()[0]
     
        embedding_fea=self.embedding(x)   

        e_x1=self.global_selfatt(embedding_fea)
        e_x2=self.local_selfatt(embedding_fea)
        x=torch.cat([e_x2,e_x1],dim=-1)
    
        x=self.linear(x)


        cos_score=self.cos_sim(x)
   
        querys=x
        N_attn,A_attn=self.caculate_Nor_Abn_Attn(querys)

        if self.flag == "Train":
            Mem_loss,Mem_score=self.get_pseudo_label_and_loss(querys,N_attn,A_attn)
            N_Loss=self.diversity_regularization(self.Nor_m_items) 
            A_Loss=self.diversity_regularization(self.Abn_m_items) 
        
            Mem_loss+=(N_Loss + A_Loss)
            A_att=A_attn[b//2:]
            A_Natt=A_attn[:b//2]
            N_att=N_attn[:b//2]
            N_Aatt=N_attn[b//2:]
            pre_att = self.cls_head(x).reshape((b, 1, -1)).mean(1)
            return {
                    "frame": pre_att,
                    'A_att': A_att.reshape((b//2, 1, -1)).mean(1),
                    "N_att": N_att.reshape((b//2, 1, -1)).mean(1),
                    "A_Natt": A_Natt.reshape((b//2, 1, -1)).mean(1),
                    "N_Aatt": N_Aatt.reshape((b//2, 1, -1)).mean(1),
                    "Mem_loss":Mem_loss,
                    'cos_score':cos_score
                }
        

        else:
            pre_att = self.cls_head(x).reshape((b, 1, -1)).mean(1)
            Mem_score=(A_attn+1-N_attn)/2
            original_res=pre_att
            final_res=(Mem_score+pre_att)/2

            return {"frame": final_res,'video': original_res}
    