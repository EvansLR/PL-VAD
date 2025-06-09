import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F


class Memory(nn.Module):

    def __init__(self, memory_size, feature_dim):
        super(Memory, self).__init__()
    
        self.memory_size = memory_size
     
        self.feature_dim = feature_dim
   
        self.key_dim = feature_dim
       
        self.temp_update =1e-5

        self.temp_gather =1e-5
      
        self.sig = nn.Sigmoid()
        
        self.loss = torch.nn.TripletMarginLoss(margin=1.0)
        self.loss_mse = torch.nn.MSELoss()
      
        self.query_linear = nn.Linear(self.feature_dim, self.key_dim)
        self.key_linear = nn.Linear(self.feature_dim, self.key_dim)

    def get_update_query(self, mem, max_indices, update_indices, score, query):
      
        m, d = mem.size()
     
        query_update = torch.zeros((m,d)).cuda()
   
        for i in range(m):
         
            idx = torch.nonzero(max_indices.squeeze(1)==i)
            a, _ = idx.size()
            if a != 0:
                query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
            else:
                query_update[i] = 0 
        return query_update
    
    def get_score(self, mem, query):
        bs, t_seg,d = query.size()
  
        m, d = mem.size()
  
        score = torch.matmul(query, torch.t(mem))
      
        score = score.view(bs*t_seg, m)
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)
        return score_query, score_memory


    def get_similarity(self,query,mem):
    
        similarity_score =self.sig(torch.matmul(query, torch.t(mem)) / (self.key_dim**0.5))
        similarity_score_top = torch.topk(similarity_score,self.memory_size//16+1, dim = -1)[0].mean(-1)
        return similarity_score_top


    def update(self, query, keys):
   
        batch_size,t_seg,dims = query.size() 
   
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*t_seg, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape)
        updated_memory = F.normalize(self.temp_update*query_update + keys, dim=1)
        return updated_memory.detach()

    def gather_loss(self,query, keys):

        batch_size, t_seg,dims = query.size() 
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*t_seg, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, self.memory_size//16+2, dim=1)
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,self.memory_size//16+1]]
        top1_loss = self.loss_mse(query_reshape, pos.detach())
        gathering_loss =self.loss(query_reshape,pos.detach(), neg.detach())
        return gathering_loss, top1_loss




    
    def forward(self, query, keys):
        query = F.normalize(query, dim=2)
        updated_memory = self.update(query, keys)
        return  updated_memory
