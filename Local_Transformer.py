import torch
from torch import nn
from einops import rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # 自定义权重：
        self.sigma_projection = nn.Linear(dim,heads)




    def forward(self, x):
        b,n,d=x.size()
        tmp_ones = torch.ones(n).cuda()
        tmp_n = torch.linspace(1, n, n).cuda()
        distances = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1,1))
        sigma=self.sigma_projection(x)
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        sigma = torch.sigmoid(sigma)
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, n)  # B H L L


        prior = distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()


        # softmax
        prior = -prior/(torch.exp(sigma)+1)
        prior = self.attend(prior)




        v = self.to_v(x) 
        # value= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [v])[0]
        v=rearrange(v,'b n (h d) -> b h n d', h = self.heads)
        out =torch.matmul(prior,v)
        out = rearrange(out, 'b h n d -> b n (h d)',h=self.heads)

        return self.to_out(out)

class Local_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x