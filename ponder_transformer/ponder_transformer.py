import sys
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# constants
ABS_MAX_STEPS = 100

# helper functions
def exists(val):
    return val is not None

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


#次元数をdim→(dim*mult)→dimに変換する
def FeedForward(dim, mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )


#Multi-head Attention
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,#headでの次元数
        heads=8,
        causal=False #TODO　何?
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5 #分散を1にするためにスケーリング
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask=None):#x:torch.Size([2, 512, 512])
        n, h, device = x.shape[1], self.heads, x.device 
        qkv = self.to_qkv(x).chunk(3, dim=-1) #q,k,vの塊のリスト作成 torch.Size([2, 512, 512])×3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)#torch.Size([2, 8, 512, 64]) head×単語数×次元数
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale #QK/√d

        mask_value = -torch.finfo(sim.dtype).max

        #学習時、次の単語を予測する際にはそれより先の情報を使えないように
        if exists(mask):
            mask = rearrange(mask, 'b i -> b () i ()') * \
                rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(
                (i, j), device=device).triu(j - i + 1).bool()
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)#torch.Size([2, 8, 512, 64])
        out = rearrange(out, 'b h n d -> b n (h d)')#torch.Size([2, 512, 512]) multi-headの情報を統合
        return self.to_out(out)

# pondering classes and helper functions
#一番左に1をpaddingする
def pad_to(t, padding, dim=-1, value=0.):
    print(dim,value)
    
    if dim > 0:
        dim = dim - t.ndim
    zeroes = -dim - 1#(a,b,c,d)において、各次元の番号(0,1,2,3)は(3,2,1,0)に変換される zerosには指定した次元の変換後の要素が代入される
    return F.pad(t, (*((0, 0) * zeroes), *padding), value=value)#padding:(左にpaddingする個数,右にpaddingする個数)


def exclusive_cumprod(t, dim=-1):
    cum_prod = t.cumprod(dim=dim)
    """
    t=
    tensor([0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000,
        0.8000])
        
    cum_prod = 0.8,0.8^2,....
    tensor([8.0000e-01, 6.4000e-01, 5.1200e-01, 4.0960e-01, 3.2768e-01, 2.6214e-01,
        2.0972e-01, 1.6777e-01, 1.3422e-01, 1.0737e-01, 8.5899e-02, 6.8719e-02,
        5.4976e-02, 4.3980e-02, 3.5184e-02, 2.8148e-02, 2.2518e-02, 1.8014e-02,
        1.4412e-02, 1.1529e-02, 9.2234e-03, 7.3787e-03, 5.9030e-03, 4.7224e-03,
        3.7779e-03, 3.0223e-03, 2.4179e-03, 1.9343e-03, 1.5474e-03, 1.2379e-03,
        9.9035e-04, 7.9228e-04, 6.3383e-04, 5.0706e-04, 4.0565e-04, 3.2452e-04,
        2.5961e-04, 2.0769e-04, 1.6615e-04, 1.3292e-04, 1.0634e-04, 8.5071e-05,
        6.8057e-05, 5.4445e-05, 4.3556e-05, 3.4845e-05, 2.7876e-05, 2.2301e-05,
        1.7841e-05, 1.4272e-05, 1.1418e-05, 9.1344e-06, 7.3075e-06, 5.8460e-06,
        4.6768e-06, 3.7414e-06, 2.9932e-06, 2.3945e-06, 1.9156e-06, 1.5325e-06,
        1.2260e-06, 9.8080e-07, 7.8464e-07, 6.2771e-07, 5.0217e-07, 4.0173e-07,
        3.2139e-07, 2.5711e-07, 2.0569e-07, 1.6455e-07, 1.3164e-07, 1.0531e-07,
        8.4250e-08, 6.7400e-08, 5.3920e-08, 4.3136e-08, 3.4509e-08, 2.7607e-08,
        2.2086e-08, 1.7668e-08, 1.4135e-08, 1.1308e-08, 9.0463e-09, 7.2370e-09,
        5.7896e-09, 4.6317e-09, 3.7054e-09, 2.9643e-09, 2.3714e-09, 1.8971e-09,
        1.5177e-09, 1.2142e-09, 9.7134e-10, 7.7707e-10, 6.2165e-10, 4.9732e-10,
        3.9786e-10, 3.1829e-10, 2.5463e-10, 2.0370e-10])
    """
    return pad_to(cum_prod, (1, -1), value=1., dim=dim)


def calc_geometric(l, dim=-1):
    #l:要素全て(λ_p),xAX_STEPS個のテンソル
    """
    tensor([1.0000, 0.8000, 0.6400, 0.5120, 0.4096, 0.3277, 0.2621, 0.2097, 0.1678,
        0.1342, 0.1074, 0.0859, 0.0687])
    ×
    tensor([0.2000, 0.2000, 0.2000, ....
    """
    return exclusive_cumprod(1 - l, dim=dim) * l
    


# main class
class Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        causal=False,
        ff_mult=4
    ):
        super().__init__()
        self.causal = causal
        self.attn = PreNorm(dim, Attention(
            dim=dim, dim_head=dim_head, heads=heads, causal=causal))
        self.ff = PreNorm(dim, FeedForward(dim=dim, mult=ff_mult))

        self.to_halt_logits = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...')
        )

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask) + x#TODO なぜ足す? +いるの? 
        x = self.ff(x) + x

        if self.causal:
            denom = torch.arange(x.shape[-2], device=x.device)
            denom = rearrange(denom, 'n -> () n ()')
            halt_input = x.cumsum(dim=1) / (denom + 1)
        else:
            halt_input = x.mean(dim=1)

        halt_logits = self.to_halt_logits(halt_input)

        return x, halt_logits


class PonderTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,#語彙数
        dim,#各トークンの次元数
        max_seq_len,
        causal=True,
        dim_head=64,
        heads=8,
        ponder_kl_div_loss_weight=0.01,
        ponder_lambda_p=0.2,
        ponder_epsilon=0.05,
        eps=1e-20
    ):
        super().__init__()
        self.eps = eps
        self.causal = causal
        self.seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # calculate max steps

        thres = 1 - ponder_epsilon#閾値
        halt_probs = calc_geometric(torch.full( #停止確率のリスト
            (ABS_MAX_STEPS,), ponder_lambda_p))
        
        cum_halt_probs = halt_probs.cumsum(dim=0) #停止確率の累積和
        self.train_max_steps = (cum_halt_probs < thres).sum().item()#累積和が1を超えないように、上限ステップ数を定義 

        self.ponder_lambda_p = ponder_lambda_p
        self.ponder_kl_div_loss_weight = ponder_kl_div_loss_weight

        # pondering block

        self.block = Block(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            causal=causal
        )

        # Decoder部に相当
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, *, labels=None, mask=None):
        n, device, eps, max_steps, causal = x.shape[1], x.device, self.eps, self.train_max_steps, self.causal
        x = self.token_emb(x)#torch.Size([2, 512])->torch.Size([2, 512, 512]) 2×トークン×次元数
        pos_emb = self.pos_emb(torch.arange(n, device=device))#厳密なpositional embedingではない?
        x = x + rearrange(pos_emb, 'n d -> () n d')#埋め込み+位置情報

        if self.training:
            assert exists(labels), 'labels must be passed in during training'

            hiddens = []#各ponder step での h のリスト
            halting_logits = []#各ponder step での λ のリスト

            # training mode
            for _ in range(max_steps):
                x, halt_logits = self.block(x)#h,λ
                hiddens.append(x)
                halting_logits.append(halt_logits)

            # stack halting probs (lambda) and y
            halting_logits = torch.stack(halting_logits, dim=1)
            halting_probs = calc_geometric(halting_logits.sigmoid(), dim=1)#停止確率分布を計算, torch.Size([2, 上限ステップ数, 512])
            hiddens = torch.stack(hiddens, dim=1)#ただの変形
            logits = self.to_logits(hiddens)

            # 事前確立分布を計算 torch.Size([上限ステップ数])
            geometric_dist = calc_geometric(torch.full(
                (max_steps,), self.ponder_lambda_p, device=device))

            print(geometric_dist)
            sys.exit()
            if self.causal:
                geometric_dist = repeat(geometric_dist, 'l -> (l n)', n=n)
                halting_probs = rearrange(
                    halting_probs, '... l n -> ... (l n)')
            kl_div_loss = F.kl_div(
                torch.log(geometric_dist + eps),
                halting_probs,
                None, None,
                'batchmean'
            )

            # calculate cross entropy loss
            labels = repeat(labels, 'b n -> b (l n)', l=max_steps)#torch.Size([2, 512])->torch.Size([2, 6656])
            logits = rearrange(logits, 'b l n d -> b d (l n)')#torch.Size([2, 13, 512, 20000])->torch.Size([2, 20000, 6656])
            ce_loss = F.cross_entropy(logits, labels, ignore_index=0)#softmax+nnllogなので、torch.Size([2, 20000, 6656])はsoftmaxでtorch.Size([2, 6656])になる
            weighted_ce_loss = ce_loss * halting_probs#L(y',y)×p_n
            
            #TODO　違う
            """
            print(ce_loss)
            print(halting_probs)
            print(weighted_ce_loss)
            tensor(10.0644, grad_fn=<NllLoss2DBackward>)
            tensor([[2.2834e-01, 2.9240e-01, 2.6229e-01,  ..., 7.9947e-04, 7.9454e-04,
                    7.9414e-04],
                    [5.4808e-01, 4.7401e-01, 4.8237e-01,  ..., 4.5369e-04, 4.5267e-04,
                    4.5260e-04]], grad_fn=<ViewBackward>)
            tensor([[2.2981e+00, 2.9429e+00, 2.6398e+00,  ..., 8.0462e-03, 7.9966e-03,
                    7.9926e-03],
                    [5.5161e+00, 4.7707e+00, 4.8548e+00,  ..., 4.5661e-03, 4.5558e-03,
                    4.5551e-03]], grad_fn=<MulBackward0>)
            """
            
            # sum loss
            loss = weighted_ce_loss.mean() + self.ponder_kl_div_loss_weight * kl_div_loss.mean()
            return loss
        else:
            # evaluation mode

            hiddens = []#各ponder step での h のリスト
            halting_logits = []#各ponder step での λ のリスト
            layer_halt = []#[ストップするかどうかのbool値のリスト]のリスト

            for i in range(self.train_max_steps):
                is_last = i == (self.train_max_steps - 1)

                x, halt_logits = self.block(x)
                hiddens.append(x)

                if self.causal:
                    halt_logits = halt_logits[..., -1]

                halting_logits.append(halt_logits)

                # calculating halting probs

                halting_probs = torch.stack(halting_logits, dim=1).sigmoid()
                p = calc_geometric(halting_probs, dim=1)[:, -1]#データの数×停止確率を計算
                should_halt = torch.rand_like(p) <= p  #同じshape、ランダムの確率が要素→bool値のリスト

                # stack the halting signal across layers and determine whether to stop early
                layer_halt.append(should_halt)
                # do not exit early if it is the last one

                if is_last:
                    continue

                # break if halting has been sampled for all layers

                layer_was_halted = torch.any(torch.stack(layer_halt), dim=0)

                if torch.all(layer_was_halted):
                    break

            # calculate max number of layers

            max_num_layers = len(layer_halt)

            # stack the hiddens and the boolean tensor indicating halting for each layer

            hiddens = torch.stack(hiddens, dim=1)
            layer_halt = torch.stack(layer_halt, dim=1)

            # calculate the index of the first halt signal, and make it the last layer if none of them halted

            halt_layer_indices = (layer_halt.cumsum(dim=1) == 0).sum(
                dim=1).clamp(max=max_num_layers - 1)

            # select out the correct hidden layers to logits

            halt_layer_indices_expanded = repeat(
                halt_layer_indices, 'b -> b () n d', n=hiddens.shape[-2], d=hiddens.shape[-1])
            hiddens = hiddens.gather(1, halt_layer_indices_expanded)
            hiddens = rearrange(hiddens, 'b () n d -> b n d')

            return self.to_logits(hiddens), halt_layer_indices



def main():
    model = PonderTransformer(
        num_tokens = 20000,
        dim = 512,
        max_seq_len = 512
    )

    mask = torch.ones(2, 512).bool()
    x = torch.randint(0, 20000, (2, 512))#torch.Size([2, 512]),データ数=2,単語ID×512のシーケンスデータ
    y = torch.randint(0, 20000, (2, 512))#torch.Size([2, 512])
    loss = model(x, labels = y, mask = mask)
    loss.backward()
    
    x = torch.randint(0, 20000, (2, 512))
    mask = torch.ones(2, 512).bool()
    model.eval() # setting to eval makes it return the logits as well as the halting indices
    logits, layer_indices = model(x,  mask = mask) # (2, 512, 20000), (2)
    #print(f"logits:{logits},layer_indices:{layer_indices}")
    """
    logits:tensor([[[ 0.2034,  0.2681,  0.2987,  ...,  0.3540,  0.8200, -0.1811],
         [-0.2441,  0.7420,  0.2854,  ..., -1.2270,  0.5369,  0.1349],
         [-0.7548, -0.1693, -0.1432,  ...,  0.6816,  0.7811, -0.6181],
         ...,
         [-0.2387, -0.5372,  0.2591,  ..., -0.0798, -0.1982,  0.1153],
         [-0.7237,  0.3258,  0.5387,  ..., -0.5340,  0.7234,  1.0987],
         [-0.3054, -0.1425,  0.2786,  ...,  0.6499,  0.5277, -0.0944]],

        [[-0.0688,  0.1086, -0.0593,  ...,  0.0345,  0.5378,  0.2081],
         [ 1.2130,  0.5203,  0.2475,  ..., -0.7494,  0.2491,  0.2153],
         [-0.4355, -0.0700, -0.1908,  ...,  0.3313,  0.4661, -0.2633],
         ...,
         [-0.2692, -0.5628,  0.0895,  ...,  0.0712, -0.1872,  0.4158],
         [ 0.0582,  0.8416,  0.5380,  ..., -0.0096, -0.3193,  0.0355],
         [ 0.1168, -0.8726,  0.6553,  ..., -0.8682,  0.1414,  0.2011]]],
       grad_fn=<AddBackward0>),
       
    layer_indices:tensor([1, 0])
    """
    
    
if __name__ == '__main__':
    main()