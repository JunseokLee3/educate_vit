# Implementing Vi(sual)T(transformer) in PyTorch

paper : [AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf).


**코드 참조 및 글 참조[here](https://github.com/FrancescoSaverioZuppichini/ViT)**

이 코드 vision transformerd을 공부하기 위한 코드이며, 최적화 되어 있지 않음.

시작하기 전에 다음을 수행할 것을 강력히 권장:

- 트랜스포머 설명 [The Illustrated Transformer
](https://jalammar.github.io/illustrated-transformer/) website
- watch [Yannic Kilcher video about ViT](https://www.youtube.com/watch?v=TrdevFK_am4&t=1000s)
- 라이브러리 [Einops](https://github.com/arogozhnikov/einops/) doc


다음 그림은 ViT의 아키텍처를 보여 줌.

![alt](https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/ViT.png?raw=true)

입력 이미지가 16x16 플랫 패치로 분해된다.(이미지 크기가 조정되지 않음). 그런 다음 완전히 연결된 일반 레이어를 사용하여 삽입하고 앞에 특별한 'cls' 토큰이 추가되고 '위치 인코딩'이 더해진다. 과 텐서는 먼저 표준 트랜스포머로 전달된 다음 분류 헤드로 전달된다.

이 문서는 다음 섹션으로 구성되어 있습니다:

- Data
- Patches Embeddings
    - CLS Token
    - Position Embedding
- Transformer
    - Attention
    - Residuals
    - MLP
    - TransformerEncoder
- Head
- ViT

우리는 bottom-up 접근 방식으로 모델을 블록별로 구현할 것이다. 필요한 모든 패키지를 가져오는 것으로 시작할 수 있음.


```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

import urllib.request
from PIL import Image

```

이미지 : 

![cat](cat.jpg)

## Data

```python
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
print('image.size :',x.shape)
x = x.unsqueeze(0) # add batch dim
print('after unsqueeze(0) :',x.shape)
```

    image.size : torch.Size([3, 224, 224])
    after unsqueeze(0) : torch.Size([1, 3, 224, 224])



## Patches Embeddings

첫 번째 단계는 이미지를 여러 패치로 분해하여 평평하게 만드는 것.

![alt](https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/Patches.png?raw=true)

Quoting from the paper:

![alt](https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/paper1.png?raw=true)

2D 이미지를 처리하기 위해 이미지 $\mathsf x \in \mathbb R^{H \times W \times C}$를 평평한 2D 패치 $\mathsf x_p \in \mathbb R^{N \times (P^2 \cdot C)}$ 시퀀스로 재구성한다. 여기서 $(H, W)$는 원본 이미지의 해상도, $C$는 채널 수, $(P, P)$는 각 이미지 패치의 해상도, $N=HW/P^2$은 결과 패치 수이다.

이것은 `einops`를 사용하면 쉽게 할 수 있다.

```python
patch_size = 16 # 16 pixels
pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
```

이제 일반 선형 레이어를 사용하여 투영해야 한다.

![alt](https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/PatchesProjected.png?raw=true)



```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
PatchEmbedding()(x).shape
```

    torch.Size([1, 196, 768])



**참고** ***원본 구현을 확인한 후, 저자들이 성능 향상을 위해 Linear 계층 대신 Conv2d 계층을 사용하고 있다는 것을 알게 됨.*** 이는 kernel_size를 사용하고 patch_size와 동일한 보폭을 사용함으로써 얻을 수 있고 직관적으로 컨볼루션 작업은 각 패치에 개별적으로 적용됨. 따라서, 우리는 먼저 컨볼루션 레이어를 적용한 후 결과 이미지를 평탄화해야 함.


```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
a_k=nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)(x)
print('커널 들어 간 후 shape : ',a_k.shape)    
PatchEmbedding()(x).shape
```

    커널 들어 간 후 shape :  torch.Size([1, 768, 14, 14])
    torch.Size([1, 196, 768])



### CLS Token

다음 단계는 `cls 토큰`과 위치 임베딩을 추가하는 것입니다. `cls 토큰`은 **각** 시퀀스(프로젝션된 패치)에서 삽입된 숫자일 뿐.

> **참고** Vision Transformer의 흥미로운 점 중 하나는 아키텍처가 클래스 토큰을 사용한다는 것입니다. 이러한 클래스 토큰은 입력 시퀀스의 시작 부분에 추가되는 무작위로 초기화된 토큰입니다. 이 클래스 토큰의 이유는 무엇이며 어떤 역할을 합니까? **Class Token은 무작위로 초기화되므로 자체적으로 유용한 정보가 포함되어 있지 않습니다.** 그러나 Class Token은 Transformer가 더 깊고 더 많은 계층의 시퀀스에서 다른 토큰의 정보를 축적할 수 있습니다. Vision Transformer가 최종적으로 시퀀스의 최종 분류를 수행할 때 MLP 헤드를 사용하여 마지막 계층의 클래스 토큰 데이터만 보고 다른 정보는 보지 않습니다. **이 작업은 클래스 토큰이 시퀀스의 다른 토큰에서 추출된 정보를 저장하는 데 사용되는 자리 표시자 데이터 구조임을 시사합니다.** 이 절차에 빈 토큰을 할당하면 Vision Transformer가 다른 개별 토큰 중 하나에 대해 최종 출력을 편향시킬 가능성이 낮아집니다. [링크](https://deepganteam.medium.com/vision-transformers-for-computer-vision-9f70418fe41a)

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        return x
    
PatchEmbedding()(x).shape
```




    torch.Size([1, 197, 768])



`cls_token`은 무작위로 초기화되는 토치 매개변수이며, 앞으로 `b`(배치)번 복사되고 `torch.cat`을 사용하여 투영된 패치 앞에 추가됩니다.

### Position Embedding

**지금까지 모델은 패치의 원래 위치에 대해 알지 못했습니다. 이 공간 정보를 전달해야 합니다. 이것은 다양한 방법으로 수행할 수 있습니다. ViT에서는 모델이 학습하도록 합니다.** 위치 임베딩은 투영된 패치에 추가되는 N_PATCHES + 1(토큰), EMBED_SIZE 모양의 텐서일 뿐.


```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
PatchEmbedding()(x).shape
```



    torch.Size([1, 197, 768])



`.positions` 필드에 위치 임베딩을 추가하고 `.forward` 함수의 패치에 합산.

## Transformer

이제 Transformer 구현이 필요. ViT에서는 Encoder만 사용하며 아키텍처는 다음 그림과 같이 시각화.

<img src="https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/TransformerBlock.png?raw=true" alt="drawing" width="200"/>

Let's start with the Attention part

### Attention

따라서 Attention은 Query, Key 및 Value의 세 가지 입력을 취하고 Query와 Value을 사용하여 Attention 매트릭스를 계산. Value에 "attend"하는 데 사용한다. 이 경우, 우리는 더 작은 입력 크기를 가진 n개의 헤드로 분할된다는 것을 의미하는 다중 헤드 attention을 사용하고 있다.

![alt](https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/TransformerBlockAttention.png?raw=true)

PyTorch의 `nn.MultiHadAttention`을 사용하거나 자체적으로 구현할 수 있음. 완성도를 위해 다음과 같이 표시:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
patches_embedded = PatchEmbedding()(x)
MultiHeadAttention()(patches_embedded).shape
```




    torch.Size([1, 197, 768])


4개의 완전히 연결된 계층이 있습니다. 하나는 쿼리, 키, 값용이고 마지막 하나는 드롭아웃입니다.

아이디어는 쿼리와 키 사이에 곱을 사용하여 각 요소가 나머지와 함께 중요한 순서인 "얼마나"인지 아는 것입니다. 그런 다음 이 정보를 사용하여 값을 조정합니다.

`forward` 방법은 이전 레이어의 쿼리, 키 및 값을 입력으로 받아 3개의 선형 레이어를 사용하여 투영. 다중 헤드 어텐션을 구현하기 때문에 결과를 다중 헤드로 재정렬 해야 함.

이것은 einops에서 `rearrange`를 사용하여 수행.

* 쿼리, 키 및 값은 항상 동일하므로 단순화를 위해 하나의 입력만 있습니다. 

```python
queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.n_heads)
keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.n_heads)
values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.n_heads)
```

결과 키, 쿼리 및 값의 모양은 `BATCH, HEADS, SEQUENCE_LEN EMBEDDING_SIZE` 이다.

attention 행렬을 계산하려면 먼저 쿼리와 키 사이의 행렬 곱셈(마지막 축에 대한 합)을 수행해야 한다. 이 작업은 `torch.einsum`을 사용하여 쉽게 수행할 수 있음.

```python
energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys
```

결과 벡터는 `BATCH, HEADS, QUERY_LEN, KEY_LEN` 형태를 갖음. 그런 다음 마지막으로 attention은 임베딩의 크기에 기초한 스케일링 계수로 나눈 결과 벡터의 소프트맥스 이다.


마지막으로, attention을 사용하여 Value을 조정 함.

```python
torch.einsum('bhal, bhlv -> bhav ', att, values)
```

**Note** 단일 행렬을 사용하여 `queries, kesys 및 values`을 한 번에 계산할 수 있다. 


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
patches_embedded = PatchEmbedding()(x)
MultiHeadAttention()(patches_embedded).shape
```




    torch.Size([1, 197, 768])



### Residuals

Transformer 블록에 잔차 연결부가 있음

![alt](https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/TransformerBlockAttentionRes.png?raw=true)


```python
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
```

## MLP

주의의 출력은 입력을 `확장`하는 요인으로 업샘플링하는 두 개의 레이어로 구성된 완전히 연결된 레이어로 전달 됨.

<img src="https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/TransformerBlockAttentionZoom.png?raw=true" alt="drawing" width="200"/>



```python
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
```

**Finally**, we can create the Transformer Encoder Block

<img src="https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/images/TransformerBlock.png?raw=true" alt="drawing" width="200"/>


`ResidualAdd`를 사용하면 이 블록을 정의할 수 있습니다.


```python
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
```

Let's test it


```python
patches_embedded = PatchEmbedding()(x)
TransformerEncoderBlock()(patches_embedded).shape
```




    torch.Size([1, 197, 768])

### Transformer

ViT에서는 원래 Transformer의 인코더 부분만 사용.


```python
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
                
```

## Head

마지막 계층은 클래스 확률을 제공하는 General fully connected임. 먼저 전체 시퀀스에 대해 기본 평균을 수행 함.


```python
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
```

## Vi(sual) T(rasnformer)

우리는 최종 ViT 아키텍처를 생성하기 위해 `PatchEmbedding`, `TransformerEncoder` 및 `ClassificationHead`를 구성 함.


```python
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
        
```

매개변수의 수를 확인하기 위해 `torchsummary`를 사용할 수 있음.


```python
summary(ViT(), (3, 224, 224), device='cpu')
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1          [-1, 768, 14, 14]         590,592
             Rearrange-2             [-1, 196, 768]               0
        PatchEmbedding-3             [-1, 197, 768]               0
             LayerNorm-4             [-1, 197, 768]           1,536
                Linear-5            [-1, 197, 2304]       1,771,776
               Dropout-6          [-1, 8, 197, 197]               0
                Linear-7             [-1, 197, 768]         590,592
    MultiHeadAttention-8             [-1, 197, 768]               0
               Dropout-9             [-1, 197, 768]               0
          ResidualAdd-10             [-1, 197, 768]               0
            LayerNorm-11             [-1, 197, 768]           1,536
               Linear-12            [-1, 197, 3072]       2,362,368
                 GELU-13            [-1, 197, 3072]               0
              Dropout-14            [-1, 197, 3072]               0
               Linear-15             [-1, 197, 768]       2,360,064
              Dropout-16             [-1, 197, 768]               0
          ResidualAdd-17             [-1, 197, 768]               0
            LayerNorm-18             [-1, 197, 768]           1,536
               Linear-19            [-1, 197, 2304]       1,771,776
              Dropout-20          [-1, 8, 197, 197]               0
               Linear-21             [-1, 197, 768]         590,592
    MultiHeadAttention-22             [-1, 197, 768]               0
              Dropout-23             [-1, 197, 768]               0
          ResidualAdd-24             [-1, 197, 768]               0
            LayerNorm-25             [-1, 197, 768]           1,536
               Linear-26            [-1, 197, 3072]       2,362,368
                 GELU-27            [-1, 197, 3072]               0
              Dropout-28            [-1, 197, 3072]               0
               Linear-29             [-1, 197, 768]       2,360,064
              Dropout-30             [-1, 197, 768]               0
          ResidualAdd-31             [-1, 197, 768]               0
            LayerNorm-32             [-1, 197, 768]           1,536
               Linear-33            [-1, 197, 2304]       1,771,776
              Dropout-34          [-1, 8, 197, 197]               0
               Linear-35             [-1, 197, 768]         590,592
    MultiHeadAttention-36             [-1, 197, 768]               0
              Dropout-37             [-1, 197, 768]               0
          ResidualAdd-38             [-1, 197, 768]               0
            LayerNorm-39             [-1, 197, 768]           1,536
               Linear-40            [-1, 197, 3072]       2,362,368
                 GELU-41            [-1, 197, 3072]               0
              Dropout-42            [-1, 197, 3072]               0
               Linear-43             [-1, 197, 768]       2,360,064
              Dropout-44             [-1, 197, 768]               0
          ResidualAdd-45             [-1, 197, 768]               0
            LayerNorm-46             [-1, 197, 768]           1,536
               Linear-47            [-1, 197, 2304]       1,771,776
              Dropout-48          [-1, 8, 197, 197]               0
               Linear-49             [-1, 197, 768]         590,592
    MultiHeadAttention-50             [-1, 197, 768]               0
              Dropout-51             [-1, 197, 768]               0
          ResidualAdd-52             [-1, 197, 768]               0
            LayerNorm-53             [-1, 197, 768]           1,536
               Linear-54            [-1, 197, 3072]       2,362,368
                 GELU-55            [-1, 197, 3072]               0
              Dropout-56            [-1, 197, 3072]               0
               Linear-57             [-1, 197, 768]       2,360,064
              Dropout-58             [-1, 197, 768]               0
          ResidualAdd-59             [-1, 197, 768]               0
            LayerNorm-60             [-1, 197, 768]           1,536
               Linear-61            [-1, 197, 2304]       1,771,776
              Dropout-62          [-1, 8, 197, 197]               0
               Linear-63             [-1, 197, 768]         590,592
    MultiHeadAttention-64             [-1, 197, 768]               0
              Dropout-65             [-1, 197, 768]               0
          ResidualAdd-66             [-1, 197, 768]               0
            LayerNorm-67             [-1, 197, 768]           1,536
               Linear-68            [-1, 197, 3072]       2,362,368
                 GELU-69            [-1, 197, 3072]               0
              Dropout-70            [-1, 197, 3072]               0
               Linear-71             [-1, 197, 768]       2,360,064
              Dropout-72             [-1, 197, 768]               0
          ResidualAdd-73             [-1, 197, 768]               0
            LayerNorm-74             [-1, 197, 768]           1,536
               Linear-75            [-1, 197, 2304]       1,771,776
              Dropout-76          [-1, 8, 197, 197]               0
               Linear-77             [-1, 197, 768]         590,592
    MultiHeadAttention-78             [-1, 197, 768]               0
              Dropout-79             [-1, 197, 768]               0
          ResidualAdd-80             [-1, 197, 768]               0
            LayerNorm-81             [-1, 197, 768]           1,536
               Linear-82            [-1, 197, 3072]       2,362,368
                 GELU-83            [-1, 197, 3072]               0
              Dropout-84            [-1, 197, 3072]               0
               Linear-85             [-1, 197, 768]       2,360,064
              Dropout-86             [-1, 197, 768]               0
          ResidualAdd-87             [-1, 197, 768]               0
            LayerNorm-88             [-1, 197, 768]           1,536
               Linear-89            [-1, 197, 2304]       1,771,776
              Dropout-90          [-1, 8, 197, 197]               0
               Linear-91             [-1, 197, 768]         590,592
    MultiHeadAttention-92             [-1, 197, 768]               0
              Dropout-93             [-1, 197, 768]               0
          ResidualAdd-94             [-1, 197, 768]               0
            LayerNorm-95             [-1, 197, 768]           1,536
               Linear-96            [-1, 197, 3072]       2,362,368
                 GELU-97            [-1, 197, 3072]               0
              Dropout-98            [-1, 197, 3072]               0
               Linear-99             [-1, 197, 768]       2,360,064
             Dropout-100             [-1, 197, 768]               0
         ResidualAdd-101             [-1, 197, 768]               0
           LayerNorm-102             [-1, 197, 768]           1,536
              Linear-103            [-1, 197, 2304]       1,771,776
             Dropout-104          [-1, 8, 197, 197]               0
              Linear-105             [-1, 197, 768]         590,592
    MultiHeadAttention-106             [-1, 197, 768]               0
             Dropout-107             [-1, 197, 768]               0
         ResidualAdd-108             [-1, 197, 768]               0
           LayerNorm-109             [-1, 197, 768]           1,536
              Linear-110            [-1, 197, 3072]       2,362,368
                GELU-111            [-1, 197, 3072]               0
             Dropout-112            [-1, 197, 3072]               0
              Linear-113             [-1, 197, 768]       2,360,064
             Dropout-114             [-1, 197, 768]               0
         ResidualAdd-115             [-1, 197, 768]               0
           LayerNorm-116             [-1, 197, 768]           1,536
              Linear-117            [-1, 197, 2304]       1,771,776
             Dropout-118          [-1, 8, 197, 197]               0
              Linear-119             [-1, 197, 768]         590,592
    MultiHeadAttention-120             [-1, 197, 768]               0
             Dropout-121             [-1, 197, 768]               0
         ResidualAdd-122             [-1, 197, 768]               0
           LayerNorm-123             [-1, 197, 768]           1,536
              Linear-124            [-1, 197, 3072]       2,362,368
                GELU-125            [-1, 197, 3072]               0
             Dropout-126            [-1, 197, 3072]               0
              Linear-127             [-1, 197, 768]       2,360,064
             Dropout-128             [-1, 197, 768]               0
         ResidualAdd-129             [-1, 197, 768]               0
           LayerNorm-130             [-1, 197, 768]           1,536
              Linear-131            [-1, 197, 2304]       1,771,776
             Dropout-132          [-1, 8, 197, 197]               0
              Linear-133             [-1, 197, 768]         590,592
    MultiHeadAttention-134             [-1, 197, 768]               0
             Dropout-135             [-1, 197, 768]               0
         ResidualAdd-136             [-1, 197, 768]               0
           LayerNorm-137             [-1, 197, 768]           1,536
              Linear-138            [-1, 197, 3072]       2,362,368
                GELU-139            [-1, 197, 3072]               0
             Dropout-140            [-1, 197, 3072]               0
              Linear-141             [-1, 197, 768]       2,360,064
             Dropout-142             [-1, 197, 768]               0
         ResidualAdd-143             [-1, 197, 768]               0
           LayerNorm-144             [-1, 197, 768]           1,536
              Linear-145            [-1, 197, 2304]       1,771,776
             Dropout-146          [-1, 8, 197, 197]               0
              Linear-147             [-1, 197, 768]         590,592
    MultiHeadAttention-148             [-1, 197, 768]               0
             Dropout-149             [-1, 197, 768]               0
         ResidualAdd-150             [-1, 197, 768]               0
           LayerNorm-151             [-1, 197, 768]           1,536
              Linear-152            [-1, 197, 3072]       2,362,368
                GELU-153            [-1, 197, 3072]               0
             Dropout-154            [-1, 197, 3072]               0
              Linear-155             [-1, 197, 768]       2,360,064
             Dropout-156             [-1, 197, 768]               0
         ResidualAdd-157             [-1, 197, 768]               0
           LayerNorm-158             [-1, 197, 768]           1,536
              Linear-159            [-1, 197, 2304]       1,771,776
             Dropout-160          [-1, 8, 197, 197]               0
              Linear-161             [-1, 197, 768]         590,592
    MultiHeadAttention-162             [-1, 197, 768]               0
             Dropout-163             [-1, 197, 768]               0
         ResidualAdd-164             [-1, 197, 768]               0
           LayerNorm-165             [-1, 197, 768]           1,536
              Linear-166            [-1, 197, 3072]       2,362,368
                GELU-167            [-1, 197, 3072]               0
             Dropout-168            [-1, 197, 3072]               0
              Linear-169             [-1, 197, 768]       2,360,064
             Dropout-170             [-1, 197, 768]               0
         ResidualAdd-171             [-1, 197, 768]               0
              Reduce-172                  [-1, 768]               0
           LayerNorm-173                  [-1, 768]           1,536
              Linear-174                 [-1, 1000]         769,000
    ================================================================
    Total params: 86,415,592
    Trainable params: 86,415,592
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 364.33
    Params size (MB): 329.65
    Estimated Total Size (MB): 694.56
    ----------------------------------------------------------------
    





    (tensor(86415592), tensor(86415592), tensor(329.6493), tensor(694.5562))



et voilà

```
================================================================
Total params: 86,415,592
Trainable params: 86,415,592
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 364.33
Params size (MB): 329.65
Estimated Total Size (MB): 694.56
---------------------------------------------------------------
```

By the way, I am working on a **new computer vision library called [glasses](https://github.com/FrancescoSaverioZuppichini/glasses), check it out if you like**
