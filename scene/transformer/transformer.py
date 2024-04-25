import torch.nn as nn
import torch

from scene.transformer.layer_norm import LayerNorm
from scene.transformer.multi_head_attention import MultiHeadAttention
from scene.transformer.position_wise_feed_forward import PositionwiseFeedForward

class Spatial_Audio_Attention_Layer(nn.Module):
    def __init__(self,args):
        super(Spatial_Audio_Attention_Layer, self).__init__()
        self.args = args
        
        self.enc_dec_attention = MultiHeadAttention(d_model=self.args.d_model, n_head=self.args.n_head)
        self.norm1 = LayerNorm(d_model=self.args.d_model)
        self.dropout1 = nn.Dropout(p=self.args.drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=self.args.d_model, hidden=self.args.ffn_hidden, drop_prob=self.args.drop_prob)
        self.norm2 = LayerNorm(d_model=self.args.d_model)
        self.dropout2 = nn.Dropout(p=self.args.drop_prob)
        
    def forward(self, x, enc_source):
        _x = x
        x, att = self.enc_dec_attention(q=x, k=enc_source, v=enc_source, mask=None)
        
        # 4. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x, att
    

class Spatial_Audio_Attention_Module(nn.Module):
    def __init__(self,args):
        super(Spatial_Audio_Attention_Module, self).__init__()
        self.args = args
        self.layers = nn.ModuleList([Spatial_Audio_Attention_Layer(args) for _ in range(args.n_layer)])
        
    def forward(self,x, enc_source):
        attention = []
        for layer in self.layers:
            x,att = layer(x, enc_source)
            attention.append(att.mean(dim=1).unsqueeze(dim=1))
        attention = torch.cat(attention,dim=1) #B, layer, N, 3
        return x, attention