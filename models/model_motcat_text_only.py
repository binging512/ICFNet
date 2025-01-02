import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

import ot

from models.model_utils import *
from .clip import clip

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIPText(nn.Module):
    def __init__(self, clip_name='ViT-L/14', out_channels=64) -> None:
        super(CustomCLIPText,self).__init__()
        self.clip, self.clip_preprocess = clip.load(clip_name, download_root='./pretrained')
        self.adapter = Adapter(c_in=768)
        self.adapter_alpha = nn.parameter.Parameter(torch.tensor(0.2))
        self.adapter_projection = nn.Sequential(nn.Linear(768, out_channels), nn.ReLU(inplace=True))
        
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
        
    def forward(self, texts):
        text_tokens = clip.tokenize(texts).cuda()
        text_feat = self.clip.encode_text(text_tokens).float()
        text_adapt = self.adapter(text_feat)
        text_feat = self.adapter_alpha*text_adapt + (1-self.adapter_alpha)*text_feat
        text_feat = self.adapter_projection(text_feat)
        
        return text_feat


class OT_Attn_assem(nn.Module):
    def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)
    
    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow 
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double(), reg=self.ot_reg,reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()
            
            dist = self.cost_map * flow # (N, M)
            dist = torch.sum(dist) # (1,) float
            return flow, dist
        
        else:
            raise NotImplementedError

        

    def forward(self,x,y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()
        
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        
        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist

       
#############################
### MOTCAT Implementation ###
#############################
class MOTCAT_text_only_Surv(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2"):
        super(MOTCAT_text_only_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        
        ### Text Encoder
        self.text_encoder = CustomCLIPText(clip_name='ViT-L/14', out_channels = size[2])
        self.status_encoder = nn.Sequential(nn.Linear(8, 64),
                                            nn.ELU(),
                                            nn.AlphaDropout(p=0.2),
                                            nn.Linear(64, size[2]),
                                            nn.ELU(),)
        
        ### Text Transformer + Attention Head
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu', batch_first=True)
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=2)
        self.text_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.text_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        x_demo_text = kwargs['text_info'][0]['demographic_text']
        x_diag_text = kwargs['text_info'][0]['diagnosis_text']
        x_treatment_text = kwargs['text_info'][0]['treatment_text']
        x_status = kwargs['text_info'][0]['status_tensor'].unsqueeze(0).cuda()
        
        ### Text
        h_demo_text = self.text_encoder(x_demo_text)
        h_diag_text = self.text_encoder(x_diag_text)
        h_treatment_text = self.text_encoder(x_treatment_text)
        h_status = self.status_encoder(x_status)
        h_text_bag = torch.stack([h_demo_text, h_diag_text, h_treatment_text,h_status], dim=0)      # 4x1xC
        
        ### Text
        h_text_trans = self.text_transformer(h_text_bag)
        A_text, h_text = self.text_attention_head(h_text_trans.squeeze(1))
        A_text = torch.transpose(A_text, 1, 0)
        h_text = torch.mm(F.softmax(A_text, dim=1) , h_text)
        h_text = self.text_rho(h_text).squeeze()
        
        if self.fusion == 'concat':
            h = self.mm(h_text)
                
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        attention_scores = {'coattn': 0, 'path': 0, 'omic': 0}
        
        return hazards, S, Y_hat, attention_scores