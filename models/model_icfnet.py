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

class TextEncoder(nn.Module):
    def __init__(self, clip_name='ViT-L/14', out_channels=64) -> None:
        super(TextEncoder,self).__init__()
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

class ROD(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ROD, self).__init__()
        self.projector = nn.Sequential(nn.Linear(in_channels, in_channels),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout))
        self.fusor = nn.Sequential(nn.Linear(in_channels, out_channels),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout))
        
    def forward(self, h_path, h_path_omic, h_path_text):
        h_path_r = self.projector(h_path).squeeze()
        h_path_o = h_path_r - h_path_omic - h_path_text
        h_path_f = h_path_r + h_path_o
        h_path_f = self.fusor(h_path_f)
        
        return h_path_f, h_path_o
    
class ROD_pro(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ROD_pro, self).__init__()
        self.projector = nn.Sequential(nn.Linear(in_channels, in_channels),
                                       nn.GELU(),
                                       nn.Linear(in_channels, in_channels),
                                       nn.GELU())
        self.fusor = nn.Sequential(nn.Linear(in_channels, in_channels),
                                       nn.GELU(),
                                       nn.Linear(in_channels, out_channels),
                                       nn.GELU(),
                                       nn.Dropout(dropout))
        
    def forward(self, h_path, h_path_omic, h_path_text):
        h_path_r = self.projector(h_path).squeeze()
        h_path_o = h_path_r - h_path_omic - h_path_text
        h_path_f = h_path_r + h_path_o
        h_path_f = self.fusor(h_path_f)
        
        return h_path_f, h_path_o


#############################
### ICFNet Implementation ###
#############################
class ICFNet(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2"):
        super(ICFNet, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        ### Text Encoder
        self.text_encoder = TextEncoder(clip_name='ViT-L/14', out_channels = size[2])
        self.status_encoder_dt = nn.Sequential(nn.Linear(5, 64),
                                            nn.ELU(),
                                            nn.AlphaDropout(p=0.2),
                                            nn.Linear(64, size[2]),
                                            nn.ELU(),)
        
        ### OT-based Co-attention
        self.coattn_omic = OT_Attn_assem(impl=ot_impl,ot_reg=ot_reg,ot_tau=ot_tau)
        self.coattn_text = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Path_Omic Transformer + Attention Head
        path_omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_omic_transformer = nn.TransformerEncoder(path_omic_encoder_layer, num_layers=2)
        self.path_omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Path_Text Transformer + Attention Head
        path_text_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu', batch_first=True)
        self.path_text_transformer = nn.TransformerEncoder(path_text_encoder_layer, num_layers=2)
        self.path_text_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_text_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Text Transformer + Attention Head
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu', batch_first=True)
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=2)
        self.text_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.text_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Residual Orthogonal Decomposition
        # self.rod = ROD(size[2], size[2], dropout)
        self.rod = ROD_pro(size[2], size[2], dropout)
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*5, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'uni_module':
            self.mm = Uni_Module(size[2], nhead=8, dropout=dropout)
        else:
            self.mm = None
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier_omic = nn.Sequential(nn.Linear(size[2], size[2]), nn.ReLU(), nn.Linear(size[2], n_classes))
        self.classifier_path_omic = nn.Sequential(nn.Linear(size[2], size[2]), nn.ReLU(), nn.Linear(size[2], n_classes))
        self.classifier_path = nn.Sequential(nn.Linear(size[2], size[2]), nn.ReLU(), nn.Linear(size[2], n_classes))
        self.classifier_path_text = nn.Sequential(nn.Linear(size[2], size[2]), nn.ReLU(), nn.Linear(size[2], n_classes))
        self.classifier_text = nn.Sequential(nn.Linear(size[2], size[2]), nn.ReLU(), nn.Linear(size[2], n_classes))
        
        
    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        x_demo_text = kwargs['text_info'][0]['demographic_text']
        x_treatment_text = kwargs['text_info'][0]['treatment_text']
        x_status = kwargs['text_info'][0]['status_tensor'].unsqueeze(0).cuda()
        x_status_dt = torch.cat([x_status[:,:3],x_status[:,-2:]], dim=1)
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)      # 6x1xC
        
        ### Text
        h_demo_text = self.text_encoder(x_demo_text)
        h_treatment_text = self.text_encoder(x_treatment_text)
        h_status_dt = self.status_encoder_dt(x_status_dt)
        
        h_text_bag = torch.stack([h_demo_text, h_treatment_text, h_status_dt], dim=0)      # 3x1xC
        
        ### Coattn
        A_coattn_omic, _ = self.coattn_omic(h_path_bag, h_omic_bag)
        h_path_omic_coattn = torch.mm(A_coattn_omic.squeeze(), h_path_bag.squeeze()).unsqueeze(1)     # 6x1xC
        A_coattn_text, _ = self.coattn_text(h_path_bag, h_text_bag)
        h_path_text_coattn = torch.mm(A_coattn_text.squeeze(), h_path_bag.squeeze()).unsqueeze(1) # 4x1xC
        
        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()
        
        ### Path-Omic
        h_path_omic_trans = self.path_omic_transformer(h_path_omic_coattn)
        A_path_omic, h_path_omic = self.path_omic_attention_head(h_path_omic_trans.squeeze(1))
        A_path_omic = torch.transpose(A_path_omic, 1, 0)
        h_path_omic = torch.mm(F.softmax(A_path_omic, dim=1) , h_path_omic)
        h_path_omic = self.path_omic_rho(h_path_omic).squeeze()
        
        ### Path-Text
        h_path_text_trans = self.path_text_transformer(h_path_text_coattn)
        A_path_text, h_path_text = self.path_text_attention_head(h_path_text_trans.squeeze(1))
        A_path_text = torch.transpose(A_path_text, 1, 0)
        h_path_text = torch.mm(F.softmax(A_path_text, dim=1) , h_path_text)
        h_path_text = self.path_text_rho(h_path_text).squeeze()
        
        ### Text
        h_text_trans = self.text_transformer(h_text_bag)
        A_text, h_text = self.text_attention_head(h_text_trans.squeeze(1))
        A_text = torch.transpose(A_text, 1, 0)
        h_text = torch.mm(F.softmax(A_text, dim=1) , h_text)
        h_text = self.text_rho(h_text).squeeze()
        
        ### Pure Path
        h_path = torch.mean(h_path_bag, dim=0)
        
        ### ROD
        h_path, h_path_o = self.rod(h_path, h_path_omic, h_path_text)
        
        
        if self.fusion == 'concat':
            h = self.mm(torch.cat([h_omic, h_path_omic, h_path, h_path_text, h_text], dim=0))
        elif self.fusion == 'uni_module':
            h = self.mm(h_omic, h_path_omic, h_path, h_path_text, h_text)
                
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        ### Survival Layer for each modality
        # omic
        logits_omic = self.classifier_omic(h_omic).unsqueeze(0)
        hazards_omic = torch.sigmoid(logits_omic)
        S_omic = torch.cumprod(1 - hazards_omic, dim=1)
        # path_omic
        logits_path_omic = self.classifier_path_omic(h_path_omic).unsqueeze(0)
        hazards_path_omic = torch.sigmoid(logits_path_omic)
        S_path_omic = torch.cumprod(1 - hazards_path_omic, dim=1)
        # path
        logits_path = self.classifier_path(h_path).unsqueeze(0)
        hazards_path = torch.sigmoid(logits_path)
        S_path = torch.cumprod(1 - hazards_path, dim=1)
        # path_text
        logits_path_text = self.classifier_path_text(h_path_text).unsqueeze(0)
        hazards_path_text = torch.sigmoid(logits_path_text)
        S_path_text = torch.cumprod(1 - hazards_path_text, dim=1)
        # text
        logits_text = self.classifier_text(h_text).unsqueeze(0)
        hazards_text = torch.sigmoid(logits_text)
        S_text = torch.cumprod(1 - hazards_text, dim=1)
        
        attention_scores = {}
        meta_dict = {
            'hazards_omic': hazards_omic, 'S_omic':S_omic,
            'hazards_path_omic': hazards_path_omic, 'S_path_omic':S_path_omic,
            'hazards_path': hazards_path, 'S_path':S_path,
            'hazards_path_text': hazards_path_text, 'S_path_text':S_path_text,
            'hazards_text': hazards_text, 'S_text':S_text,
        }
        feat_dict = {'h_path_omic':h_path_omic, 'h_path_text':h_path_text,'h_path':h_path,'h_path_o':h_path_o,}
        return hazards, S, Y_hat, attention_scores, meta_dict, feat_dict