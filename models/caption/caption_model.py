"""
    Module contains final Model and all pieces of it.
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Mapping(nn.Module):
    """
    Maps image embedding to GPT-2 embedding.
    """

    def __init__(
        self,
        ep_len,
        num_layers,
        embed_size,
        n_heads,
        forward_expansion,
        dropout,
        device="cpu",
    ):
        super(Mapping, self).__init__()

        self.ep_len = ep_len
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion

        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=embed_size * forward_expansion,
                dropout=dropout,
                batch_first=True,
                device=device,
            ),
            num_layers=num_layers,
        ).to(self.device)

        self.mapper = nn.Linear(embed_size, ep_len * embed_size* forward_expansion).to(self.device)

        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)

        x = x.view(
            *(
                [-1, self.ep_len, self.embed_size* self.forward_expansion]
                if train_mode
                else [self.ep_len, self.embed_size* self.forward_expansion]
            )
        )  # for batched input

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model, device="cpu"):
        super(TextDecoder, self).__init__()

        self.device = device

        # self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )

        return text_features.logits


class CaptionModel(nn.Module):
    """
    Final Model class. Puts all pieces together and generates caption based on image.
    """

    def __init__(
        self,
        text_model,
        ep_len,
        embed_size,
        num_layers,
        n_heads,
        forward_expansion,
        dropout,
        max_len,
        device="cpu",
    ):
        """
        Model constructor.
        Args:
            num_layers: number of layers in the TransformerEncoder
            n_heads: number of heads in the MultiHeadAttention
            forward_expansion: expansion factor for the feedforward layer
            dropout: dropout probability
            max_len: maximum length of the generated text
        """
        super(CaptionModel, self).__init__()

        self.device = device
        self.ep_len = ep_len

        self.mp = Mapping(
            ep_len=self.ep_len,
            num_layers=num_layers,
            embed_size=embed_size,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
        )
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.td = TextDecoder(model=text_model, device=device)
        
        self.max_len = max_len

        self.freeze_layers()

    def freeze_layers(self):
        for p in [*list(self.td.parameters())[14:-14],]:  # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False

    def forward(self, img_emb, text):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text
        img_mapped = self.mp(img_emb, train_mode=True)

        # tokenize text
        cap = self.tokenizer(text, padding=True, return_tensors='pt')
        trg_cap, att_mask = cap["input_ids"].cuda(), cap["attention_mask"].cuda()
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]
        
        # embed all texts and con cat with map sos
        text_emb = self.td.model.transformer.wte(x)

        # N, len, embed_size
        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat(
            [torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1
        )

        pos_emb = self.td.model.transformer.wpe(
            torch.arange(x.shape[1]).to(self.td.device)
        )
        pos_emb = pos_emb.expand_as(x)

        x += pos_emb

        res = self.td(x, attention_mask=x_mask)
        res = torch.softmax(res, dim=2)

        res = res[:, self.ep_len :, :].reshape(-1, res.shape[-1])
        y = y.reshape(-1)

        return res,y