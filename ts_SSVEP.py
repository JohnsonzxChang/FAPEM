from Patch_SSVEP import *

class SSVEPModelAll(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        assert isinstance(configs, Config)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = configs.stride

        # patching and embedding
        
        self.patch_embedding_simple = PatchEmbeddingSimple(
            configs.d_model, 3, configs.dropout/2)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        self.head_fc = CombineHead(configs.enc_in, configs.d_model, configs.pred_len, 3,
                                head_dropout=configs.dropout)
        # elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        self.head_ip = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                head_dropout=configs.dropout)
        # elif self.task_name == 'classification':
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(0.95)
        self.projection = nn.Sequential(
                nn.Linear(configs.seq_len * configs.d_model * configs.enc_in * 2, configs.num_class, bias=False), 
                # nn.Tanh(),
                # nn.Linear(configs.num_class * 2, configs.num_class)
            )
        

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out
   
    def classification_simple(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        n_vars = x_enc.shape[1]
        # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out = self.patch_embedding_simple(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        # print(output.shape)
        output = output.reshape(output.shape[0], -1)
        # print(output.shape)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    
    def forecast_simple(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(2, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        n_vars = x_enc.shape[1]
        # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out = self.patch_embedding_simple(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head_fc(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        # assert dec_out.shape == (x_enc.shape[0], 50, 9)
        # assert stdev.shape == (x_enc.shape[0], 9, 1, 3), stdev.shape
        # # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, :, 0, 0].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + \
        #           (means[:, :, 0, 0].unsqueeze(1).repeat(1, self.pred_len, 1))
        # print(dec_out.shape)
        return dec_out

    def forward000(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc = x_enc.reshape(x_enc.shape[0], -1, x_enc.shape[3]).permute(0, 2, 1)
        # print(x_enc.shape)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_enc = x_enc.permute(0, 2, 3, 1)
            dec_out = self.forecast_simple(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :].permute((0,2,1))  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            x_enc = x_enc.permute(0, 2, 3, 1)
            dec_out = self.classification_simple(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
    def forward1(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc = x_enc.permute(0, 2, 3, 1)
        n_vars = x_enc.shape[1] # 9
        # Embedding #
        enc_out = self.patch_embedding_simple(x_enc)# x_enc: 2, 9, 50, 3
        # print(x_enc.shape, enc_out.shape) # [2, 9, 50, 3] [2*9, 50, d_model]
        # Encoder #
        enc_out, attns = self.encoder(enc_out)
        # print(enc_out.shape) # [2*9, 50, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        # print(enc_out.shape) # [2, 9, d_model, 50]
        # Decoder1 #
        dec_out = self.dropout(enc_out)
        dec_out = self.head_fc(dec_out)
        dec_out = dec_out.permute(0, 3, 1, 2)
        # print(dec_out.shape) # [2, 3, 9, 50]
        # Decoder2 #
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        # print(output.shape)# [2, 9*50*d_model]
        output = self.projection(output)
        
        return dec_out, output, enc_out  # [B, L, D]  # [B, N]
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc_aux = self._forward_fc(x_enc)
        assert x_enc_aux.shape == x_enc.shape, (x_enc_aux.shape, x_enc.shape)
        x = torch.cat([x_enc, x_enc_aux], dim=-1)
        x = x.permute(0, 2, 3, 1)
        n_vars = x.shape[1] # 9
        # Embedding #
        enc_out = self.patch_embedding_simple(x)# x_enc: 2, 9, 50, 3
        # print(x_enc.shape, enc_out.shape) # [2, 9, 50, 3] [2*9, 50, d_model]
        # Encoder #
        enc_out, attns = self.encoder(enc_out)
        # print(enc_out.shape) # [2*9, 50, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        # print(enc_out.shape) # [2, 9, d_model, 50]
        # Decoder2 #
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        # print(output.shape)# [2, 9*50*d_model]
        output = self.projection(output)
        return output, x_enc_aux
    
    def _forward_fc(self, x_enc):
        x_enc = x_enc.permute(0, 2, 3, 1)
        n_vars = x_enc.shape[1] # 9
        # Embedding #
        enc_out = self.patch_embedding_simple(x_enc)# x_enc: 2, 9, 50, 3
        # print(x_enc.shape, enc_out.shape) # [2, 9, 50, 3] [2*9, 50, d_model]
        # Encoder #
        enc_out, attns = self.encoder(enc_out)
        # print(enc_out.shape) # [2*9, 50, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        # print(enc_out.shape) # [2, 9, d_model, 50]
        # Decoder1 #
        dec_out = self.dropout(enc_out)
        dec_out = self.head_fc(dec_out)
        dec_out = dec_out.permute(0, 3, 1, 2)
        # print(dec_out.shape) # [2, 3, 9, 50]
        return dec_out
        

if __name__ == '__main__':
    from torchinfo import summary
    # Assuming `configs` is defined and `Model` is initialized with it

    configs = Config()
    model = SSVEPModel(configs)
    summary(model, input_size=(2,3,9,50))