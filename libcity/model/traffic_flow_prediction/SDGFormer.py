import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lape_dim, graph, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.graph=graph
        if self.graph == 1:
            self.value_embedding1 = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)
    def forward(self, x, lap_mx):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        if self.graph==1:
            x += self.value_embedding1(origin_x[:, :, :, -self.feature_dim:])

        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(lap_mx)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class STEncoder(nn.Module):
    def __init__(
            self, dim, geo_size,input_window, qkv_bias=False,
            attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(geo_size, 10), requires_grad=True)
        self.Linear3 = nn.Linear(dim, dim//2)
        self.Linear2 = nn.Linear(dim, dim//2)
        self.Linear1 = nn.Linear(dim, dim//2)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.t_q_conv = nn.Conv2d(dim, dim//4, kernel_size=1, bias=qkv_bias)
        self.t_q_conv2 = nn.Conv2d(dim, dim//4, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim//4, kernel_size=1, bias=qkv_bias)
        self.t_k_conv2 = nn.Conv2d(dim, dim//4, kernel_size=1, bias=qkv_bias)
        self.scale = 8 ** -0.5
        self.c = 7
        self.att1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),

        )
        self.att2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),

        )
        self.attn1 = torch.nn.Parameter(torch.tensor([0.6]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.4]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.4]), requires_grad=True)

        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)
    def create_adaptive_high_freq_mask(self, x_fft):
        B, j, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, j, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=2, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, j, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((
                                     normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask
    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape
        Ad = F.softmax((x @ x.permute(0, 1, 3, 2))[-1, T//2, :, :], dim=-1)


        Aad = F.softmax(F.relu(torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1) + Ad

        x1 = x.reshape(-1, N, D)

        mask1 = torch.zeros(N, N, device=x.device, requires_grad=False)
        mask2 = torch.zeros(N, N, device=x.device, requires_grad=False)
        mask3 = torch.zeros(N, N, device=x.device, requires_grad=False)

        index = torch.topk(Aad, k=int(N * 0.6), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, Aad, torch.full_like(Aad, 0))

        index = torch.topk(Aad, k=int(N * 0.5), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, Aad, torch.full_like(Aad, 0))

        index = torch.topk(Aad, k=int(N * 0.4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, Aad, torch.full_like(Aad, 0))


        out1 = self.Linear1((attn1 @ x1).reshape(B, T, N, D))
        out2 = self.Linear2((attn2 @ x1).reshape(B, T, N, D))
        out3 = self.Linear3((attn3 @ x1).reshape(B, T, N, D))
        ZS = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3


        input = x.permute(0, 2, 1, 3)

        f = torch.fft.rfft(input, dim=2, norm='ortho')

        # low_pass
        freq_mask = self.create_adaptive_high_freq_mask(f)
        low_pass = f * freq_mask.to(x.device)
        high_pass=1-freq_mask*f
        low_pass = torch.fft.irfft(low_pass, n=T, dim=2, norm='ortho')

        # high_pass
        high_pass = torch.fft.irfft(high_pass, n=T, dim=2, norm='ortho')
        high_pass = high_pass.permute(0, 2, 1, 3)
        low_pass = low_pass.permute(0, 2, 1, 3)
        x_l = self.att1(low_pass).permute(0, 3, 1, 2) + x.permute(0, 3, 1, 2)
        x_h = self.att2(high_pass).permute(0, 3, 1, 2) + x.permute(0, 3, 1, 2)

        t_ql = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_qh = self.t_q_conv2(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_kl = self.t_k_conv(x_l).permute(0, 3, 2, 1)
        t_kh = self.t_k_conv2(x_h).permute(0, 3, 2, 1)
        t_ql = t_ql.reshape(B, N, T, -1, 8).permute(0, 1, 3, 2, 4)
        t_qh = t_qh.reshape(B, N, T, -1, 8).permute(0, 1, 3, 2, 4)
        t_kl = t_kl.reshape(B, N, T, -1, 8).permute(0, 1, 3, 2, 4)
        t_kh = t_kh.reshape(B, N, T, -1, 8).permute(0, 1, 3, 2, 4)

        t_attnl = (t_ql @ t_kl.transpose(-2, -1)) * self.scale
        t_attnh = (t_qh @ t_kh.transpose(-2, -1)) * self.scale
        t_attnl = F.softmax(t_attnl, dim=-1)
        t_attnh = F.softmax(t_attnh, dim=-1)
        t_h = (t_attnh @ t_kh).transpose(2, 3).reshape(B, N, T, -1).transpose(1, 2)
        t_l = (t_attnl @ t_kl).transpose(2, 3).reshape(B, N, T, -1).transpose(1, 2)
        ZT = torch.cat((t_h, t_l), dim=-1)

        Z=torch.cat((ZS, ZT), dim=-1)

        x = self.proj(Z)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class STEncoderBlock(nn.Module):

    def __init__(
        self, dim, geo_size,input_window, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STEncoder(
            dim, geo_size=geo_size,input_window=input_window, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):

        x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SDGFormer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self._logger = getLogger()
        self.dataset = config.get('dataset')
        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        self.graph = config.get('graph',1)
        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Use use_curriculum_learning!')

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.graph, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim,geo_size=self.num_nodes,input_window=self.input_window,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    def forward(self, batch, lap_mx=None):

        x = batch['X']


        enc = self.enc_embed_layer(x, lap_mx)
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, 1, 1, 1, 1)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1)

    def get_loss_func(self, set_loss):
        if set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if y_true.size()[1]>12:
            return lf(y_predicted, y_true)
        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level - 1, self.task_level))
                self._logger.info('Current batches_seen is {}'.format(batches_seen))


            tal_loss=0
            for i in range(self.task_level):
                tal_loss += lf(y_predicted[:, i, :, :], y_true[:, i, :, :])
            return tal_loss/self.task_level

        else:
            return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)