import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet

from mtl.datasets.definitions import RESNET34PATH
from mtl.utils.pytorch_code import load_state_dict_from_url

class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = SqueezeAndExcitation(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], False, progress, **encoder_kwargs
            )
            if pretrained:
                state_dict = load_state_dict_from_url(RESNET34PATH,
                                                    progress=progress)
                model.load_state_dict(state_dict, strict=False)

        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch, cfg, upsample=True):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following

        ## 48 from paper code
        self.skip_add = cfg.skip_add
        self.upsample = upsample

        if self.skip_add:
            self.skip_layer = nn.Sequential(nn.Conv2d(skip_4x_ch, 48, 1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU())
            self.last_conv = nn.Sequential(nn.Conv2d(bottleneck_ch+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        )
        self.features_to_predictions = torch.nn.Conv2d(256, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        if self.upsample:
            features_4x = F.interpolate(
                features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
            )
        else:
            features_4x = features_bottleneck

        if self.skip_add:
            x = self.skip_layer(features_skip_4x)
            x = torch.cat((x, features_4x), dim=1)
            x = self.last_conv(x)
            predictions_4x = self.features_to_predictions(x)
            return predictions_4x, x
        else:
            predictions_4x = self.features_to_predictions(features_4x)
            return predictions_4x, features_4x

class DecoderDeeplabV3pSelfAtten(torch.nn.Module):
    def __init__(self, features_init_ch, num_out_ch):
        super(DecoderDeeplabV3pSelfAtten, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following

        ## 48 from paper code
       
        self.features_to_predictions = torch.nn.Conv2d(features_init_ch, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_init, features_self_atten):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        features_out=features_init+features_self_atten
        x=self.features_to_predictions(features_out)

        return x
    

class DecoderDeeplabV3pAllConnect(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_2x_ch, skip_4x_ch, skip_8x_ch, num_out_ch, cfg, upsample=True):
        super(DecoderDeeplabV3pAllConnect, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following

        ## 48 from paper code
        self.skip_add = cfg.skip_add
        self.upsample = upsample

        if self.skip_add:
            self.skip_layer_8x = nn.Sequential(nn.Conv2d(skip_8x_ch, 48, 1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU())
            self.skip_layer_4x = nn.Sequential(nn.Conv2d(skip_4x_ch, 48, 1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU())
            self.skip_layer_2x = nn.Sequential(nn.Conv2d(skip_2x_ch, 48, 1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU())                                
            self.comb_layer_8x = nn.Sequential(nn.Conv2d(bottleneck_ch+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
            self.comb_layer_4x = nn.Sequential(nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
            self.comb_layer_2x = nn.Sequential(nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
        self.features_to_predictions = torch.nn.Conv2d(256, num_out_ch, kernel_size=1, stride=1)

    @staticmethod
    def merge_bottleneck_skip_features(features_bottleneck_scale, features_skip_scale, skip_layer, comb_layer):
        features = F.interpolate(
                features_bottleneck_scale, size=features_skip_scale.shape[2:], mode='bilinear', align_corners=False)
        x = skip_layer(features_skip_scale)
        x = torch.cat((x, features), dim=1)
        x = comb_layer(x)

        return x


    def forward(self, features_bottleneck, features_skip):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        feature_8x = self.merge_bottleneck_skip_features(features_bottleneck, features_skip[8], self.skip_layer_8x, self.comb_layer_8x)
        feature_4x = self.merge_bottleneck_skip_features(feature_8x, features_skip[4], self.skip_layer_4x, self.comb_layer_4x)
        features_2x = self.merge_bottleneck_skip_features(feature_4x, features_skip[2], self.skip_layer_2x, self.comb_layer_2x)
        predictions_2x = self.features_to_predictions(features_2x)
        return predictions_2x, features_2x


class DecoderDeeplabV3pAllConnectAllAttend(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_2x_ch, skip_4x_ch, skip_8x_ch, num_out_ch):
        super(DecoderDeeplabV3pAllConnectAllAttend, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following

        ## 48 from paper code
        self.skip_layer_8x = nn.Sequential(nn.Conv2d(skip_8x_ch, 48, 1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU())
        self.skip_layer_4x = nn.Sequential(nn.Conv2d(skip_4x_ch, 48, 1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU())
        self.skip_layer_2x = nn.Sequential(nn.Conv2d(skip_2x_ch, 48, 1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU())                                
        self.comb_layer_8x = nn.Sequential(nn.Conv2d(bottleneck_ch+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.comb_layer_4x = nn.Sequential(nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.comb_layer_2x = nn.Sequential(nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.pred_layer_8x = torch.nn.Conv2d(256, num_out_ch, kernel_size=1, stride=1)
        self.pred_layer_4x = torch.nn.Conv2d(256, num_out_ch, kernel_size=1, stride=1)
        self.pred_layer_2x = torch.nn.Conv2d(256, num_out_ch, kernel_size=1, stride=1)

    @staticmethod
    def merge_bottleneck_skip_features(features_bottleneck_scale, features_skip_scale, skip_layer, \
                                        comb_layer, pred_layer, input_resolution):
        features = F.interpolate(
                features_bottleneck_scale, size=features_skip_scale.shape[2:], mode='bilinear', align_corners=False)
        x = skip_layer(features_skip_scale)
        x = torch.cat((x, features), dim=1)
        x = comb_layer(x)
        pred = pred_layer(x)
        pred = F.interpolate(pred, size=input_resolution, mode='bilinear', align_corners=False)

        return pred, x


    def forward(self, features_bottleneck, features_skip, input_resolution):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        pred_8x, feature_8x = self.merge_bottleneck_skip_features(features_bottleneck, features_skip[8], self.skip_layer_8x, \
                        self.comb_layer_8x, self.pred_layer_8x, input_resolution)
        pred_4x, feature_4x = self.merge_bottleneck_skip_features(feature_8x, features_skip[4], self.skip_layer_4x, \
                        self.comb_layer_4x, self.pred_layer_4x, input_resolution)
        pred_2x, features_2x = self.merge_bottleneck_skip_features(feature_4x, features_skip[2], self.skip_layer_2x, \
                        self.comb_layer_2x, self.pred_layer_2x, input_resolution)

        feature_dict = {8: feature_8x, 4: feature_4x, 2: features_2x}
        pred_dict = {8: pred_8x, 4: pred_4x, 2: pred_2x}
        return pred_dict, feature_dict


class SelfAttentionDecodeAllScale(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_out_ch):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.resolution = tuple([int(res/2) for res in input_resolution])

        self.attend_8x = SelfAttention(self.in_channels, self.out_channels)
        self.attend_4x = SelfAttention(self.in_channels, self.out_channels)
        self.attend_2x = SelfAttention(self.in_channels, self.out_channels)

        self.features_to_predictions = torch.nn.Conv2d(3*256, num_out_ch, kernel_size=1, stride=1)

    
    def cross_modal_distill(self, attend_features, task_features, attend_layer, resolution):
        attend_features = attend_layer(attend_features)
        task_features = task_features + attend_features

        task_features_interplolate = F.interpolate(task_features, size=resolution, mode='bilinear', align_corners=False)

        return task_features_interplolate

    def forward(self, feature_dict_task, feature_dict_attend, input_resolution):
        resolution = tuple([int(res/2) for res in input_resolution])

        task_features_8x_to_2x = self.cross_modal_distill(feature_dict_attend[8], feature_dict_task[8], self.attend_8x, resolution)
        task_features_4x_to_2x = self.cross_modal_distill(feature_dict_attend[4], feature_dict_task[4], self.attend_4x, resolution)
        task_features_2x_to_2x = self.cross_modal_distill(feature_dict_attend[2], feature_dict_task[2], self.attend_2x, resolution)

        final_features = torch.cat((task_features_8x_to_2x, task_features_4x_to_2x, task_features_2x_to_2x), dim=1)

        prediction = self.features_to_predictions(final_features)

        return prediction



class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels,cfg,rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following
        self.cfg = cfg
        if (self.cfg.aspp_add):
            self.conv1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
            self.conv2 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
            self.conv3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
            self.conv4 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
            self.pooling=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                            torch.nn.BatchNorm2d(out_channels),
                                            torch.nn.ReLU())
            
            self.conv_out=ASPPpart(5*out_channels,out_channels,kernel_size=1, stride=1, padding=0, dilation=1)
        else:
            self.conv_out = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        if (self.cfg.aspp_add):
            x1=self.conv1(x)
            x2=self.conv2(x)
            x3=self.conv3(x)
            x4=self.conv4(x)
            x5=self.pooling(x)
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
            x_cat=torch.cat((x1,x2,x3,x4,x5), dim=1)
            out = self.conv_out(x_cat)
        else:
            out=self.conv_out(x)
        return out


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # with torch.no_grad():
        #     self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed
