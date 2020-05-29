import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p,\
    SelfAttention, SqueezeAndExcitation, DecoderDeeplabV3pAllConnect
from mtl.datasets.definitions import *


class ModelTaskDistillAllConnect(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())
        ch_out_seg=outputs_desc[MOD_SEMSEG]
        ch_out_depth=outputs_desc[MOD_DEPTH]
        ch_attention = 48
        self.add_se = cfg.add_se

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.se_seg = SqueezeAndExcitation(ch_out_encoder_bottleneck)
        self.se_depth = SqueezeAndExcitation(ch_out_encoder_bottleneck)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256,cfg)
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256,cfg)

        self.decoder_seg1 = DecoderDeeplabV3pAllConnect(256, 64, ch_out_encoder_4x, 128, ch_out_seg, cfg)
        self.decoder_depth1 = DecoderDeeplabV3pAllConnect(256, 64, ch_out_encoder_4x, 128, ch_out_depth, cfg)

        self.self_attention_seg = SelfAttention(256, ch_attention)
        self.self_attention_depth = SelfAttention(256, ch_attention)

        self.decoder_seg2 = DecoderDeeplabV3p(256, ch_attention, ch_out_seg, cfg, upsample=False)
        self.decoder_depth2 = DecoderDeeplabV3p(256, ch_attention, ch_out_depth, cfg, upsample=False)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        # features_tasks = self.aspp(features_lowest)
        if self.add_se:
            features_seg = self.se_seg(features_lowest)
            features_depth = self.se_depth(features_lowest)

            features_task_seg = self.aspp_seg(features_seg)
            features_task_depth = self.aspp_depth(features_depth)

        else:
            features_task_seg = self.aspp_seg(features_lowest)
            features_task_depth = self.aspp_depth(features_lowest)

        predictions_2x_seg1, features_seg = self.decoder_seg1(features_task_seg, features)
        predictions_2x_depth1, features_depth = self.decoder_depth1(features_task_depth, features)
        # predictions_4x, _ = self.decoder(features_tasks, features[4])

        predictions_1x_seg1 = F.interpolate(predictions_2x_seg1, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_depth1 = F.interpolate(predictions_2x_depth1, size=input_resolution, mode='bilinear', align_corners=False)

        attention_seg = self.self_attention_seg(features_seg)
        attention_depth = self.self_attention_depth(features_depth)

        predictions_2x_seg2, _ = self.decoder_seg2(features_seg, attention_depth)
        predictions_2x_depth2, _ = self.decoder_depth2(features_depth, attention_seg)

        predictions_1x_seg2 = F.interpolate(predictions_2x_seg2, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_depth2 = F.interpolate(predictions_2x_depth2, size=input_resolution, mode='bilinear', align_corners=False)
        # predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)
        
        out={MOD_SEMSEG:[predictions_1x_seg1, predictions_1x_seg2],
            MOD_DEPTH:[predictions_1x_depth1, predictions_1x_depth2]}
        # out = {}
        # offset = 0

        # for task, num_ch in self.outputs_desc.items():
        #     out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
        #     offset += num_ch

        return out
