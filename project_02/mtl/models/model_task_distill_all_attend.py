import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p,\
    SelfAttention, SqueezeAndExcitation, DecoderDeeplabV3pAllConnect,DecoderDeeplabV3pSelfAtten,\
    DecoderDeeplabV3pAllConnectAllAttend, SelfAttentionDecodeAllScale
from mtl.datasets.definitions import *


class ModelTaskDistillAllConnectAllAttend(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())
        ch_out_seg=outputs_desc[MOD_SEMSEG]
        ch_out_depth=outputs_desc[MOD_DEPTH]
        ch_attention = 256
        self.add_se = cfg.add_se

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, True, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256,cfg)
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256,cfg)

        # self.decoder_seg1 = DecoderDeeplabV3pAllConnectAllAttend(256, 64, ch_out_encoder_4x, 128, ch_out_seg)
        # self.decoder_depth1 = DecoderDeeplabV3pAllConnectAllAttend(256, 64, ch_out_encoder_4x, 128, ch_out_depth)
        self.decoder_seg1 = DecoderDeeplabV3pAllConnectAllAttend(256, 64, ch_out_encoder_4x, 512, ch_out_seg)
        self.decoder_depth1 = DecoderDeeplabV3pAllConnectAllAttend(256, 64, ch_out_encoder_4x, 512, ch_out_depth)

        self.attend_decode_seg = SelfAttentionDecodeAllScale(256, 256, ch_out_seg)
        self.attend_decode_depth = SelfAttentionDecodeAllScale(256, 256, ch_out_depth)


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_task_seg = self.aspp_seg(features_lowest)
        features_task_depth = self.aspp_depth(features_lowest)

        predictions_seg1_dict, features_seg_dict = self.decoder_seg1(features_task_seg, features, input_resolution)
        predictions_depth1_dict, features_depth_dict = self.decoder_depth1(features_task_depth, features, input_resolution)


        predictions_2x_seg2 = self.attend_decode_seg(features_seg_dict, features_depth_dict, input_resolution)
        predictions_2x_depth2 = self.attend_decode_depth(features_depth_dict, features_seg_dict, input_resolution)

        predictions_1x_seg2 = F.interpolate(predictions_2x_seg2, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_depth2 = F.interpolate(predictions_2x_depth2, size=input_resolution, mode='bilinear', align_corners=False)

        prediction_level1 = {MOD_SEMSEG: predictions_seg1_dict, MOD_DEPTH: predictions_depth1_dict}
        prediction_level2 = {MOD_SEMSEG: predictions_1x_seg2, MOD_DEPTH: predictions_1x_depth2}


        out = {}
        for task, num_ch in self.outputs_desc.items():
            preds = []
            for k,v in prediction_level1[task].items():
                preds.append(v)
            preds.append(prediction_level2[task])
            out[task] = preds

        return out
