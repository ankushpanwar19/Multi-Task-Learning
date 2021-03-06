import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p
from mtl.datasets.definitions import *


class ModelBranchedArch(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())
        ch_out_seg=outputs_desc[MOD_SEMSEG]
        ch_out_depth=outputs_desc[MOD_DEPTH]

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256,cfg)
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256,cfg)

        self.decoder_seg = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_seg,cfg)
        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_depth,cfg)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        # features_tasks = self.aspp(features_lowest)

        features_task_seg = self.aspp_seg(features_lowest)
        features_task_depth = self.aspp_depth(features_lowest)
        
        predictions_4x_seg, _ = self.decoder_seg(features_task_seg, features[4])
        predictions_4x_depth, _ = self.decoder_depth(features_task_depth, features[4])
        # predictions_4x, _ = self.decoder(features_tasks, features[4])

        predictions_1x_seg = F.interpolate(predictions_4x_seg, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_depth = F.interpolate(predictions_4x_depth, size=input_resolution, mode='bilinear', align_corners=False)
        # predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)
        
        out={MOD_SEMSEG:predictions_1x_seg,MOD_DEPTH:predictions_1x_depth}
        # out = {}
        # offset = 0

        # for task, num_ch in self.outputs_desc.items():
        #     out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
        #     offset += num_ch

        return out
