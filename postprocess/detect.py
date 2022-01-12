import torch
from ops.nms import nms
from ops.box import decode


class Detect(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.background_label = cfg.DETECT.BACKGROUND_LABEL
        self.top_k = cfg.DETECT.TOP_K
        self.nms_thresh = cfg.DETECT.NMS_THRESH
        self.conf_thresh = cfg.DETECT.CONFIDENCE_THRESH
        self.variance = cfg.ENCODE.VARIANCE
        self.config = cfg

    @torch.no_grad()
    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors * 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch * num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors, 4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(self.config, loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
