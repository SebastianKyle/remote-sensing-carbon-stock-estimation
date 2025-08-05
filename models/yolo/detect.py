import torch
import torch.nn as nn
import math
from .conv import Conv, DWConv
from .block import DFL
from .tal import make_anchors, dist2bbox

class Detect(nn.Module):
    """YOLOv12 Detection Head"""
    def __init__(self, ch, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = 16
        self.nl = len(ch)
        self.no = num_classes + self.reg_max * 4
        self.stride = None

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(ch[0], c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ),
            nn.Sequential(
                Conv(ch[1], c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ),
            nn.Sequential(
                Conv(ch[2], c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
        ])

        c3 = max(ch[0], min(self.num_classes, 100))
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(
                    DWConv(ch[0], ch[0], 3),
                    Conv(ch[0], c3, 1)
                ),
                nn.Sequential(
                    DWConv(c3, c3, 3),
                    Conv(c3, c3, 1)
                ),
                nn.Conv2d(c3, num_classes, 1)
            ),
            nn.Sequential(
                nn.Sequential(
                    DWConv(ch[1], ch[1], 3),
                    Conv(ch[1], c3, 1)
                ),
                nn.Sequential(
                    DWConv(c3, c3, 3),
                    Conv(c3, c3, 1)
                ),
                nn.Conv2d(c3, num_classes, 1)
            ),
            nn.Sequential(
                nn.Sequential(
                    DWConv(ch[2], ch[2], 3),
                    Conv(ch[2], c3, 1)
                ),
                nn.Sequential(
                    DWConv(c3, c3, 3),
                    Conv(c3, c3, 1)
                ),
                nn.Conv2d(c3, num_classes, 1)
            )
        ])
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.bias_initialized = False

    def forward(self, x):
        # if self.stride is None:
        self._initialize_stride(x)
        if self.training: 
            self.bias_init()

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        return self._inference(x)

    def _initialize_stride(self, x):
        """Compute strides based on feature map sizes."""
        self.stride = torch.tensor([640 / xi.shape[-2] for xi in x])

    def _inference(self, x):

        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) # (B, num_classes + 4 * reg_max, H x W)

        self.anchors, self.stride = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        pred_dist = box
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0), xywh=False) * self.stride

        return torch.cat((dbox, cls), 1), pred_dist, x
        
    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        if self.bias_initialized == False: 
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.num_classes] = math.log(5 / m.num_classes / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            self.bias_initialized = True