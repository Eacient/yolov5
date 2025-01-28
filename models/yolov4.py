import os
import platform
import sys
from pathlib import Path
import yaml

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    
def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * nn.functional.softplus(x).tanh()

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, activation='mish', batch_norm=True, p=1):
        super().__init__()
        self.cv = nn.Conv2d(c1, c2, k, s, autopad(k), bias=not batch_norm)
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=1e-4) if batch_norm else nn.Identity()
        self.activation=activation
        if activation == 'mish':
            self.act = Mish()
        elif activation=='leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.cv(x)))
    def __len__(self):
        return 1
    def to_cfg(self):
        if self.batch_norm:
            cfg = """
[convolutional]
batch_normalize=1
filters={}
size={}
stride={}
pad={}
activation={}""".format(self.cv.out_channels, self.cv.kernel_size[0], self.cv.stride[0], 1, self.activation)
        else:
            cfg = """
[convolutional]
filters={}
size={}
stride={}
pad={}
activation={}""".format(self.cv.out_channels, self.cv.kernel_size[0], self.cv.stride[0], 1, self.activation)
        return cfg

class ResB(nn.Module):
    def __init__(self, c, e=0.5, activation='mish'):
        super().__init__()
        c_ = int(c*e)
        self.pc = Conv(c, c_, 1, activation=activation)
        self.cv = Conv(c_, c, 3, activation=activation)
        self.sc = """
[shortcut]
from=-{}
activation=linear"""
    
    def forward(self, x):
        return x + self.cv(self.pc(x))
        
    def __len__(self):
        return len(self.pc) + len(self.cv) + 1
        
    def to_cfg(self):
        return '\n'.join([self.pc.to_cfg(), self.cv.to_cfg(), self.sc.format(len(self))])

class CSP(nn.Module):
    def __init__(self, c, n=1, path=False, e=0.5, activation='mish', extra=""):
        super().__init__()
        c_ = c // 2 if path else c
        self.cv1 = Conv(c, c_, 1, activation=activation)
        self.cv2 = Conv(c, c_, 1, activation=activation)
        self.m = nn.Sequential(*[ResB(c_, e, activation) for _ in range(n)])
        self.cv3 = Conv(c_, c_, 1, activation=activation)
        self.cv4 = Conv(2*c_, c, 1, activation=activation)
        self.extra = extra
        self.route_split="""
[route]
layers = -{}"""
        self.route_merge = self.route_merge = """
[route]
layers = -1,-{}"""
    
    def forward(self, x):
        return self.cv4(torch.cat([self.cv3(self.m(self.cv2(x))), self.cv1(x)], 1))
        
    def __len__(self):
        return 2+sum(len(_) for _ in [self.cv1, self.cv2, self.cv3, self.cv4, *self.m])
    def to_cfg(self):
        cfg = '\n'.join([self.cv1.to_cfg(), 
                         self.route_split.format(1+len(self.cv1)), 
                         *[_.to_cfg() for _ in [self.cv2, *self.m, self.cv3]],
                         self.route_merge.format(2+len(self.cv2)+len(self.cv3)+
                                                 sum(len(_) for _ in self.m)), 
                         self.cv4.to_cfg(), self.extra])
        return cfg
        
class InvResB(nn.Module):
    def __init__(self, c, e=2, activation='mish', shortcut=False):
        super().__init__()
        c_ = int(c*e)
        self.cv = Conv(c, c_, 3, activation=activation)
        self.pc = Conv(c_, c, 1, activation=activation)
        self.shortcut = shortcut
        self.sc = """
[shortcut]
from=-{}
activation=linear"""
    
    def forward(self, x):
        return x+self.pc(self.cv(x)) if self.shortcut else self.pc(self.cv(x))
        
    def __len__(self):
        return len(self.cv) + len(self.pc) + (1 if self.shortcut else 0)
        
    def to_cfg(self):
        return '\n'.join([self.cv.to_cfg(), self.pc.to_cfg()]+
                         ([self.sc.format(len(self))] if self.shortcut else []))
        
class convMLP(nn.Module):
    def __init__(self, c1, c2, n=1, e=2, activation='leaky', shortcut=False):
        super().__init__()
        self.pc = Conv(c1, c2, 1, activation=activation)
        self.m = nn.Sequential(*[InvResB(c2, e, activation, shortcut) for _ in range(n)])
    
    def forward(self, x):
        return self.m(self.pc(x))
        
    def __len__(self):
        return sum(len(_) for _ in [self.pc, *self.m])

    def to_cfg(self):
        return '\n'.join([_.to_cfg() for _ in [self.pc, *self.m]])
# print(convMLP(128, 128, 1))
# print(len(convMLP(128, 128, 1)))
# print(convMLP(128, 128, 1).to_cfg())
    
class SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.ModuleList([nn.MaxPool2d(x, 1, padding=x//2) for x in (5, 9, 13)])
    
    def forward(self, x):
        xs = [m(x) for m in self.m][::-1]+[x]
        return torch.cat(xs, 1)
    
    def __len__(self):
        return 6
    
    def to_cfg(self):
        return """
[maxpool]
stride=1
size=5

[route]
layers=-2

[maxpool]
stride=1
size=9

[route]
layers=-4

[maxpool]
stride=1
size=13

[route]
layers=-1,-3,-5,-6"""
        
class UpSample(nn.Module):
    def __init__(self, c1, c2, stride=2, activation='leaky'):
        super().__init__()
        self.cv = Conv(c1, c2, 1, activation=activation)
        self.up = nn.Upsample(None, stride, 'nearest')
    
    def forward(self, x):
        return self.up(self.cv(x))
        
    def __len__(self):
        return len(self.cv) + 1
        
    def to_cfg(self):
        return self.cv.to_cfg() + '\n' + """
[upsample]
stride={}""".format(int(self.up.scale_factor))

class LateralCat(nn.Module):
    def __init__(self, f1, c1, c2, activation='leaky'):
        super().__init__()
        self.f1 = f1
        self.cv = Conv(c1, c2, 1, activation=activation)
        self.lateral = """
[route]
layers = {}"""
        self.cat = """
[route]
layers = -1, -{}"""
    
    def forward(self, x, x_l):
        return torch.cat([self.cv(x_l), x], 1)
        
    def __len__(self):
        return len(self.cv) + 2
        
    def to_cfg(self):
        return self.lateral.format(self.f1[1]) + '\n' + self.cv.to_cfg() + '\n' + self.cat.format(len(self))

class UpSampleFuse(nn.Module):
    def __init__(self, f1, c1, c2, c3, n=2, stride=2, e=2, activation='leaky', shortcut=False):
        super().__init__()
        self.upsample = UpSample(c1, c3, stride, activation)
        self.lateral = LateralCat((-1, f1[1]), c2, c3, activation)
        self.fuse = convMLP(2*c3, c3, n, e, activation, shortcut)
    def forward(self, x):
        x, x_l = x
        x = self.upsample(x)
        x = self.lateral(x, x_l)
        x = self.fuse(x)
        return x
        
    def __len__(self):
        return sum(len(_) for _ in [self.upsample, self.lateral, self.fuse])
        
    def to_cfg(self):
        return '\n'.join(_.to_cfg() for _ in [self.upsample, self.lateral, self.fuse])
    
class DownSampleFuse(nn.Module):
    def __init__(self, f1, c1, c2, c3, n=2, stride=2, e=2, activation="leaky", shortcut=False):
        super().__init__()
        self.f1 = f1
        self.downsample = Conv(c1, c3, 3, stride, activation)
        self.fuse = convMLP(c2+c3, c3, n, e, activation, shortcut)
        self.route1 = """
[route]
layers = {}"""
        self.route2 = """
[route]
layers = -1, {}"""

    def forward(self, x):
        x, x_l = x
        x = self.downsample(x)
        x = torch.cat([x, x_l], 1)
        x = self.fuse(x)
        return x
    
    def __len__(self):
        return len(self.downsample) + len(self.fuse) + 2
    
    def to_cfg(self):
        return '\n'.join([self.route1.format(self.f1[0]),
                          self.downsample.to_cfg(),
                          self.route2.format(self.f1[1]),
                          self.fuse.to_cfg()])

from torchvision.ops import box_iou, complete_box_iou_loss, distance_box_iou_loss
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
def focal_loss_with_logits(x, y, alpha=0.5, gamma=2):
    bce_loss = binary_cross_entropy_with_logits(x, y, reduction='none')
    pt = torch.exp(-bce_loss)
    return alpha * (1 - pt) ** gamma * bce_loss

class yolo(nn.Module):
    def __init__(self, anchors, nc, yolo_index, stride, mask, scale, new_coord=False):
        super(yolo, self).__init__()
        self.anchors_total = torch.Tensor(anchors)
        self.anchors = self.anchors_total[mask]
        self.index = yolo_index  # index of this layer in layers
        self.stride = stride  # layer stride
        self.na = len(self.anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.anchors_total = self.anchors_total / 608

        self.mask = mask
        self.scale = scale

        self.ignore_thresh = 0.7
        self.truth_thresh = 1.0
        self.iou_thresh = 0.213
        self.max_delta = 5
        self.cls_normalizer = 1
        self.focal_normalizer = 1
        self.iou_normalizer = 0.07
        self.new_coord = new_coord

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
        self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        io = p  # inference output
        
        if not self.training:
            torch.sigmoid_(io[..., 4:])
            if not self.scale or self.scale == 1:
                io[..., :2] = torch.sigmoid(io[..., :2])  # xy
            else:
                io[..., :2] = torch.sigmoid(io[..., :2]) * self.scale - 0.5*(self.scale-1)  # xy
            # to layer w and h
            io[..., :2] += self.grid # to absolute
            if not self.new_coord:
                io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            else:
                io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * self.anchor_wh

            io[..., :4] *= self.stride # to net w and h
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]
        else:
            return io

    def norm_ciou_loss(self, pred, truth, iou_loss='CIOU'):
        x1, y1, w1, h1 = pred.split(1,1)
        pred_box = torch.cat([x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2], 1)
        x2, y2, w2, h2 = truth.split(1,1)
        truth_box = torch.cat([x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2], 1)
        iou_loss = self.iou_normalizer * complete_box_iou_loss(pred_box, truth_box, reduction='mean')
        # iou_loss = self.iou_normalizer * distance_box_iou_loss(pred_box, truth_box, reduction='none')
        return iou_loss.sum()

    def box_iou(self, box1, box2):
        o_shape1 = box1.shape[:-1]
        o_shape2 = box2.shape[:-1]
        box1 = box1.reshape(-1, 4)
        box2 = box2.reshape(-1, 4)
        x1, y1, w1, h1 = box1.split(1, -1)
        x2, y2, w2, h2 = box2.split(1, -1)
        box1 = torch.cat((x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2), -1)
        box2 = torch.cat((x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2), -1)
        return box_iou(box1, box2).reshape(*o_shape1, *o_shape2)

    def shape_iou(self, shape1, shape2):
        o_shape1 = shape1.shape[:-1]
        o_shape2 = shape2.shape[:-1]
        shape1 = shape1.reshape(-1, 2)
        shape2 = shape2.reshape(-1, 2)
        w1, h1 = shape1.split(1, -1)
        w2, h2 = shape2.split(1, -1)
        x1 = torch.zeros_like(w1)
        x2 = torch.zeros_like(w2)
        box1 = torch.cat((x1, x1, w1, h1), -1)
        box2 = torch.cat((x2, x2, w2, h2), -1)
        # print(box1.shape, box2.shape)
        return box_iou(box1, box2).reshape(*o_shape1, *o_shape2)

    def accumulate_loss(self, indices):
        # 将所有索引组合成一个单一的张量
        indices = torch.stack(indices, dim=-1)  # Shape: [num_elements, num_dims]
        # 找到唯一索引和对应的计数
        unique_indices, inverse_indices, counts = torch.unique(indices, dim=0, return_inverse=True, return_counts=True)
        # # 初始化累加损失张量
        # accumulated_losses = torch.zeros((unique_indices.shape[0], losses.shape[1]), dtype=losses.dtype, device=losses.device)
        # accumulated_losses[inverse_indices, :] += losses
        # accumulated_losses.index_add_(0, inverse_indices, losses)

        return [x.squeeze(1) for x in unique_indices.split(1,1)], counts

    def compute_loss(self, out, targets):
        lwlh = torch.Tensor(list(out.shape)[2:4]).to(out.device)

        # out shape[bs, 3, lh, lw, 5], targets shape[bs, nt, 6]
        t_obj = torch.zeros(*out.shape[:-1], device=out.device) #[bs, 3, lh, lw]
        m_ign = torch.ones(*out.shape[:-1], device=out.device) #[bs, 3, lh, lw]
        c_tgt = torch.zeros(*out.shape[:-1], device=out.device, dtype=torch.long) #[bs, 3, lh, lw]
        l_iou = torch.zeros(1, device=out.device)
        tot_iou = 0

        pred_boxes = out[..., :4] #[bs,3,lh,lw,4]
        if not self.scale or self.scale == 1:
            pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2])  # xy
        else:
            pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2]) * self.scale - 0.5*(self.scale-1)  # xy
        # convert to layer scale and normalize
        pred_boxes[..., :2] = (pred_boxes[..., :2] + self.grid) / lwlh
        if not self.new_coord:
            pred_boxes[..., 2:4] = (torch.exp(pred_boxes[..., 2:4]) * self.anchor_wh) / lwlh
        else:
            pred_boxes[..., 2:4] = (torch.sigmoid(pred_boxes[..., 2:4]) * 2) ** 2 * self.anchor_wh / lwlh
        

        # negtive ignore ones 意义不明，一些不是最大anchor也不是形状最好的anchor     
        for b in range(out.shape[0]): #标量组部分索引
            truth_boxes = targets[targets[:,0]==b]
            if len(truth_boxes) == 0:
                continue
            truth_boxes = truth_boxes[:, 2:6] #[nt_b, 4]
            pred_boxes_b = pred_boxes[b] #[3, lh, lw, 4]
            pred_iou = self.box_iou(truth_boxes, pred_boxes_b) #[nt_b, 3, lh, lw]
            ignore_ind = pred_iou.max(0)[0] > self.ignore_thresh #[3, lh, lw] bool完全索引
            m_ign[b][ignore_ind] = 0

        truth_bids = targets[:, 0].type(torch.long) #[nt]
        gjgi = (targets[:, 2:4] *lwlh).type(torch.long) #[nt], [nt]
        gj = gjgi[:, 0]
        gi = gjgi[:, 1]
        truth_shapes = targets[:, 4:6] #[nt, 2]
        anchor_iou = self.shape_iou(truth_shapes, self.anchors_total.to(truth_shapes.device)) #[nt, 9]
        extra_pos_mask = anchor_iou[:, self.mask] > self.iou_thresh #[nt, 3]

        max_pos_ind = anchor_iou.max(1)[1] #[nt] #整型向量组部分索引
        max_ind = torch.sum(torch.stack([max_pos_ind==i for i in self.mask],1),1) > 0
        max_ind = torch.nonzero(max_ind).squeeze(1)
        max_pos_ind = anchor_iou[:, self.mask].max(1)[1]

        # max_pos_loss
        if len(max_ind) > 0:
            inds = (truth_bids[max_ind], max_pos_ind[max_ind], gi[max_ind], gj[max_ind]) #整型向量组部分索引
            l_iou += self.norm_ciou_loss(
                pred_boxes[*inds, :], #[nt, 4]
                targets[max_ind, 2:6] #[nt, 4]
            )
            unique_inds, tgt_cnt = self.accumulate_loss(inds)
            c_tgt[*unique_inds] = tgt_cnt
            t_obj[*inds] = 1
            m_ign[*inds] = 1
            tot_iou += torch.diag(self.box_iou(
                pred_boxes[*inds, :], #[nt, 4]
                targets[max_ind, 2:6] #[nt, 4]
            )).sum().item()

        # # extra_pos_loss
        # if len(targets) > 0:
        #     overlap_ind = torch.stack([torch.arange(max_pos_ind.shape[0], device=max_pos_ind.device), 
        #                             max_pos_ind], 1) #[nt, 2] 整型张量完全索引
        #     overlap_ind = overlap_ind[max_ind] #[n_m, 2]
        #     extra_pos_mask[overlap_ind] = False
        #     extra_ind = torch.nonzero(extra_pos_mask) # [nt_e, 2]
        #     if len(extra_ind) != 0:
        #         extra_ind, extra_pos_ind = [x.squeeze(1) for x in extra_ind.split(1, 1)] #[nt_e] [nt_e] 整型向量组部分索引
        #         inds = (truth_bids[extra_ind], extra_pos_ind, gi[extra_ind], gj[extra_ind])
        #         l_iou += self.norm_ciou_loss(
        #             pred_boxes[*inds], #[nt_e, 4]
        #             targets[extra_ind, :][:, 2:6] #[nt_e, 4]
        #         )
        #         unique_inds, tgt_cnt = self.accumulate_loss(inds)
        #         c_tgt[*unique_inds] += tgt_cnt
        #         t_obj[*inds] = 1
        #         m_ign[*inds] = 1
        #         tot_iou += torch.diag(self.box_iou(
        #             pred_boxes[*inds], #[nt_e, 4]
        #             targets[extra_ind, :][:, 2:6] #[nt_e, 4]
        #         )).sum().item()
        # else:
        extra_ind = []

        print('\ncheck_ignore', 'total', torch.ones_like(m_ign).sum().item(), 'ignored', (1-m_ign).sum().item(), 'positive', targets.shape[0], len(max_ind), len(extra_ind))
        print('[YOLO Matching]', 'rewritten', ((c_tgt>1)*(c_tgt-1)).sum().item())

        l_obj = self.cls_normalizer * (binary_cross_entropy_with_logits(out[...,4], t_obj, reduction='none') * m_ign).mean()
        # l_obj = self.focal_normalizer * (focal_loss_with_logits(out[...,4], t_obj) * m_ign).mean()


        return l_iou, l_obj, tot_iou/(len(max_ind)+len(extra_ind)) if len(max_ind) > 0 or len(extra_ind) > 0 else 0, len(max_ind)+len(extra_ind)

    def __len__(self):
        return 1

    def to_cfg(self):
        return """
[yolo]
mask = {}
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
#classes=80
classes=1
num=9
scale_x_y = {}
jitter=.3
ignore_thresh = {}
truth_thresh = {}
iou_thresh={}
max_delta={}
iou_loss=ciou
cls_normalizer={}
iou_normalizer={}
nms_kind=greedynms
beta_nms=0.6
""".format(','.join([str(_) for _ in self.mask]), self.scale, 
    self.ignore_thresh, self.truth_thresh, self.iou_thresh, 
    self.max_delta, self.cls_normalizer, self.iou_normalizer)
    

class YOLOv4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        with open(cfg, encoding="ascii", errors="ignore") as f:
            self.yaml = yaml.safe_load(f)  # model dict
        self.model = parse_model(self.yaml)
        self.multi_layer = 'pan4' in self.model

    def get_param_groups(self):
        g0 = []
        g1 = []
        g2 = []
        for n, w in self.named_parameters():
            if not w.requires_grad:
                continue
            if '.weight' in n:
                if '.bn' in n:
                    g1.append(w)
                else:
                    g0.append(w)
            elif '.bias' in n:
                g2.append(w)
        return {'params': g0}, {'params': g1, 'weight_decay': 0}, {'params': g2, 'weight_decay': 0}


    def forward(self, x):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        # down-top
        x = self.model.b0(x)
        x = self.model.b1(x)
        x = self.model.b2(x)
        x3 = self.model.b3(x)
        x4 = self.model.b4(x3)
        x = self.model.b5(x4)
        x = self.model.spp(x)
        # top-down
        x_f5 = self.model.fpn5(x)
        x_f4 = self.model.fpn4([x_f5, x4])
        x_f3 = self.model.fpn3([x_f4, x3])
        if self.multi_layer:
            # pan
            x_p4 = self.model.pan4([x_f3, x_f4])
            x_p5 = self.model.pan5([x_p4, x_f5])
            # out
            out3 = self.model.yolo3(x_f3)
            out4 = self.model.yolo4(x_p4)
            out5 = self.model.yolo5(x_p5)
            return [out3, out4, out5]
        else:
            out3 = self.model.yolo3(x_f3)
            return [out3]
        # out3 = self.model.yolo3(x_f3)
        # return [out3]

    def to_cfg(self):
        return """
[net]
# Testing
batch=1
subdivisions=1
# Training
#batch=64
#subdivisions=16
#subdivisions=64
width=608
height=608
channels=3
momentum=0.949
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
#max_batches = 500500
max_batches=6000
policy=steps
#steps=400000,450000
steps=4800,5400
scales=.1,.1

#cutmix=1
mosaic=1""" + '\n'\
    +'\n'.join(['\n'.join(mi.to_cfg() for mi in m) for _, m in self.model.items()])
    
    def load_weights(self, weight_path):
        # Read weights file
        with open(weight_path, 'rb') as f:
            # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
            self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
            self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

            weights = np.fromfile(f, dtype=np.float32)  # the rest are weights
        
        print('total weights', weights.shape[0])
        ptr = 0
        for name, module in self.named_modules():
            if not isinstance(module, Conv):
                continue
            print('loading', name, 'from', ptr)
            if(ptr >= len(weights)):
                print('end of file')
                break
            if module.batch_norm:
                bn = module.bn
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = module.cv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(module.cv.bias)
                module.cv.bias.data.copy_(conv_b)
                ptr += nb

            nw = module.cv.weight.numel()  # number of weights
            module.cv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(module.cv.weight))
            ptr += nw
        print('total loaded', ptr)

    def save_weights(self, weight_path):
        with open(weight_path, 'wb') as f:
            self.version.tofile(f)  # (int32) version info: major, minor, revision
            self.seen.tofile(f)  # (int64) number of images seen during training
            for name, module in self.named_modules():
                if not isinstance(module, Conv):
                    continue
                print('saving', name)
                if module.batch_norm:
                    module.bn.bias.data.cpu().numpy().tofile(f)
                    module.bn.weight.data.cpu().numpy().tofile(f)
                    module.bn.running_mean.data.cpu().numpy().tofile(f)
                    module.bn.running_var.data.cpu().numpy().tofile(f)
                else:
                    module.cv.bias.data.cpu().numpy().tofile(f)
                module.cv.weight.data.cpu().numpy().tofile(f)
        with open(weight_path, 'rb') as f:
            self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
            self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # the rest are weights
        print('total weights', weights.shape[0])

def parse_model(cfg):
    val = '12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401'
    anchors = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
    yolo_index = -1
    strides = [8, 16, 32, 64, 128]
    nc = 1

    modules = {}

    layers = 0
    ds = {}
    in_c = cfg['in_c']
    embed = cfg['embed']
    neck_shortcut = cfg['neck_shortcut']
    for name, stage in cfg['model'].items():
        blocks = []
        for block in stage:
            depth = block[1]
            block_class = eval(block[2])
            args = []
            for arg in block[3]:
                if isinstance(arg, str):
                    try:
                        arg = eval(arg)
                    except:
                        arg = arg
                        print('str arg', arg)
                args.append(arg)
            print(args)
            # has depth
            if block[2] in ['CSP', 'convMLP', 'ResB', 'InvResB', 'UpSampleFuse', "DownSampleFuse"]:
                if block[2] in ['UpSampleFuse', "DownSampleFuse"]:
                    args.insert(3, depth)
                elif block[2] in ['CSP']:
                    args.insert(1, depth)
                else:
                    args.insert(2, depth)
            # has multi input
            if block[2] in ['route', 'UpSampleFuse', 'DownSampleFuse', 'LateralCat']:
                f = [block[0],] if not isinstance(block[0], list) else block[0]
                f_1 = []
                for fi in f:
                    if isinstance(fi, str):
                        f_1.append(ds[fi]-1)
                    else:
                        f_1.append(fi)
                args.insert(0, f_1)
            # yolo
            if block[2] == 'yolo':
                yolo_index += 1
                mask = args[0]
                anchor = anchors[mask]
                stride = strides[yolo_index]
                scale = args[1]
                new_coord = args[2]
                args = [anchors, nc, yolo_index, stride, mask, scale, new_coord]
            print(block[2], args)
            module = block_class(*args)
            layers += len(module)
            blocks.append(module)
        modules[name] = nn.Sequential(*blocks)
        ds[name] = layers
    return nn.ModuleDict(modules)


def get_yolo_detections_batch(out, conf=0.25, profile=False):
    if profile:
        counts = [0 for _ in range(out[0][0].shape[0])]
        conf_preds = [[] for _ in range(len(out))]
        # print('counts', len(counts))
        for l, l_out in enumerate(out):
            pred = l_out[0]
            l_conf_preds = conf_preds[l]
            conf_mask = (pred[:, :, 4] > conf)
            for batch, (b_pred, b_mask) in enumerate(zip(pred, conf_mask)):
                l_conf_preds.append(b_pred[b_mask].detach())
                counts[batch] += l_conf_preds[-1].shape[0]
                print('layer', l+3, 'batch', batch, ':', l_conf_preds[-1].shape)
        conf_preds = [torch.cat(layers, 0) for layers in zip(*conf_preds)] 
        return torch.cat(conf_preds, 0), sum([[i]*c for i, c in enumerate(counts)], [])
    else:
        preds = torch.cat([o[0] for o in out], 1)
        idxs = torch.Tensor(sum([[i]*len(p) for i,p in enumerate(preds)], [])).to(preds.device)
        preds = preds.flatten(0,1)
        mask = preds[:, 4] > conf
        preds = preds[mask]
        idxs = idxs[mask]
        return preds, idxs

def convert_dets(dets):
    bx, by, bw, bh = dets[:, :4].split(1, dim=1)
    scores = dets[:, 4]
    x1 = bx - bw/2
    x2 = bx + bw/2
    y1 = by - bh/2
    y2 = by + bh/2
    return torch.cat([x1,y1,x2,y2], 1), scores

def box_to_coco(box, score, scale=608, orig=1000):
    x1, y1, x2, y2 = box/scale
    contour_info_area = (x2-x1) * (y2-y1) * (orig)**2
    left, top, right, bottom = round(x1*orig), round(y1*orig), round(x2*orig), round(y2*orig)
    contour_plist = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return {
        'contour_class': str(0), 'contour_score': str(score),
        'contour_info_area': str(contour_info_area), 'contour_plist': contour_plist
    }

import json
def save_json(input_path, boxes, scores, out_dir, test_dir):
    if len(boxes) == 0: return False
    contour_dicts = {str(i): box_to_coco(box, score) for i, (box, score) in enumerate(zip(boxes, scores))}
    json_dict = {
        'roi_path': input_path,
        'contours': contour_dicts
    }
    
    save_path = out_dir + input_path[len(test_dir):].replace('.jpg', '.json')
    # # # 确保输出目录存在
    # # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
    print('saved', save_path)
    return True

import numpy as np

def prune(model):
    import torch_pruning as tp
    print(model)
    for p in model.parameters():
        p.requires_grad_(True)

    device = torch.device(0)
    example_inputs = torch.randn(1, 3, 608, 608).to(device)

    ignored_layers = [model.model.yolo3[1], model.model.yolo3[2], model.model.yolo4[1], model.model.yolo4[2], model.model.yolo5[1], model.model.yolo5[2]]
    ignored_layers += [model.model.b0, model.model.b1]
    # ignored_layers += [model.model.b2, model.model.b3, model.model.b4, model.model.b5]
    # ignored_layers += [model.model.fpn5, model.model.fpn4, model.model.fpn3, model.model.pan4, model.model.pan5]
    # ignored_layers += [model.model.yolo3[0], model.model.yolo4[0], model.model.yolo5[0]]
    # for name, m in model.named_modules():
    #     print(name)
    #     if isinstance(m, yolo):
    #         ignored_layers.append(m)
    # print(ignored_layers)

    # imp = tp.importance.MagnitudeImportance(p=2) # L2 norm pruning
    # iterative_steps = 3 # progressive pruning
    # pruner = tp.pruner.MagnitudePruner(
    #     model,
    #     example_inputs,
    #     importance=imp,
    #     iterative_steps=iterative_steps,
    #     pruning_ratio=0.7, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    #     ignored_layers=ignored_layers,
    # )

    # imp = tp.importance.GroupTaylorImportance()
    imp = tp.importance.GroupNormImportance(2)
    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        iterative_steps=5,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    )
    
    # Taylor expansion requires gradients for importance estimation
    # loss = model(example_inputs).sum() # A dummy loss, please replace this line with your loss function and data!
    # loss.backward() # before pruner.step()

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for g in pruner.step(interactive=True):
        print(g)
        g.prune()

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    with open('after.cfg', 'w+') as f:
        f.write(model.to_cfg())
    model.save_weights('pruned.weights')
    print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))
    print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))

def test_time(model, batch_size=1, half=False, repetitions=300):
    dummy_input = torch.zeros(batch_size, 3, 608, 608).cuda()
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    dummy_input = dummy_input
    if half:
        dummy_input = dummy_input.type(torch.float16)
        model = model.half()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    print('start testing ...\n')
    timings = np.zeros((repetitions, 1))
    for rep in range(repetitions):
        starter.record()
        with torch.no_grad():
            _ = model(dummy_input)
            # _ = model.backbone.bottom_up(torch.zeros(8, 3, 800, 800).cuda())
        ender.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time
    print(torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9, torch.cuda.max_memory_reserved()/1e9)
    ####################################################################################

    avg = timings.sum()/repetitions/batch_size
    print('\navg={}\n'.format(avg))

from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import time
from torchvision.ops import batched_nms
from test_loader_dali import get_dali_iter
import copy
def model_infer(model, test_dir, out_dir, batch_size=16, half=False, save=False, label_dir=False, clear=False):

    roi_paths = glob.glob(test_dir+'/*/*.jpg')
    print(len(roi_paths))
    if save and not label_dir:
        for roi_path in roi_paths:
            save_path = out_dir + roi_path[len(test_dir):]
            save_dir = save_path[:-len(save_path.split('/')[-1])]
            os.makedirs(save_dir, exist_ok=True)

    loader = get_dali_iter(test_dir, batch_size, num_threads=8, device_id=0)

    labels = None
    if label_dir:
        label_cache = test_dir+'/labels.cache'
        # read cache
        if os.path.exists(label_cache) and not clear:
            labels = np.load(label_cache)
            if(len(labels) != len(roi_paths)):
                print('rebuild val label cache')
                os.remove(label_cache)
                labels = None
        # build labels and write cache
        if not labels:
            label_paths = glob.glob(test_dir+'/*/*.txt')
            assert(len(label_paths) == len(roi_paths))
            collect = []
            for i, p in enumerate(label_paths):
                with open(p, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        collect.append(np.array([i, *[float(x) for x in parts]]))
            labels = np.stack(collect)
            np.save(label_cache, labels)
            obj_inds = np.unique(labels[:,1]).tolist()
        nt = {o:0 for o in obj_inds}
        nb = {o:0 for o in obj_inds}
        tp_r = {o:0 for o in obj_inds}
        tp_p = {o:0 for o in obj_inds}
            
    total_counts = 0
    if half:
        model = copy.deepcopy(model).half()
    start = time.time()
    for i, data in enumerate(loader):
        if (i * batch_size >= len(roi_paths)): break
        input_paths = roi_paths[batch_size*i:batch_size*(i+1)] if batch_size*(i+1) < len(roi_paths) else roi_paths[batch_size*i:]
        print(batch_size*i, batch_size*(i+1), len(input_paths))
        # input = []
        # for p in input_paths:
        #     input.append(get_test_input(p))
        # input = torch.stack(input, 0)
        input = data[0]['img']
        data_time = time.time() - start
        infer_start = time.time()
        # input = load_inputs(input_paths)
        if half:
            input = input.type(torch.float16)
        with torch.no_grad():
            out = model(input)
            dets, idxs = get_yolo_detections_batch(out)
        infer_time = time.time() - infer_start

        post_start = time.time()
        boxes, scores = convert_dets(dets)
        retain_ind = batched_nms(boxes, scores, idxs, 0.6)
        total_counts += len(retain_ind)
        post_time = time.time() - post_start

        label_start = time.time()
        if label_dir:
            pred = boxes[retain_ind]
            bid = idxs[retain_ind]
            for iter in range(len(input_paths)):
                b = iter + i*batch_size
                iter_label = labels[labels[:,0]==b][:, 1:] #[nt, 5]
                obj_label = torch.Tensor(iter_label[:, 0]).to(pred.device) #[nt]
                box_label = torch.Tensor(iter_label[:, 1:]).to(pred.device)
                x, y, w, h = box_label.split(1,1)
                iter_truth = torch.concat([x-w/2, y-h/2, x+w/2, y+h/2], 1) * 608 #[nt, 4]
                iter_pred = pred[bid==iter] #[np, 4]

                for o in obj_inds:
                    nt[o] += (iter_label==o).sum().item()

                if len(iter_truth) == 0 or len(iter_pred) == 0:
                    continue
                iter_ious = box_iou(iter_pred, iter_truth) #[np, nt]
                b_obj_label = obj_label[iter_ious.max(1)[1]] # [np]
                tp_r_total = iter_ious.max(0)[0] > 0.33 #[nt]
                tp_p_total = iter_ious.max(1)[0] > 0.33 #[np]

                for o in obj_inds:
                    nb[o] += (b_obj_label==o).sum().item()
                    tp_r[o] += (tp_r_total[obj_label==o]).sum().item()
                    tp_p[o] += (tp_p_total[b_obj_label==o]).sum().item()
        label_time = time.time() - label_start

        save_start = time.time()
        if save and not label_dir:
            boxes = boxes[retain_ind].cpu().numpy()
            scores = scores[retain_ind].cpu().numpy()
            idxs = idxs[retain_ind].cpu().numpy()
            ids = [idxs==i for i in range(len(input_paths))]
            with ThreadPoolExecutor(max_workers=16) as executor:  # 可以调整max_workers的数量
                # 提交所有任务到线程池
                futures = [
                    executor.submit(save_json, input_path, boxes[id], scores[id], out_dir, test_dir)
                    for input_path, id in zip(input_paths, ids)
                ]
                
                # 随着future完成，可以在这里进行一些进度跟踪或者错误处理
                for future in as_completed(futures):
                    try:
                        total_counts += future.result()  # 这会抛出异常如果任务执行过程中有异常发生
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
        save_time = time.time() - save_start

        print('data', data_time*1000/batch_size, 
              'infer', infer_time*1000/batch_size, 
              'post', post_time*1000/batch_size, 
              'label', label_time*1000/batch_size,
              'save_json', save_time*1000/batch_size, 
              'total', total_counts)
        start = time.time()
    if label_dir:
        recall = {o: tp_r[o] / nt[o] for o in obj_inds}
        precision = {o: (tp_p[o] / nb[o] if nb[o] > 0 else 0) for o in obj_inds}
        print(recall, precision, total_counts)
        return recall, precision, total_counts
    else:
        return -1, -1, total_counts



from utils.dataloaders import LoadImagesAndLabels, create_dataloader
from torch.optim import SGD
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
def debug_train(model, train_dir, val_dir, batch_size=8, resume=None, name=None):
    n_epochs = 600
    cycle = 100
    warmup = 10
    batching=64
    lr = 0.1
    wd = 5e-4
    momentum = 0.934
    # resume='/root/autodl-tmp/Torch-Pruning/exps/runs4/epoch_500.pt'


    def update_mavg(avg_dict, k, avg_size, new, online_size):
        avg_dict[k] = avg_dict[k] * (avg_size) / (avg_size+online_size) + new / (avg_size+online_size)
        return avg_size + online_size
    if resume:
        state_dict = torch.load(resume, map_location='cpu')
        model.load_state_dict(state_dict)

    out_dir_base = 'exps/runs'
    runs = 0
    out_dir = out_dir_base + str(runs) if name is None else out_dir_base + '/' + name
    while os.path.exists(out_dir):
        runs += 1
        out_dir = out_dir_base + str(runs)
    os.makedirs(out_dir)

    # 获取一个logger对象
    logging.basicConfig(level=logging.DEBUG, filename=out_dir+'/train.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('my_logger')
    writer = SummaryWriter(out_dir)

    param_groups = model.get_param_groups()
    optimizer = SGD(param_groups, lr=lr, weight_decay=wd, momentum=momentum)
    lf = lambda x: (((1 + math.cos((x%cycle) * math.pi / cycle)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # dataset = LoadImagesAndLabels(train_dir, img_size=608, batch_size=batch_size,
    #                                 single_cls=True, prefix='dataset')
    # img, target, path, shape = dataset[0]
    # print(img.shape, target.shape, shape)
    # print(img)
    # print(target)
    loader, dataset = create_dataloader(train_dir, 608, batch_size, 32, augment=True, hyp={
        'degrees': 0,
        'translate': 0.3,
        'hsv_h': 0.1,
        'hsv_s': 1.5,
        'hsv_v': 1.5,
        'scale': 0,
        'shear': 0,
        'perspective': 0,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0,
        'copy_paste': 0,
        'blur': 0,
        # 'hsvb': (0.1, 1.5, 1.5, 1),
    })
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            model.eval()
            recall, precision, total_count = model_infer(model, val_dir, out_dir, batch_size=16, half=False, save=False, label_dir=True, clear=False)
            logger.info(' '.join([
                '[EVAL]', str(epoch), 
                str(recall), 
                str(precision),
                str(total_count)
            ]))
            for k in recall.keys():
                writer.add_scalar('Val/recall'+str(int(k)), recall[k], epoch)
                writer.add_scalar('Val/precision'+str(int(k)), precision[k], epoch)
            writer.add_scalar('Val/total_count', total_count, epoch)
            torch.save(model.state_dict(), out_dir+'/last.pt')
            if epoch % cycle == 0 and epoch > 0:
                torch.save(model.state_dict(), out_dir+'/epoch_{}.pt'.format(epoch))

        model.train()
        sumdict = defaultdict(int)
        for i, (img, target, path, shape) in enumerate(loader):
            ni = len(loader) * epoch + i
            # if epoch < warmup and resume is None:
            #     for j, x in enumerate(optimizer.param_groups):
            #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #         # x['lr'] = np.interp(ni, [0, len(loader)*warmup], [0.01 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            #         x['lr'] = np.interp(ni, [0, len(loader)*warmup], [0.0, x['initial_lr'] * lf(epoch)])
            #         if 'momentum' in x:
            #             x['momentum'] = np.interp(ni, [0, len(loader)*warmup], [0.9, momentum])

            img = img.cuda()/255
            img = torch.flip(img, [1])
            if model.multi_layer:
                out3, out4, out5 = model(img)
            else:
                out3 = model(img)[0]
            target = target.cuda()

            l_iou3, l_obj3, iou3, pos3 = model.model.yolo3[-1].compute_loss(out3, target)
            # l_iou4, l_obj4, iou4, pos4 = model.model.yolo4[-1].compute_loss(out4, target)
            # l_iou5, l_obj5, iou5, pos5 = model.model.yolo5[-1].compute_loss(out5, target)

            with torch.no_grad():  # 确保不会计算梯度
                print('\n', l_iou3.item(), l_obj3.item(), iou3, pos3)
                print(optimizer.param_groups[0]['lr'])
                update_mavg(sumdict, 'l_iou3', i, l_iou3.item(), 1)
                update_mavg(sumdict, 'l_obj3', i, l_obj3.item(), 1)
                update_mavg(sumdict, 'iou3', i, iou3, 1)
                update_mavg(sumdict, 'pos3', i, pos3, 1)
                update_mavg(sumdict, 'lr', i, optimizer.param_groups[0]['lr'], 1)
            
            # loss = l_iou3+l_obj3+l_iou4+l_obj4+l_iou5+l_obj5
            loss = l_iou3+l_obj3

            loss /= (batching / batch_size)
            loss.backward()
            if (i+1) * batch_size % batching == 0:
                optimizer.step()
                optimizer.zero_grad()
        print('[EPOCH]', epoch, 'liou:{:.2f}'.format(sumdict['l_iou3']), 'lobj:{:.4f}'.format(sumdict['l_obj3']), 'iou:{:.2f}%'.format(100*sumdict['iou3']))
        logger.info(' '.join(['[EPOCH]', str(epoch), 
                              str(sumdict['l_iou3']), 
                              str(sumdict['l_obj3']),
                              str(sumdict['iou3']),
                              'pos', str(sumdict['pos3'])]))
        for k in sumdict.keys():
            if 'l_' in k:
                writer.add_scalar('Loss/'+k, sumdict[k], epoch)
            elif 'iou' in k or 'pos' in k:
                writer.add_scalar('Metrics/'+k, sumdict[k], epoch)
            else:
                writer.add_scalar('Params/'+k, sumdict[k], epoch)
        scheduler.step()
    torch.save(model.state_dict(), out_dir+'/model.pt')



if __name__ == "__main__":
    # cfg = '/root/autodl-tmp/Torch-Pruning/examples/yolov5/yolov5/models/parse.yaml'
    cfg = '/root/autodl-tmp/Torch-Pruning/examples/yolov5/yolov5/models/compress.yaml'
    # cfg = '/root/autodl-tmp/Torch-Pruning/examples/yolov5/yolov5/models/custom.yaml'
    weight_file = '/root/autodl-tmp/B1_fudan_univ/train_test/yolov4.conv.137'
    # weight_file = '/root/autodl-tmp/B1_fudan_univ/train_test/backup/yolov4-custom_best_recall.weights'
    # weight_file = '/root/autodl-tmp/B1_fudan_univ/train_test/backup/yolov4-custom_best_map_recall.weights'

    model = YOLOv4(cfg)
    model = model.cuda()
    with open('before.cfg', 'w+') as f:
        f.write(model.to_cfg())
    # model.load_weights('/root/autodl-tmp/B1_fudan_univ/train_test/backup/yolov4-custom_best_recall.weights')
    model.load_weights(weight_file)

    train_dir = '/root/autodl-tmp/B1_fudan_univ/data/train'
    val_dir = '/root/autodl-tmp/B1_fudan_univ/data/test'

    debug_train(model, train_dir, val_dir)


    # batch_size = 16
    # half = True
    #half 90.89, 96.6 6551, 986104
    #compress 90.89, 96.8 5892, 900209
    #compress-half 90.0, 96.6 5465, 986024
    #orig 90.89 96.8 5893, 959575
    # speed
    # 1 4 8 16 32
    #orig 13.92 11.31 10.53 10.1
    #half 12.66 x 6.07 5.63 5.55
    #compress 12.17 10.04 9.47 9.08 8.76
    #compress-half 10.82 6.09 5.56 5.19 5.09
    #save-json 1.7ms


    # model.eval()
    # total_norm=0
    # for n, w in model.model.yolo4.named_parameters():
    #     print(n, w.abs().sum())
    #     total_norm += w.abs().sum()
    # print(total_norm)
    # test_dir = '/root/autodl-tmp/B1_fudan_univ/slide_roi_for_B1_C1'
    # out_dir = 'outupt-test-half'
    # model_infer(model, test_dir, out_dir, 1, True, label_dir=False)
    # test_dir = '/root/autodl-tmp/B1_fudan_univ/data/test'
    # out_dir = 'output-test-save'
    # model_infer(model, test_dir, out_dir, 32, False, label_dir=True)

    # import glob
    # # input = get_test_input('/root/autodl-tmp/B1_fudan_univ/data/valid/C01KNJTHI00363S01/C01KNJTHI00363S01_3840_21956_class_0.jpg')
    # # input2 = get_test_input('/root/autodl-tmp/B1_fudan_univ/data/valid/C01KNJTHI00363S01/C01KNJTHI00363S01_4840_16956_class_0.jpg')
    # # input = torch.stack([input, input2],0)
    # test_paths = glob.glob('/root/autodl-tmp/B1_fudan_univ/data/test/*/*.jpg')

    # for p in test_paths:
    #     input = get_test_input(p)
    #     input = torch.stack([input],0)

    #     with torch.no_grad():
    #         out = model(input.to(torch.device(0)))
    #         # print(out[0][0].shape)
    #         dets = get_yolo_detections_batch(out)
    #         greeedy_nms(dets, 0.6)

    # test_time(model, 1, True)

    

    ####################################################################################
    # prune(model)


    