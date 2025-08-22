import torch
import torch.nn.functional as F


# Helper: pad/crop to target size (H, W)
def _pad_or_crop(imgs: torch.Tensor, target_h: int, target_w: int):
    # imgs: B x H x W x C in 0..1
    b, h, w, c = imgs.shape
    # center crop if larger
    if h > target_h:
        top = (h - target_h) // 2
        imgs = imgs[:, top:top+target_h, :, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        imgs = imgs[:, :, left:left+target_w, :]
        w = target_w
    # pad if smaller
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h or pad_w:
        # F.pad uses (W_left, W_right, H_top, H_bottom) for 4D NCHW, we'll permute
        imgs_nchw = imgs.permute(0, 3, 1, 2)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        imgs_nchw = F.pad(imgs_nchw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
        imgs = imgs_nchw.permute(0, 2, 3, 1)
    return imgs




def _resize_fit_letterbox(imgs: torch.Tensor, target_h: int, target_w: int):
    # Keeps aspect ratio; letterboxes to target
    b, h, w, c = imgs.shape
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    imgs_nchw = imgs.permute(0, 3, 1, 2)
    imgs_resized = F.interpolate(imgs_nchw, size=(new_h, new_w), mode='bilinear', align_corners=False)
    imgs_resized = imgs_resized.permute(0, 2, 3, 1)
    # pad to target
    return _pad_or_crop(imgs_resized, target_h, target_w)




def _resize_stretch(imgs: torch.Tensor, target_h: int, target_w: int):
    imgs_nchw = imgs.permute(0, 3, 1, 2)
    imgs_resized = F.interpolate(imgs_nchw, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return imgs_resized.permute(0, 2, 3, 1)




class ConcatImageBatches:
}