import torch
import torch.nn.functional as F

def _pad_or_crop(imgs: torch.Tensor, target_h: int, target_w: int):
    b, h, w, c = imgs.shape
    if h > target_h:
        top = (h - target_h) // 2
        imgs = imgs[:, top:top+target_h, :, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        imgs = imgs[:, :, left:left+target_w, :]
        w = target_w
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h or pad_w:
        imgs_nchw = imgs.permute(0, 3, 1, 2)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        imgs_nchw = F.pad(imgs_nchw, (pad_left, pad_right, pad_top, pad_bottom),
                          mode='constant', value=0.0)
        imgs = imgs_nchw.permute(0, 2, 3, 1)
    return imgs

def _resize_fit_letterbox(imgs, target_h, target_w):
    b, h, w, c = imgs.shape
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    imgs_nchw = imgs.permute(0, 3, 1, 2)
    imgs_resized = F.interpolate(imgs_nchw, size=(new_h, new_w), mode='bilinear', align_corners=False)
    imgs_resized = imgs_resized.permute(0, 2, 3, 1)
    return _pad_or_crop(imgs_resized, target_h, target_w)

def _resize_stretch(imgs, target_h, target_w):
    imgs_nchw = imgs.permute(0, 3, 1, 2)
    imgs_resized = F.interpolate(imgs_nchw, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return imgs_resized.permute(0, 2, 3, 1)

class ConcatImageBatches:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "ensure_same_size": ("BOOLEAN", {"default": True}),
                "resize_method": (["fit", "pad", "stretch"], {"default": "fit"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "concat"
    CATEGORY = "video/processing"

    def concat(self, images_a, images_b, ensure_same_size=True, resize_method="fit"):
        if images_a.shape[-1] != images_b.shape[-1]:
            raise ValueError("Channel count must match between A and B")
        images_b = images_b.to(device=images_a.device, dtype=images_a.dtype)

        _, Ha, Wa, _ = images_a.shape
        _, Hb, Wb, _ = images_b.shape
        if ensure_same_size and (Ha != Hb or Wa != Wb):
            if resize_method == "fit":
                images_b = _resize_fit_letterbox(images_b, Ha, Wa)
            elif resize_method == "pad":
                images_b = _pad_or_crop(images_b, Ha, Wa)
            elif resize_method == "stretch":
                images_b = _resize_stretch(images_b, Ha, Wa)

        out = torch.cat([images_a, images_b], dim=0).clamp(0.0, 1.0)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "ConcatImageBatches": ConcatImageBatches
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatImageBatches": "Concat Image Batches"
}
