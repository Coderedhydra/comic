from typing import Optional


def enhance_with_realesrgan(input_path: str, output_path: str, scale: int = 2, tile: int = 0) -> bool:
    """Try Real-ESRGAN super-resolution. Returns True on success, False to fallback."""
    try:
        # Lazy import to avoid heavy deps on cold start
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import cv2

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=None,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=not torch.cuda.is_available(),
        )
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return False
        output, _ = upsampler.enhance(img, outscale=scale)
        cv2.imwrite(output_path, output)
        return True
    except Exception:
        return False

