"""
restoration.py

Functions for color restoration pipeline.

Techniques used:
- Bilateral Filtering for noise removal (preserves edges)
- Gray World white balance in LAB color space
- CLAHE on L channel for contrast enhancement
- Saturation increase in HSV space for color enhancement
- Sharpening via a simple kernel to enhance details

Required libraries: OpenCV, NumPy
"""
import cv2
import numpy as np
import os


def is_grayscale(img):
    """Return True if image is essentially grayscale (all channels similar)."""
    if len(img.shape) < 3 or img.shape[2] == 1:
        return True
    b, g, r = cv2.split(img)
    # compute mean absolute difference between channels
    diff = (np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16))) +
            np.mean(np.abs(b.astype(np.int16) - r.astype(np.int16))) +
            np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16)))) / 3.0
    return diff < 10.0


def estimate_noise(img):
    """Estimate noise level using high-frequency residual standard deviation.

    Convert to grayscale, blur and subtract to get high-frequency components.
    Return the standard deviation as a noise estimate.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(residual))


def nl_means_denoise(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """Apply Non-Local Means color denoising."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)


def detect_spots_mask(img, thresh=30, blur_size=9, min_frac=5e-5):
    """Detect small bright/dark spots (dust/scratches) and return a binary mask.

    Returns mask and a boolean indicating whether there are enough spots to inpaint.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, blur_size)
    residual = cv2.absdiff(gray, med)
    _, mask = cv2.threshold(residual, thresh, 255, cv2.THRESH_BINARY)
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    count = int(np.sum(mask > 0))
    have_spots = count > (img.shape[0] * img.shape[1] * min_frac)
    return mask.astype(np.uint8), have_spots


def inpaint_spots(img, mask):
    """Inpaint detected spots using Telea method."""
    if mask is None or np.sum(mask) == 0:
        return img
    # default radius 3 retained if not provided
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted


def is_noisy(img, noise_thresh=10.0):
    """Decide whether an image is noisy based on estimated noise.

    `noise_thresh` is a tunable threshold; typical values 8-12.
    """
    noise_lvl = estimate_noise(img)
    return noise_lvl > noise_thresh, noise_lvl


def contrast_score(img):
    """Compute a simple contrast score (stddev of luminance)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def is_low_contrast(img, contrast_thresh=30.0):
    """Detect low contrast images by thresholding luminance std deviation."""
    score = contrast_score(img)
    return score < contrast_thresh, score


def mse(imageA, imageB):
    """Compute Mean Squared Error between two images (grayscale)."""
    err = np.mean((imageA.astype('float32') - imageB.astype('float32')) ** 2)
    return float(err)


def psnr(imageA, imageB):
    """Compute PSNR between two images using OpenCV helper."""
    return float(cv2.PSNR(imageA, imageB))


def ssim(img1, img2):
    """Compute a simple single-channel SSIM index between two images.

    Implements the luminance-contrast-structure SSIM for grayscale images.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def single_scale_retinex(img, sigma):
    """Single Scale Retinex for a single-channel image."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    # avoid log(0)
    img = img.astype(np.float32) + 1.0
    blur = blur.astype(np.float32) + 1.0
    retinex = np.log(img) - np.log(blur)
    return retinex


def multi_scale_retinex(img, scales):
    """Apply Multi-Scale Retinex to a single-channel image.

    scales: list of sigma values
    """
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in scales:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / float(len(scales))
    return retinex


def msrcr(img, scales=(15, 80, 250), G=192, b=-30, alpha=125, beta=46):
    """Multi-Scale Retinex with Color Restoration (MSRCR).

    Returns BGR uint8 image.
    Parameters chosen to be reasonable defaults; they can be tuned.
    """
    img = img.astype(np.float32)
    img = np.clip(img, 1.0, 255.0)

    # split channels in BGR but process on each channel
    B, Gc, R = cv2.split(img)
    # compute MSR on each channel
    msr_B = multi_scale_retinex(B, scales)
    msr_G = multi_scale_retinex(Gc, scales)
    msr_R = multi_scale_retinex(R, scales)

    # color restoration factor
    # avoid division by zero: sum of channels
    sum_channels = (B + Gc + R) + 1.0
    crf_B = np.log(alpha * B / sum_channels + 1.0)
    crf_G = np.log(alpha * Gc / sum_channels + 1.0)
    crf_R = np.log(alpha * R / sum_channels + 1.0)

    # apply MSRCR
    out_B = G * msr_B * crf_B
    out_G = G * msr_G * crf_G
    out_R = G * msr_R * crf_R

    out = cv2.merge([out_B, out_G, out_R])

    # linear scaling to 0-255 with gain and offset
    out = (out - np.min(out)) / (np.max(out) - np.min(out) + 1e-8) * 255.0
    out = out + b
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out



def remove_noise(img, d=9, sigma_color=75, sigma_space=75):
    """Remove noise while preserving edges using bilateral filtering.

    Bilateral filter smooths regions while keeping edges sharp by
    combining spatial and intensity information.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def white_balance(img):
    """Apply white balance using a Gray World assumption in LAB space.

    Convert to LAB and shift the a/b channels so their means equal the
    neutral midpoint (128). This reduces yellow/red tints common in
    old photos.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    l, a, b = cv2.split(lab)

    # Compute channel means
    mean_a = np.mean(a)
    mean_b = np.mean(b)

    # Shift a and b so their means move to 128 (neutral)
    a = a - (mean_a - 128)
    b = b - (mean_b - 128)

    # Clip and convert back
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    l = np.clip(l, 0, 255).astype(np.uint8)

    balanced = cv2.merge([l, a, b])
    balanced = cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)
    return balanced


def white_balance_grayworld(img):
    """Simple Gray-World white balance applied in BGR domain.

    Scales each BGR channel so their means match the overall mean luminance.
    This is robust for faded color casts like yellowing.
    """
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)
    # average per-channel
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    # overall mean
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    # avoid division by zero
    mean_b = max(mean_b, 1.0)
    mean_g = max(mean_g, 1.0)
    mean_r = max(mean_r, 1.0)

    b = np.clip(b * (mean_gray / mean_b), 0, 255)
    g = np.clip(g * (mean_gray / mean_g), 0, 255)
    r = np.clip(r * (mean_gray / mean_r), 0, 255)

    balanced = cv2.merge([b, g, r]).astype(np.uint8)
    return balanced


def enhance_contrast(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Enhance contrast using CLAHE on the L channel in LAB space.

    CLAHE (Contrast Limited AHE) improves local contrast while avoiding
    over-amplification of noise.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced


def increase_saturation(img, scale=1.25):
    """Increase color saturation in HSV space to restore faded colors.

    Multiply the S channel by `scale` and clip to valid range.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    s = np.clip(s * scale, 0, 255)

    hsv_sat = cv2.merge([h, s, v]).astype(np.uint8)
    saturated = cv2.cvtColor(hsv_sat, cv2.COLOR_HSV2BGR)
    return saturated


def sharpen_image(img):
    """Sharpen image using a simple kernel (unsharp-ish).

    This kernel emphasizes center pixel and subtracts neighbors.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp


def restore_image(img, sat_scale=1.25, nlm_h=10, median_k=5, clahe_clip=2.0,
                  sat_scale_override=None, unsharp_amount=0.5, spot_thresh=30,
                  spot_blur=9, spot_min_frac=5e-5, inpaint_radius=3):
    """Restore image with tunable parameters for classical DIP restoration.

    Parameters are tunable to create presets for mild/balanced/aggressive.
    """
    # Step 1: Choose denoising method adaptively based on estimated noise
    noise_lvl = estimate_noise(img)
    if noise_lvl > 10.0:
        denoise = nl_means_denoise(img, h=nlm_h, hColor=nlm_h)
    else:
        denoise = cv2.medianBlur(img, median_k)

    # Step 2: White balance using Gray-World
    result = white_balance_grayworld(denoise)

    # Spot detection & inpainting (remove dust/specks) if present
    mask, have_spots = detect_spots_mask(result, thresh=spot_thresh, blur_size=spot_blur, min_frac=spot_min_frac)
    if have_spots:
        # perform inpainting with provided radius to remove detected specks
        result = cv2.inpaint(result, mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Step 3: Contrast enhancement using CLAHE on L channel (or MSRCR)
    # By default use CLAHE; MSRCR will be applied if msr=True passed via kwargs.
    lab2 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l2, a2, b2 = cv2.split(lab2)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    cl = clahe.apply(l2)
    merged = cv2.merge((cl, a2, b2))
    contrast = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Step 4: Increase saturation in HSV
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    scale = sat_scale_override if sat_scale_override is not None else sat_scale
    s = np.clip(s * scale, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h.astype(np.uint8), s, v.astype(np.uint8)))
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 5: Mild sharpening via unsharp masking
    blurred = cv2.GaussianBlur(color, (0, 0), sigmaX=1.0)
    sharpen = cv2.addWeighted(color, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0)

    return sharpen


def restore_image_msr_wrapper(img, **kwargs):
    """Wrapper: if msr True in kwargs, apply MSRCR instead of CLAHE step.

    This function calls restore_image but replaces the contrast step output
    with MSRCR if requested.
    """
    msr_flag = kwargs.pop('msr', False)
    # run the standard pipeline first
    restored = restore_image(img, **kwargs)
    if not msr_flag:
        return restored

    # If msr_flag True: recompute a base image from denoising + inpainting
    noise_lvl = estimate_noise(img)
    nlm_h = kwargs.get('nlm_h', 10)
    median_k = kwargs.get('median_k', 5)
    if noise_lvl > 10.0:
        base = nl_means_denoise(img, h=nlm_h, hColor=nlm_h)
    else:
        base = cv2.medianBlur(img, median_k)
    base = white_balance_grayworld(base)
    mask, have_spots = detect_spots_mask(base, thresh=kwargs.get('spot_thresh', 30), blur_size=kwargs.get('spot_blur', 9), min_frac=kwargs.get('spot_min_frac', 5e-5))
    if have_spots:
        base = cv2.inpaint(base, mask, kwargs.get('inpaint_radius', 3), cv2.INPAINT_TELEA)

    # Apply MSRCR
    msr_img = msrcr(base)

    # After MSRCR, increase saturation and sharpen as in main pipeline
    hsv = cv2.cvtColor(msr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    scale = kwargs.get('sat_scale_override', None) or kwargs.get('sat_scale', 1.25)
    s = np.clip(s * scale, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h.astype(np.uint8), s, v.astype(np.uint8)))
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blurred = cv2.GaussianBlur(color, (0, 0), sigmaX=1.0)
    unsharp_amount = kwargs.get('unsharp_amount', 0.5)
    final = cv2.addWeighted(color, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0)
    return final


def analyze_and_restore(img, sat_scale=1.25, noise_thresh=10.0, contrast_thresh=30.0):
    """Analyze image and run an adaptive restoration pipeline.

    Returns (restored_image, info_dict) where info_dict contains detection
    results and quality metrics (computed after restoration).
    """
    info = {}

    # detect grayscale
    info['is_grayscale'] = is_grayscale(img)

    # noise detection
    noisy, noise_lvl = is_noisy(img, noise_thresh=noise_thresh)
    info['is_noisy'] = noisy
    info['noise_level'] = noise_lvl

    # contrast detection
    low_contrast, contrast_score_val = is_low_contrast(img, contrast_thresh=contrast_thresh)
    info['is_low_contrast'] = low_contrast
    info['contrast_score'] = contrast_score_val

    # Choose pipeline
    # Use the restored image from the main restore pipeline
    restored = restore_image(img, sat_scale=sat_scale)

    # Metrics (use grayscale comparison)
    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    info['mse'] = mse(orig_gray, res_gray)
    info['psnr'] = psnr(orig_gray, res_gray)
    info['ssim'] = ssim(orig_gray, res_gray)

    return restored, info
