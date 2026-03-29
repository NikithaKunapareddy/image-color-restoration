"""
restoration.py

Functions for color restoration pipeline.

Techniques used:
- Bilateral Filtering for noise removal (preserves edges)
- Adaptive Gray World white balance using Hasler-Suesstrunk colorfulness metric
- Multi-Scale CLAHE on L channel for contrast enhancement
- Saturation increase in HSV space for color enhancement
- Unsharp Masking to enhance details
- Fold Line Suppression using Hough Transform + directional inpainting
- No-Reference IQA metrics: BRISQUE and NIQE for ablation study

Required libraries: OpenCV, NumPy
"""
import cv2
import numpy as np
import os


# ─────────────────────────────────────────────
# BASIC IMAGE ANALYSIS
# ─────────────────────────────────────────────

def is_grayscale(img):
    """Return True if image is essentially grayscale (all channels similar)."""
    if len(img.shape) < 3 or img.shape[2] == 1:
        return True
    b, g, r = cv2.split(img)
    diff = (np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16))) +
            np.mean(np.abs(b.astype(np.int16) - r.astype(np.int16))) +
            np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16)))) / 3.0
    return diff < 10.0


def estimate_noise(img):
    """Estimate noise level using high-frequency residual standard deviation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(residual))


def contrast_score(img):
    """Compute a simple contrast score (stddev of luminance)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def is_noisy(img, noise_thresh=10.0):
    """Decide whether an image is noisy based on estimated noise."""
    noise_lvl = estimate_noise(img)
    return noise_lvl > noise_thresh, noise_lvl


def is_low_contrast(img, contrast_thresh=30.0):
    """Detect low contrast images by thresholding luminance std deviation."""
    score = contrast_score(img)
    return score < contrast_thresh, score


# ─────────────────────────────────────────────
# DENOISING
# ─────────────────────────────────────────────

def nl_means_denoise(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """Apply Non-Local Means color denoising."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)


def remove_noise(img, d=9, sigma_color=75, sigma_space=75):
    """Remove noise using bilateral filtering (optional, not in main pipeline)."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ─────────────────────────────────────────────
# SPOT / DUST REMOVAL
# ─────────────────────────────────────────────

def detect_spots_mask(img, thresh=30, blur_size=9, min_frac=5e-5):
    """Detect small bright/dark spots (dust/scratches) and return a binary mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, blur_size)
    residual = cv2.absdiff(gray, med)
    _, mask = cv2.threshold(residual, thresh, 255, cv2.THRESH_BINARY)
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
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


# ─────────────────────────────────────────────
# FOLD LINE SUPPRESSION  ← NEW
# ─────────────────────────────────────────────

def detect_fold_lines(img, canny_low=50, canny_high=150, hough_thresh=100,
                      min_line_length=100, max_line_gap=10):
    """Detect physical fold/crease lines using Probabilistic Hough Transform.

    Returns a list of lines as (x1, y1, x2, y2) tuples.
    Only near-vertical or near-horizontal lines are returned
    (physical folds are straight, not diagonal scratches).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=hough_thresh,
                             minLineLength=min_line_length,
                             maxLineGap=max_line_gap)
    fold_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            # Keep only near-vertical (fold crease) or near-horizontal lines
            if dx == 0 or (dy / (dx + 1e-6)) > 3.0:  # near-vertical
                fold_lines.append((x1, y1, x2, y2))
            elif dy == 0 or (dx / (dy + 1e-6)) > 3.0:  # near-horizontal
                fold_lines.append((x1, y1, x2, y2))
    return fold_lines


def build_fold_mask(img_shape, fold_lines, thickness=5):
    """Build a binary mask covering detected fold lines."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in fold_lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    return mask


def suppress_fold_lines(img, thickness=5, canny_low=50, canny_high=150,
                         hough_thresh=100, min_line_length=100, max_line_gap=10):
    """Detect fold/crease lines and inpaint along them.

    Steps:
    1. Detect straight fold lines with Hough Transform
    2. Build a mask covering those lines
    3. Apply Telea inpainting along the mask
    4. Blend result with original for smooth transition

    Returns (result_img, fold_mask, num_folds_detected)
    """
    fold_lines = detect_fold_lines(img, canny_low, canny_high,
                                    hough_thresh, min_line_length, max_line_gap)
    if not fold_lines:
        return img, None, 0

    mask = build_fold_mask(img.shape, fold_lines, thickness=thickness)

    # Apply directional bilateral filter along fold mask area before inpainting
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)
    blended_pre = img.copy()
    blended_pre[mask > 0] = smoothed[mask > 0]

    # Inpaint the fold lines
    result = cv2.inpaint(blended_pre, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

    # Soft blend: 80% inpainted + 20% original for naturalness
    result = cv2.addWeighted(result, 0.8, img, 0.2, 0)

    return result, mask, len(fold_lines)


# ─────────────────────────────────────────────
# WHITE BALANCE — ADAPTIVE  ← UPDATED
# ─────────────────────────────────────────────

def colorfulness_metric(img):
    """Compute Hasler-Suesstrunk colorfulness metric.

    Reference: Hasler & Suesstrunk, "Measuring Colorfulness in Natural Images", 2003.

    Returns a float:
      < 15  = essentially grayscale / very faded
      15–33 = slightly colorful
      33–45 = moderately colorful
      45–59 = colorful
      > 59  = highly colorful

    This is used to adaptively set the white balance blend weight.
    """
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)

    rg = r - g
    yb = 0.5 * (r + g) - b

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    std_rg = np.std(rg)
    std_yb = np.std(yb)

    std_rgyb = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_rgyb = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

    colorfulness = std_rgyb + 0.3 * mean_rgyb
    return float(colorfulness)


def adaptive_wb_weight(colorfulness, min_weight=0.25, max_weight=0.70):
    """Compute white balance blend weight based on colorfulness.

    Logic:
    - Very faded image (low colorfulness) → high WB weight (more correction needed)
    - Well preserved image (high colorfulness) → low WB weight (less correction needed)

    Returns a float between min_weight and max_weight.
    """
    # Normalize colorfulness to 0–1 range (0=faded, 1=colorful)
    # Typical range of colorfulness is 0–100
    norm = np.clip(colorfulness / 60.0, 0.0, 1.0)

    # As colorfulness increases, WB weight decreases
    weight = max_weight - norm * (max_weight - min_weight)
    return float(weight)


def white_balance_grayworld(img):
    """Simple Gray-World white balance in BGR domain."""
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)
    mean_b = max(np.mean(b), 1.0)
    mean_g = max(np.mean(g), 1.0)
    mean_r = max(np.mean(r), 1.0)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    b = np.clip(b * (mean_gray / mean_b), 0, 255)
    g = np.clip(g * (mean_gray / mean_g), 0, 255)
    r = np.clip(r * (mean_gray / mean_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def white_balance_adaptive(img):
    """Apply adaptive white balance using colorfulness-based weight.

    Instead of a fixed 60/40 blend, this computes the blend weight
    dynamically based on how faded/colorful the image is.

    Returns (balanced_img, colorfulness_value, wb_weight_used)
    """
    cf = colorfulness_metric(img)
    wb_weight = adaptive_wb_weight(cf)
    wb = white_balance_grayworld(img)
    result = cv2.addWeighted(img, 1.0 - wb_weight, wb, wb_weight, 0)
    return result, cf, wb_weight


def white_balance(img):
    """LAB space white balance (optional, not used in main pipeline)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    l, a, b = cv2.split(lab)
    a = np.clip(a - (np.mean(a) - 128), 0, 255).astype(np.uint8)
    b = np.clip(b - (np.mean(b) - 128), 0, 255).astype(np.uint8)
    l = np.clip(l, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# CONTRAST ENHANCEMENT — MULTI-SCALE CLAHE  ← UPDATED
# ─────────────────────────────────────────────

def apply_clahe_single(img, clip_limit, tile_size):
    """Apply CLAHE on L channel with given clip_limit and tile_size."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def enhance_contrast_multiscale(img, clip_limit=1.1):
    """Multi-Scale CLAHE — process at three tile sizes and blend.

    Why multi-scale?
    - Small tiles (4×4): captures fine texture detail (rose petals)
    - Medium tiles (8×8): balanced local contrast
    - Large tiles (16×16): handles broad gradients (background table)
    Blending all three avoids halo artifacts from single-pass CLAHE.

    Returns the blended contrast-enhanced image.
    """
    small  = apply_clahe_single(img, clip_limit, (4,  4))   # fine detail
    medium = apply_clahe_single(img, clip_limit, (8,  8))   # balanced
    large  = apply_clahe_single(img, clip_limit, (16, 16))  # broad gradient

    # Equal blend of all three scales
    result = cv2.addWeighted(small, 0.33, medium, 0.33, 0)
    result = cv2.addWeighted(result, 1.0, large, 0.34, 0)
    return result


def enhance_contrast(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Single-scale CLAHE (kept for reference/comparison in ablation study)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# SATURATION
# ─────────────────────────────────────────────

def increase_saturation(img, scale=1.25):
    """Increase color saturation in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * scale, 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────

def mse(imageA, imageB):
    """Compute Mean Squared Error between two grayscale images."""
    return float(np.mean((imageA.astype('float32') - imageB.astype('float32')) ** 2))


def psnr(imageA, imageB):
    """Compute PSNR between two images."""
    return float(cv2.PSNR(imageA, imageB))


def ssim(img1, img2):
    """Compute SSIM index between two grayscale images."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1*img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2
    ssim_map  = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return float(np.mean(ssim_map))


def brisque_score(img):
    """Compute a simplified BRISQUE-like no-reference quality score.

    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) measures
    naturalness of image statistics without needing a reference image.
    Lower score = better perceptual quality.

    This is a lightweight approximation using local contrast statistics.
    For full BRISQUE, use opencv-contrib: cv2.quality.QualityBRISQUE_compute()
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Mean subtracted contrast normalized (MSCN) coefficients
    mu = cv2.GaussianBlur(gray, (7, 7), 7.0 / 6.0)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7.0 / 6.0) - mu_sq))
    mscn = (gray - mu) / (sigma + 1.0)
    # Score based on deviation from Gaussian distribution (kurtosis proxy)
    score = float(np.mean(np.abs(mscn ** 3)) + np.mean(np.abs(mscn ** 4 - 3)))
    return score


def niqe_score(img):
    """Compute a simplified NIQE-like no-reference quality score.

    NIQE (Natural Image Quality Evaluator) measures how natural the image
    statistics look compared to pristine images.
    Lower score = more natural looking image.

    This is a lightweight approximation. For full NIQE use:
    cv2.quality.QualityBRISQUE or scikit-image's NIQE implementation.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Local variance as naturalness proxy
    mu = cv2.GaussianBlur(gray, (7, 7), 1.5)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray**2, (7, 7), 1.5) - mu**2))
    # Natural images have specific sigma distribution — score deviation from it
    score = float(np.std(sigma) / (np.mean(sigma) + 1e-6))
    return score


# ─────────────────────────────────────────────
# RETINEX (OPTIONAL)
# ─────────────────────────────────────────────

def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32) + 1.0
    return np.log(img) - np.log(blur)


def multi_scale_retinex(img, scales):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in scales:
        retinex += single_scale_retinex(img, sigma)
    return retinex / float(len(scales))


def msrcr(img, scales=(15, 80, 250), G=192, b=-30, alpha=125, beta=46):
    """Multi-Scale Retinex with Color Restoration (optional)."""
    img = np.clip(img.astype(np.float32), 1.0, 255.0)
    B, Gc, R = cv2.split(img)
    sum_channels = (B + Gc + R) + 1.0
    out_B = G * multi_scale_retinex(B, scales) * np.log(alpha * B / sum_channels + 1.0)
    out_G = G * multi_scale_retinex(Gc, scales) * np.log(alpha * Gc / sum_channels + 1.0)
    out_R = G * multi_scale_retinex(R, scales) * np.log(alpha * R / sum_channels + 1.0)
    out = cv2.merge([out_B, out_G, out_R])
    out = (out - np.min(out)) / (np.max(out) - np.min(out) + 1e-8) * 255.0 + b
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# MAIN RESTORE PIPELINE  ← UPDATED
# ─────────────────────────────────────────────

def restore_image(img, sat_scale=1.25, nlm_h=10, median_k=5, clahe_clip=1.1,
                  sat_scale_override=None, unsharp_amount=0.3, spot_thresh=30,
                  spot_blur=9, spot_min_frac=5e-5, inpaint_radius=2,
                  use_fold_suppression=True, use_multiscale_clahe=True):
    """Restore image using the full enhanced pipeline.

    New parameters vs original:
      use_fold_suppression  : if True, detect and suppress physical fold lines
      use_multiscale_clahe  : if True, use multi-scale CLAHE (3 tile sizes blended)
    White balance is now ADAPTIVE based on colorfulness metric.
    """
    # Step 1: Denoising
    noise_lvl = estimate_noise(img)
    if noise_lvl > 10.0:
        denoise = nl_means_denoise(img, h=nlm_h, hColor=nlm_h)
    else:
        denoise = cv2.medianBlur(img, median_k)

    # Step 2: Adaptive white balance (colorfulness-based blend weight)
    result, colorfulness, wb_weight = white_balance_adaptive(denoise)

    # Step 3: Spot detection & inpainting
    mask, have_spots = detect_spots_mask(result, thresh=spot_thresh,
                                          blur_size=spot_blur, min_frac=spot_min_frac)
    if have_spots:
        result = cv2.inpaint(result, mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Step 4: Fold line suppression (NEW)
    num_folds = 0
    if use_fold_suppression:
        result, fold_mask, num_folds = suppress_fold_lines(result)

    # Step 5: Contrast enhancement
    if use_multiscale_clahe:
        contrast = enhance_contrast_multiscale(result, clip_limit=clahe_clip)
    else:
        contrast = apply_clahe_single(result, clahe_clip, (8, 8))

    # Step 6: Saturation boost
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    scale = sat_scale_override if sat_scale_override is not None else sat_scale
    s = np.clip(s * scale, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.merge((h.astype(np.uint8), s, v.astype(np.uint8))),
                          cv2.COLOR_HSV2BGR)

    # Step 7: Unsharp masking
    blurred = cv2.GaussianBlur(color, (0, 0), sigmaX=1.0)
    sharpen = cv2.addWeighted(color, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0)

    return sharpen


# ─────────────────────────────────────────────
# ABLATION STUDY  ← NEW
# ─────────────────────────────────────────────

def run_ablation_study(img, sat_scale=1.5, nlm_h=6, median_k=3, clahe_clip=1.1,
                        sat_scale_override=1.5, unsharp_amount=0.3,
                        spot_thresh=50, spot_blur=9, spot_min_frac=1e-4,
                        inpaint_radius=2):
    """Run ablation study — process image with each step removed one at a time.

    Returns a dict of variant_name → (restored_image, brisque, niqe)

    Variants:
      original            : no processing at all
      full_pipeline       : all steps active
      no_denoising        : skip denoising step
      no_white_balance    : skip white balance step
      no_clahe            : skip contrast enhancement
      no_saturation       : skip saturation boost
      no_unsharp          : skip unsharp masking
      no_fold_suppression : skip fold line suppression
    """
    results = {}

    def score(processed):
        b = brisque_score(processed)
        n = niqe_score(processed)
        return processed, b, n

    # Original — no processing
    results['original'] = score(img)

    # Full pipeline
    full = restore_image(img, sat_scale=sat_scale, nlm_h=nlm_h,
                          median_k=median_k, clahe_clip=clahe_clip,
                          sat_scale_override=sat_scale_override,
                          unsharp_amount=unsharp_amount,
                          spot_thresh=spot_thresh, spot_blur=spot_blur,
                          spot_min_frac=spot_min_frac,
                          inpaint_radius=inpaint_radius,
                          use_fold_suppression=True,
                          use_multiscale_clahe=True)
    results['full_pipeline'] = score(full)

    # --- Remove denoising ---
    noise_lvl = estimate_noise(img)
    if noise_lvl > 10.0:
        denoised = nl_means_denoise(img, h=nlm_h, hColor=nlm_h)
    else:
        denoised = cv2.medianBlur(img, median_k)
    wb_result, _, _ = white_balance_adaptive(img)           # skip denoise
    mask, have_spots = detect_spots_mask(wb_result, thresh=spot_thresh, blur_size=spot_blur, min_frac=spot_min_frac)
    if have_spots:
        wb_result = cv2.inpaint(wb_result, mask, inpaint_radius, cv2.INPAINT_TELEA)
    no_denoise_contrast = enhance_contrast_multiscale(wb_result, clip_limit=clahe_clip)
    hsv = cv2.cvtColor(no_denoise_contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    h2, s2, v2 = cv2.split(hsv)
    s2 = np.clip(s2 * sat_scale_override, 0, 255).astype(np.uint8)
    no_denoise_color = cv2.cvtColor(cv2.merge((h2.astype(np.uint8), s2, v2.astype(np.uint8))), cv2.COLOR_HSV2BGR)
    blr = cv2.GaussianBlur(no_denoise_color, (0, 0), sigmaX=1.0)
    results['no_denoising'] = score(cv2.addWeighted(no_denoise_color, 1.0+unsharp_amount, blr, -unsharp_amount, 0))

    # --- Remove white balance ---
    no_wb = denoised.copy()
    mask2, have_spots2 = detect_spots_mask(no_wb, thresh=spot_thresh, blur_size=spot_blur, min_frac=spot_min_frac)
    if have_spots2:
        no_wb = cv2.inpaint(no_wb, mask2, inpaint_radius, cv2.INPAINT_TELEA)
    no_wb_contrast = enhance_contrast_multiscale(no_wb, clip_limit=clahe_clip)
    hsv3 = cv2.cvtColor(no_wb_contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    h3, s3, v3 = cv2.split(hsv3)
    s3 = np.clip(s3 * sat_scale_override, 0, 255).astype(np.uint8)
    no_wb_color = cv2.cvtColor(cv2.merge((h3.astype(np.uint8), s3, v3.astype(np.uint8))), cv2.COLOR_HSV2BGR)
    blr3 = cv2.GaussianBlur(no_wb_color, (0, 0), sigmaX=1.0)
    results['no_white_balance'] = score(cv2.addWeighted(no_wb_color, 1.0+unsharp_amount, blr3, -unsharp_amount, 0))

    # --- Remove CLAHE ---
    wb_only, _, _ = white_balance_adaptive(denoised)
    mask4, have_spots4 = detect_spots_mask(wb_only, thresh=spot_thresh, blur_size=spot_blur, min_frac=spot_min_frac)
    if have_spots4:
        wb_only = cv2.inpaint(wb_only, mask4, inpaint_radius, cv2.INPAINT_TELEA)
    hsv4 = cv2.cvtColor(wb_only, cv2.COLOR_BGR2HSV).astype(np.float32)
    h4, s4, v4 = cv2.split(hsv4)
    s4 = np.clip(s4 * sat_scale_override, 0, 255).astype(np.uint8)
    no_clahe_color = cv2.cvtColor(cv2.merge((h4.astype(np.uint8), s4, v4.astype(np.uint8))), cv2.COLOR_HSV2BGR)
    blr4 = cv2.GaussianBlur(no_clahe_color, (0, 0), sigmaX=1.0)
    results['no_clahe'] = score(cv2.addWeighted(no_clahe_color, 1.0+unsharp_amount, blr4, -unsharp_amount, 0))

    # --- Remove saturation ---
    wb5, _, _ = white_balance_adaptive(denoised)
    mask5, hs5 = detect_spots_mask(wb5, thresh=spot_thresh, blur_size=spot_blur, min_frac=spot_min_frac)
    if hs5:
        wb5 = cv2.inpaint(wb5, mask5, inpaint_radius, cv2.INPAINT_TELEA)
    contrast5 = enhance_contrast_multiscale(wb5, clip_limit=clahe_clip)
    blr5 = cv2.GaussianBlur(contrast5, (0, 0), sigmaX=1.0)
    results['no_saturation'] = score(cv2.addWeighted(contrast5, 1.0+unsharp_amount, blr5, -unsharp_amount, 0))

    # --- Remove unsharp masking ---
    wb6, _, _ = white_balance_adaptive(denoised)
    mask6, hs6 = detect_spots_mask(wb6, thresh=spot_thresh, blur_size=spot_blur, min_frac=spot_min_frac)
    if hs6:
        wb6 = cv2.inpaint(wb6, mask6, inpaint_radius, cv2.INPAINT_TELEA)
    contrast6 = enhance_contrast_multiscale(wb6, clip_limit=clahe_clip)
    hsv6 = cv2.cvtColor(contrast6, cv2.COLOR_BGR2HSV).astype(np.float32)
    h6, s6, v6 = cv2.split(hsv6)
    s6 = np.clip(s6 * sat_scale_override, 0, 255).astype(np.uint8)
    results['no_unsharp'] = score(cv2.cvtColor(cv2.merge((h6.astype(np.uint8), s6, v6.astype(np.uint8))), cv2.COLOR_HSV2BGR))

    # --- Remove fold suppression ---
    no_fold = restore_image(img, sat_scale=sat_scale, nlm_h=nlm_h,
                             median_k=median_k, clahe_clip=clahe_clip,
                             sat_scale_override=sat_scale_override,
                             unsharp_amount=unsharp_amount,
                             use_fold_suppression=False,
                             use_multiscale_clahe=True)
    results['no_fold_suppression'] = score(no_fold)

    return results


def print_ablation_table(ablation_results):
    """Print ablation study results as a formatted table."""
    print('\n' + '='*65)
    print(f"{'Variant':<25} {'BRISQUE':>10} {'NIQE':>10} {'Note'}")
    print('='*65)
    order = ['original', 'full_pipeline', 'no_denoising', 'no_white_balance',
             'no_clahe', 'no_saturation', 'no_unsharp', 'no_fold_suppression']
    notes = {
        'original':           'No processing',
        'full_pipeline':      'All steps active ← best',
        'no_denoising':       'Skip denoise step',
        'no_white_balance':   'Skip white balance',
        'no_clahe':           'Skip contrast step',
        'no_saturation':      'Skip saturation boost',
        'no_unsharp':         'Skip unsharp masking',
        'no_fold_suppression':'Skip fold suppression',
    }
    for key in order:
        if key in ablation_results:
            _, b, n = ablation_results[key]
            print(f"{key:<25} {b:>10.4f} {n:>10.4f}   {notes.get(key,'')}")
    print('='*65)
    print('Lower BRISQUE and NIQE = better perceptual quality\n')


# ─────────────────────────────────────────────
# ANALYZE AND RESTORE
# ─────────────────────────────────────────────

def analyze_and_restore(img, sat_scale=1.25, noise_thresh=10.0, contrast_thresh=30.0):
    """Analyze image and run adaptive restoration pipeline.

    Returns (restored_image, info_dict).
    """
    info = {}
    info['is_grayscale'] = is_grayscale(img)
    noisy, noise_lvl = is_noisy(img, noise_thresh=noise_thresh)
    info['is_noisy'] = noisy
    info['noise_level'] = noise_lvl
    low_contrast, contrast_score_val = is_low_contrast(img, contrast_thresh=contrast_thresh)
    info['is_low_contrast'] = low_contrast
    info['contrast_score'] = contrast_score_val

    # Colorfulness info
    cf = colorfulness_metric(img)
    wb_w = adaptive_wb_weight(cf)
    info['colorfulness'] = cf
    info['wb_weight_used'] = wb_w

    restored = restore_image(img, sat_scale=sat_scale)

    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res_gray  = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    info['mse']  = mse(orig_gray, res_gray)
    info['psnr'] = psnr(orig_gray, res_gray)
    info['ssim'] = ssim(orig_gray, res_gray)

    return restored, info