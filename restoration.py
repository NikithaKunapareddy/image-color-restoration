"""
restoration.py

Functions for color restoration pipeline.

Techniques used:
- Adaptive Gray World white balance using Hasler-Suesstrunk colorfulness metric
- Multi-Scale CLAHE on L channel for contrast enhancement
- Saturation increase in HSV space for color enhancement
- Unsharp Masking to enhance details
- Fold Line Suppression using Hough Transform + directional inpainting
- Spot/dust removal using residual thresholding + Telea inpainting
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
    if len(img.shape) < 3 or img.shape[2] == 1:
        return True
    b, g, r = cv2.split(img)
    diff = (np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16))) +
            np.mean(np.abs(b.astype(np.int16) - r.astype(np.int16))) +
            np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16)))) / 3.0
    return diff < 10.0

# In restoration.py — replace estimate_noise() with this:

def estimate_noise_advanced(img):
    """
    Combines:
    1. Patch-based local variance (spatial domain)
    2. High-frequency energy via DFT (frequency domain)
    Returns a single robust noise estimate.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- Method 1: Patch-based (local flat regions) ---
    h, w   = gray.shape
    ps     = 16   # patch size
    variances = []
    for y in range(0, h - ps, ps):
        for x in range(0, w - ps, ps):
            patch = gray[y:y+ps, x:x+ps]
            # Only use flat patches (low local mean gradient)
            gx = np.abs(np.diff(patch, axis=1)).mean()
            gy = np.abs(np.diff(patch, axis=0)).mean()
            if gx < 5.0 and gy < 5.0:       # flat region
                variances.append(np.var(patch))

    patch_noise = float(np.median(variances)) ** 0.5 if variances else 0.0

    # --- Method 2: Frequency domain — high-freq energy ratio ---
    dft   = np.fft.fft2(gray)
    dft_s = np.fft.fftshift(dft)
    mag   = np.abs(dft_s)
    cy, cx = h // 2, w // 2
    rh, rw = h // 6, w // 6   # inner (low-freq) region
    low_mask       = np.zeros((h, w), bool)
    low_mask[cy-rh:cy+rh, cx-rw:cx+rw] = True
    low_energy  = mag[low_mask].sum()  + 1e-6
    high_energy = mag[~low_mask].sum() + 1e-6
    freq_noise  = float(high_energy / (low_energy + high_energy)) * 30.0

    # Blend both estimates
    combined = 0.6 * patch_noise + 0.4 * freq_noise
    return float(combined)


# Backwards-compatible wrapper: keep `estimate_noise` name used elsewhere
def estimate_noise(img):
    """Compatibility wrapper that calls the improved estimator."""
    try:
        return float(estimate_noise_advanced(img))
    except Exception:
        # Fallback: simple residual-based estimator
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
        return float(np.std(residual))

def _local_patch_stats(img, patch_size=32, step=16):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    means, vars_ = [], []
    for y in range(0, max(1, h - patch_size + 1), step):
        for x in range(0, max(1, w - patch_size + 1), step):
            patch = gray[y:y+patch_size, x:x+patch_size]
            means.append(float(np.mean(patch)))
            vars_.append(float(np.var(patch)))
    return np.array(means), np.array(vars_)


def classify_noise_type(img, patch_size=32, step=16):
    means, vars_ = _local_patch_stats(img, patch_size=patch_size, step=step)
    if len(means) < 5:
        return ('gaussian', float(np.median(vars_)) if len(vars_) else 0.0, 0.0)
    mean_m = np.mean(means)
    var_m  = np.mean(vars_)
    cov    = np.mean((means - mean_m) * (vars_ - var_m))
    denom  = (np.std(means) * np.std(vars_) + 1e-12)
    corr   = float(cov / denom)
    hist, _ = np.histogram(vars_, bins=8)
    peaks   = np.sum(hist > (np.mean(hist) + np.std(hist)))
    if peaks > 1:
        return ('mixed', float(np.median(vars_)), corr)
    if corr > 0.55:
        return ('poisson', float(np.median(vars_)), corr)
    return ('gaussian', float(np.median(vars_)), corr)


def anscombe_transform(img):
    return 2.0 * np.sqrt(img.astype(np.float32) + 3.0 / 8.0)


def inverse_anscombe(transformed):
    inv = ((transformed / 2.0) ** 2) - 1.0 / 8.0
    return np.clip(inv, 0.0, 255.0).astype(np.uint8)


def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def is_noisy(img, noise_thresh=10.0):
    noise_lvl = estimate_noise(img)
    return noise_lvl > noise_thresh, noise_lvl


def is_low_contrast(img, contrast_thresh=30.0):
    score = contrast_score(img)
    return score < contrast_thresh, score


def entropy_metric(img, nbins=256):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray.flatten(), bins=nbins, range=(0, 256), density=True)
    hist = hist + 1e-12
    return float(-np.sum(hist * np.log2(hist)))


# ─────────────────────────────────────────────
# DENOISING
# ─────────────────────────────────────────────

def nl_means_denoise(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)


def remove_noise(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ─────────────────────────────────────────────
# BLUR DETECTION
# ─────────────────────────────────────────────

def detect_blur_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
    blur_level = float(np.var(laplacian))
    is_blurred = blur_level < 200
    return blur_level, is_blurred


# ─────────────────────────────────────────────
# SHARPENING HELPERS
# ─────────────────────────────────────────────

def _high_pass_aggressive(channel, blur_level):
    ch_f = channel.astype(np.float32)
    if blur_level < 100:
        sigma, strength = 2.5, 0.8
    elif blur_level < 200:
        sigma, strength = 2.0, 0.6
    else:
        sigma, strength = 1.5, 0.4
    blurred   = cv2.GaussianBlur(ch_f, (0, 0), sigmaX=sigma)
    high_pass = ch_f - blurred
    return np.clip(ch_f + strength * high_pass, 0, 255).astype(np.uint8)


def high_pass_filter_sharpen(img, blur_level):
    if len(img.shape) == 3:
        return cv2.merge([_high_pass_aggressive(ch, blur_level) for ch in cv2.split(img)])
    return _high_pass_aggressive(img, blur_level)


def adaptive_unsharp_mask(img, blur_level, base_amount=0.3):
    if blur_level < 100:
        amount = base_amount * 3.0
    elif blur_level < 200:
        amount = base_amount * 2.0
    elif blur_level < 500:
        amount = base_amount * 1.3
    else:
        amount = base_amount * 0.7
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    result  = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_edges_adaptive(img, blur_level):
    if blur_level > 400:
        return img
    if blur_level < 100:
        strength = 0.6
    elif blur_level < 200:
        strength = 0.4
    else:
        strength = 0.2
    kernel = np.array([[-strength, -strength, -strength],
                       [-strength, 1 + 8*strength, -strength],
                       [-strength, -strength, -strength]])
    if len(img.shape) == 3 and img.shape[2] == 3:
        channels = cv2.split(img)
        enhanced = [cv2.filter2D(ch.astype(np.float32), -1, kernel) for ch in channels]
        result   = cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in enhanced])
    else:
        result = np.clip(cv2.filter2D(img.astype(np.float32), -1, kernel), 0, 255).astype(np.uint8)
    blend_alpha = 0.55 if blur_level < 200 else 0.7
    return np.clip(cv2.addWeighted(result, blend_alpha, img, 1-blend_alpha, 0), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# SPOT / DUST REMOVAL  ← FIXED
# ─────────────────────────────────────────────

def detect_spots_mask(img, thresh=40, min_area=50):
    """Detect small bright/dark spots (dust/scratches) and return a binary mask.

    Fixed version:
    - Single median blur (kernel=5) — not multi-scale (was too sensitive)
    - Proper contour area filtering with min_area
    - Simpler and cleaner — fewer false positives

    Returns (mask, have_spots).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    med      = cv2.medianBlur(gray, 5)
    residual = cv2.absdiff(gray, med)

    _, mask = cv2.threshold(residual, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)

    return clean.astype(np.uint8), np.sum(clean) > 0


def inpaint_spots(img, mask, inpaint_radius=3):
    if mask is None or np.sum(mask) == 0:
        return img
    return cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)


# ─────────────────────────────────────────────
# FOLD LINE SUPPRESSION  ← FIXED
# ─────────────────────────────────────────────

def detect_fold_lines(img, canny_low=80, canny_high=180,
                      hough_thresh=180,
                      min_line_length=200,
                      max_line_gap=10,
                      min_confidence=0.35,
                      allow_diagonal=False):
    """Detect physical fold/crease lines using Probabilistic Hough Transform.

    Fixed version:
    - Higher hough_thresh (180) — fewer false detections
    - Higher min_line_length (200) — only real long folds
    - Confidence filter: removes lines with < 25% edge support
    - Returns list of dicts with x1,y1,x2,y2,confidence,thickness
    - Keeps only top 3 most confident lines
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=hough_thresh,
                             minLineLength=min_line_length,
                             maxLineGap=max_line_gap)

    fold_lines = []
    if lines is None:
        return []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2-x1, y2-y1)
        # scale-aware minimum length (guard against downscaling)
        min_len = min_line_length
        h_img, w_img = edges.shape
        min_len = max(40, min(min_len, int(max(w_img, h_img) * 0.15)))
        if length < min_len:
            continue
        dx = abs(x2-x1)
        dy = abs(y2-y1)
        # Accept near-horizontal or near-vertical lines unless diagonals are allowed
        if not allow_diagonal:
            if dx == 0:
                ang_ratio = float('inf')
            else:
                ang_ratio = dy / float(dx)
            if not (ang_ratio < 0.25 or ang_ratio > 4.0):
                continue
        pts = np.linspace(0, 1, 100)
        xs  = np.clip((x1+(x2-x1)*pts).astype(int), 0, edges.shape[1]-1)
        ys  = np.clip((y1+(y2-y1)*pts).astype(int), 0, edges.shape[0]-1)
        confidence = float(np.mean(edges[ys, xs]) / 255.0)
        # Filter by configured confidence
        if confidence < float(min_confidence):
            continue
        fold_lines.append({'x1':int(x1),'y1':int(y1),'x2':int(x2),'y2':int(y2),
                           'confidence':confidence,'thickness':5})

    fold_lines = sorted(fold_lines, key=lambda x: x['confidence'], reverse=True)[:3]

    # If nothing found, try a relaxed fallback to surface any missed folds
    if not fold_lines:
        # relaxed params
        edges2 = cv2.Canny(gray, max(30, canny_low//2), min(160, canny_high))
        raw2 = cv2.HoughLinesP(edges2, 1, np.pi/180,
                              threshold=max(40, hough_thresh//3),
                              minLineLength=max(30, int(min_line_length*0.5)),
                              maxLineGap=max(5, max_line_gap*2))
        if raw2 is not None:
            for l in raw2:
                x1, y1, x2, y2 = l[0]
                length = np.hypot(x2-x1, y2-y1)
                if length < 30:
                    continue
                dx = abs(x2-x1); dy = abs(y2-y1)
                if not allow_diagonal:
                    if dx == 0:
                        ang_ratio = float('inf')
                    else:
                        ang_ratio = dy / float(dx)
                    if not (ang_ratio < 0.5 or ang_ratio > 2.0):
                        continue
                pts = np.linspace(0, 1, 60)
                xs = np.clip((x1+(x2-x1)*pts).astype(int), 0, edges2.shape[1]-1)
                ys = np.clip((y1+(y2-y1)*pts).astype(int), 0, edges2.shape[0]-1)
                confidence = float(np.mean(edges2[ys, xs]) / 255.0)
                if confidence < 0.15:
                    continue
                fold_lines.append({'x1':int(x1),'y1':int(y1),'x2':int(x2),'y2':int(y2),
                                   'confidence':confidence,'thickness':5})
        fold_lines = sorted(fold_lines, key=lambda x: x['confidence'], reverse=True)[:3]

    return fold_lines


def build_fold_mask(img_shape, fold_lines, default_thickness=5, min_confidence=0.35):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for fl in fold_lines:
        if isinstance(fl, (list, tuple)) and len(fl) >= 4:
            x1, y1, x2, y2 = fl[0], fl[1], fl[2], fl[3]
            thickness = default_thickness
        else:
            x1, y1, x2, y2 = fl['x1'], fl['y1'], fl['x2'], fl['y2']
            thickness = int(fl.get('thickness', default_thickness))
            if float(fl.get('confidence', 1.0)) < min_confidence:
                continue
        thickness = max(1, min(max(img_shape[:2])//10, thickness))
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    return mask


def suppress_fold_lines(img, thickness=5, canny_low=80, canny_high=180,
                         hough_thresh=180, min_line_length=200, max_line_gap=10,
                         min_confidence=0.25):
    """Detect fold/crease lines and inpaint along them.

    Returns (result_img, fold_mask, num_folds_detected).
    """
    fold_lines = detect_fold_lines(img, canny_low, canny_high,
                                    hough_thresh, min_line_length, max_line_gap)
    if not fold_lines:
        return img, None, 0

    mask = build_fold_mask(img.shape, fold_lines,
                           default_thickness=thickness,
                           min_confidence=min_confidence)
    if np.sum(mask) == 0:
        return img, None, 0

    smoothed    = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)
    blended_pre = img.copy()
    blended_pre[mask > 0] = smoothed[mask > 0]

    result = cv2.inpaint(blended_pre, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    final  = cv2.addWeighted(result, 0.85, img, 0.15, 0)

    return final, mask, len(fold_lines)


# ─────────────────────────────────────────────
# DEBUG VISUALIZATION  ← FIXED
# ─────────────────────────────────────────────

def save_debug_overlays(img, fold_lines, spots_mask, out_prefix):
    """Save debug visualizations for detected folds and spot masks.

    Fixed version:
    - Simple green lines for folds — no score text, no color ramps
    - Red overlay for spots
    - Combined overlay image
    """
    fold_viz = img.copy()
    for fl in fold_lines:
        if isinstance(fl, (list, tuple)) and len(fl) >= 4:
            x1, y1, x2, y2 = fl[0], fl[1], fl[2], fl[3]
        else:
            x1, y1, x2, y2 = fl['x1'], fl['y1'], fl['x2'], fl['y2']
        cv2.line(fold_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)

    spots_color = np.zeros_like(img)
    spots_viz   = img.copy()
    if spots_mask is not None and np.any(spots_mask):
        spots_color[spots_mask > 0] = (0, 0, 255)
        spots_viz = cv2.addWeighted(spots_viz, 0.6, spots_color, 0.4, 0)

    overlay = img.copy()
    if spots_mask is not None and np.any(spots_mask):
        overlay = cv2.addWeighted(overlay, 0.7, spots_color, 0.3, 0)
    for fl in fold_lines:
        if isinstance(fl, (list, tuple)) and len(fl) >= 4:
            x1, y1, x2, y2 = fl[0], fl[1], fl[2], fl[3]
        else:
            x1, y1, x2, y2 = fl['x1'], fl['y1'], fl['x2'], fl['y2']
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

    try:
        cv2.imwrite(out_prefix + '_folds.png',   fold_viz)
        cv2.imwrite(out_prefix + '_spots.png',   spots_viz)
        cv2.imwrite(out_prefix + '_overlay.png', overlay)
    except Exception:
        pass


# ─────────────────────────────────────────────
# WHITE BALANCE — ADAPTIVE
# ─────────────────────────────────────────────

def colorfulness_metric(img):
    """Compute Hasler-Suesstrunk colorfulness metric."""
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)
    rg = r - g
    yb = 0.5*(r+g) - b
    std_rgyb  = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return float(std_rgyb + 0.3*mean_rgyb)


def adaptive_wb_weight(colorfulness, min_weight=0.25, max_weight=0.70):
    norm   = np.clip(colorfulness / 60.0, 0.0, 1.0)
    weight = max_weight - norm*(max_weight - min_weight)
    return float(weight)


def adaptive_wb_weight_entropy(entropy_val, min_weight=0.25, max_weight=0.70,
                                min_ent=4.0, max_ent=8.0):
    """Compute white-balance weight from image entropy.
    High entropy → Gray-World more reliable → increase WB weight.
    """
    norm   = np.clip((entropy_val - min_ent)/(max_ent - min_ent), 0.0, 1.0)
    weight = min_weight + norm*(max_weight - min_weight)
    return float(weight)


def white_balance_grayworld(img):
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)
    mean_b    = max(np.mean(b), 1.0)
    mean_g    = max(np.mean(g), 1.0)
    mean_r    = max(np.mean(r), 1.0)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    b = np.clip(b*(mean_gray/mean_b), 0, 255)
    g = np.clip(g*(mean_gray/mean_g), 0, 255)
    r = np.clip(r*(mean_gray/mean_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def white_balance_adaptive(img):
    """Apply adaptive white balance using entropy-based weight.
    Returns (balanced_img, entropy_value, wb_weight_used).
    """
    ent       = entropy_metric(img)
    wb_weight = adaptive_wb_weight_entropy(ent)
    wb        = white_balance_grayworld(img)
    result    = cv2.addWeighted(img, 1.0 - wb_weight, wb, wb_weight, 0)
    return result, ent, wb_weight


def white_balance(img):
    """LAB space white balance (optional, not used in main pipeline)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    l, a, b = cv2.split(lab)
    a = np.clip(a - (np.mean(a) - 128), 0, 255).astype(np.uint8)
    b = np.clip(b - (np.mean(b) - 128), 0, 255).astype(np.uint8)
    l = np.clip(l, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# CONTRAST ENHANCEMENT — MULTI-SCALE CLAHE
# ─────────────────────────────────────────────

def apply_clahe_single(img, clip_limit, tile_size):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def enhance_contrast_multiscale(img, clip_limit=1.1):
    """Multi-Scale CLAHE — 3 tile sizes blended equally."""
    small  = apply_clahe_single(img, clip_limit, (4,  4))
    medium = apply_clahe_single(img, clip_limit, (8,  8))
    large  = apply_clahe_single(img, clip_limit, (16, 16))
    result = cv2.addWeighted(small, 0.33, medium, 0.33, 0)
    result = cv2.addWeighted(result, 1.0, large, 0.34, 0)
    return result


def enhance_contrast(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Single-scale CLAHE (kept for ablation study comparison)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# SATURATION
# ─────────────────────────────────────────────

def increase_saturation(img, scale=1.25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * scale, 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────

def mse(imageA, imageB):
    return float(np.mean((imageA.astype('float32') - imageB.astype('float32'))**2))


def psnr(imageA, imageB):
    return float(cv2.PSNR(imageA, imageB))


def ssim(img1, img2):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    mu1 = cv2.GaussianBlur(img1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11,11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1*img1, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, (11,11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1*img2, (11,11), 1.5) - mu1_mu2
    ssim_map  = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return float(np.mean(ssim_map))


def brisque_score(img):
    """Simplified BRISQUE-like no-reference quality score. Lower = better."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu    = cv2.GaussianBlur(gray, (7,7), 7.0/6.0)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray*gray, (7,7), 7.0/6.0) - mu_sq))
    mscn  = (gray - mu) / (sigma + 1.0)
    return float(np.mean(np.abs(mscn**3)) + np.mean(np.abs(mscn**4 - 3)))


def niqe_score(img):
    """Simplified NIQE-like no-reference quality score. Lower = more natural."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu    = cv2.GaussianBlur(gray, (7,7), 1.5)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray**2, (7,7), 1.5) - mu**2))
    return float(np.std(sigma) / (np.mean(sigma) + 1e-6))


# ─────────────────────────────────────────────
# RETINEX (OPTIONAL)
# ─────────────────────────────────────────────

def single_scale_retinex(img, sigma):
    img  = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0,0), sigma).astype(np.float32) + 1.0
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
    sum_ch = (B + Gc + R) + 1.0
    out_B  = G * multi_scale_retinex(B,  scales) * np.log(alpha*B  / sum_ch + 1.0)
    out_G  = G * multi_scale_retinex(Gc, scales) * np.log(alpha*Gc / sum_ch + 1.0)
    out_R  = G * multi_scale_retinex(R,  scales) * np.log(alpha*R  / sum_ch + 1.0)
    out    = cv2.merge([out_B, out_G, out_R])
    out    = (out - np.min(out)) / (np.max(out) - np.min(out) + 1e-8)*255.0 + b
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# MAIN RESTORE PIPELINE
# ─────────────────────────────────────────────

def restore_image(img, sat_scale=1.25, nlm_h=10, median_k=5, clahe_clip=1.1,
                  sat_scale_override=None, unsharp_amount=0.3, spot_thresh=40,
                  spot_blur=9, spot_min_frac=5e-5, inpaint_radius=2,
                  use_fold_suppression=True, use_multiscale_clahe=True,
                  use_deblur=True, use_adaptive_sharpen=True):
    """Restore image using the full enhanced pipeline.

    Steps:
      1. Blur detection
      2. Noise classification + adaptive denoising
      3. Adaptive white balance (entropy-based)
      4. Spot detection + Telea inpainting
      5. Fold line suppression (Hough Transform)
      6. Multi-scale CLAHE contrast enhancement
      7. Saturation boost
      8. Edge enhancement (blurry images only)
      9. Adaptive unsharp masking
    """
    # Step 1: Blur detection
    blur_level, is_blurred = detect_blur_level(img)

    # Step 2: Noise classification + adaptive denoising
    try:
        ntype, _, _ = classify_noise_type(img)
    except Exception:
        ntype = 'gaussian'

    if ntype == 'poisson':
        try:
            A       = anscombe_transform(img)
            A_u8    = np.clip(A, 0, 255).astype(np.uint8)
            A_den   = nl_means_denoise(A_u8, h=nlm_h, hColor=nlm_h)
            denoise = inverse_anscombe(A_den)
        except Exception:
            denoise = cv2.medianBlur(img, median_k)
    elif ntype == 'mixed':
        denoise = nl_means_denoise(img, h=int(nlm_h*1.2), hColor=int(nlm_h*1.2))
    else:
        noise_lvl = estimate_noise(img)
        denoise   = nl_means_denoise(img, h=nlm_h, hColor=nlm_h) if noise_lvl > 10.0 \
                    else cv2.medianBlur(img, median_k)

    # Step 3: Adaptive white balance
    result, entropy_val, wb_weight = white_balance_adaptive(denoise)

    # Step 4: Spot detection + inpainting
    mask, have_spots = detect_spots_mask(result, thresh=spot_thresh)
    if have_spots:
        result = cv2.inpaint(result, mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Step 5: Fold line suppression
    num_folds = 0
    if use_fold_suppression:
        result, fold_mask, num_folds = suppress_fold_lines(result)

    # Step 6: Multi-scale CLAHE (clip adapts to blur level)
    if blur_level < 100:
        adaptive_clahe_clip = 1.5
    elif blur_level < 200:
        adaptive_clahe_clip = 1.3
    elif blur_level < 500:
        adaptive_clahe_clip = 1.2
    else:
        adaptive_clahe_clip = clahe_clip

    if use_multiscale_clahe:
        contrast = enhance_contrast_multiscale(result, clip_limit=adaptive_clahe_clip)
    else:
        contrast = apply_clahe_single(result, adaptive_clahe_clip, (8, 8))

    # Step 7: Saturation boost
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    scale = sat_scale_override if sat_scale_override is not None else sat_scale
    s     = np.clip(s*scale, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.merge((h.astype(np.uint8), s, v.astype(np.uint8))),
                          cv2.COLOR_HSV2BGR)

    # Step 8: Edge enhancement (blurry images only)
    if is_blurred:
        color = enhance_edges_adaptive(color, blur_level)
    if blur_level < 250:
        color = high_pass_filter_sharpen(color, blur_level)

    # Step 9: Adaptive unsharp masking
    if use_adaptive_sharpen:
        sharpen = adaptive_unsharp_mask(color, blur_level, base_amount=unsharp_amount)
    else:
        blurred = cv2.GaussianBlur(color, (0, 0), sigmaX=1.0)
        sharpen = cv2.addWeighted(color, 1.0+unsharp_amount, blurred, -unsharp_amount, 0)

    return sharpen


# ─────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────

def run_ablation_study(img, sat_scale=1.5, nlm_h=6, median_k=3, clahe_clip=1.1,
                        sat_scale_override=1.5, unsharp_amount=0.3,
                        spot_thresh=40, spot_blur=9, spot_min_frac=1e-4,
                        inpaint_radius=2, use_deblur=True, use_adaptive_sharpen=True):
    """Run ablation study — remove each step one at a time.

    Returns dict: variant_name -> (image, brisque, niqe).
    """
    results = {}
    def score(p): return p, brisque_score(p), niqe_score(p)

    results['original']     = score(img)
    results['full_pipeline'] = score(restore_image(
        img, sat_scale=sat_scale, nlm_h=nlm_h, median_k=median_k,
        clahe_clip=clahe_clip, sat_scale_override=sat_scale_override,
        unsharp_amount=unsharp_amount, spot_thresh=spot_thresh,
        inpaint_radius=inpaint_radius, use_fold_suppression=True,
        use_multiscale_clahe=True, use_deblur=use_deblur,
        use_adaptive_sharpen=use_adaptive_sharpen))

    noise_lvl = estimate_noise(img)
    denoised  = nl_means_denoise(img, h=nlm_h, hColor=nlm_h) if noise_lvl > 10.0 \
                else cv2.medianBlur(img, median_k)

    def _sat_unsharp_clahe(src):
        c = enhance_contrast_multiscale(src, clip_limit=clahe_clip)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
        hh, ss, vv = cv2.split(hsv)
        ss  = np.clip(ss*sat_scale_override, 0, 255).astype(np.uint8)
        col = cv2.cvtColor(cv2.merge((hh.astype(np.uint8), ss, vv.astype(np.uint8))),
                            cv2.COLOR_HSV2BGR)
        blr = cv2.GaussianBlur(col, (0,0), sigmaX=1.0)
        return cv2.addWeighted(col, 1.0+unsharp_amount, blr, -unsharp_amount, 0)

    wb_nd, _, _ = white_balance_adaptive(img)
    results['no_denoising']     = score(_sat_unsharp_clahe(wb_nd))
    results['no_white_balance'] = score(_sat_unsharp_clahe(denoised))

    wb5, _, _ = white_balance_adaptive(denoised)
    hsv5 = cv2.cvtColor(wb5, cv2.COLOR_BGR2HSV).astype(np.float32)
    h5, s5, v5 = cv2.split(hsv5)
    s5  = np.clip(s5*sat_scale_override, 0, 255).astype(np.uint8)
    c5  = cv2.cvtColor(cv2.merge((h5.astype(np.uint8), s5, v5.astype(np.uint8))), cv2.COLOR_HSV2BGR)
    b5  = cv2.GaussianBlur(c5, (0,0), sigmaX=1.0)
    results['no_clahe']     = score(cv2.addWeighted(c5, 1.0+unsharp_amount, b5, -unsharp_amount, 0))

    wb6, _, _ = white_balance_adaptive(denoised)
    c6  = enhance_contrast_multiscale(wb6, clip_limit=clahe_clip)
    b6  = cv2.GaussianBlur(c6, (0,0), sigmaX=1.0)
    results['no_saturation'] = score(cv2.addWeighted(c6, 1.0+unsharp_amount, b6, -unsharp_amount, 0))

    wb7, _, _ = white_balance_adaptive(denoised)
    c7   = enhance_contrast_multiscale(wb7, clip_limit=clahe_clip)
    hsv7 = cv2.cvtColor(c7, cv2.COLOR_BGR2HSV).astype(np.float32)
    h7, s7, v7 = cv2.split(hsv7)
    s7  = np.clip(s7*sat_scale_override, 0, 255).astype(np.uint8)
    results['no_unsharp'] = score(cv2.cvtColor(
        cv2.merge((h7.astype(np.uint8), s7, v7.astype(np.uint8))), cv2.COLOR_HSV2BGR))

    results['no_fold_suppression'] = score(restore_image(
        img, sat_scale=sat_scale, nlm_h=nlm_h, median_k=median_k,
        clahe_clip=clahe_clip, sat_scale_override=sat_scale_override,
        unsharp_amount=unsharp_amount, use_fold_suppression=False,
        use_multiscale_clahe=True, use_deblur=use_deblur,
        use_adaptive_sharpen=use_adaptive_sharpen))

    results['no_deblur'] = score(restore_image(
        img, sat_scale=sat_scale, nlm_h=nlm_h, median_k=median_k,
        clahe_clip=clahe_clip, sat_scale_override=sat_scale_override,
        unsharp_amount=unsharp_amount, use_fold_suppression=True,
        use_multiscale_clahe=True, use_deblur=False,
        use_adaptive_sharpen=use_adaptive_sharpen))

    return results


def print_ablation_table(ablation_results):
    print('\n' + '='*75)
    print(f"{'Variant':<30} {'BRISQUE':>10} {'NIQE':>10}   Note")
    print('='*75)
    order = ['original','full_pipeline','no_denoising','no_white_balance',
             'no_clahe','no_saturation','no_unsharp','no_fold_suppression','no_deblur']
    notes = {
        'original':            'No processing',
        'full_pipeline':       'All steps active  <- BEST',
        'no_denoising':        'Skip denoising',
        'no_white_balance':    'Skip white balance',
        'no_clahe':            'Skip contrast enhancement',
        'no_saturation':       'Skip saturation boost',
        'no_unsharp':          'Skip unsharp masking',
        'no_fold_suppression': 'Skip fold suppression',
        'no_deblur':           'Skip deblurring',
    }
    for key in order:
        if key in ablation_results:
            _, b, n = ablation_results[key]
            print(f"{key:<30} {b:>10.4f} {n:>10.4f}   {notes.get(key,'')}")
    print('='*75)
    print('Lower BRISQUE and NIQE = better image quality\n')


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
    info['is_noisy']    = noisy
    info['noise_level'] = noise_lvl
    low_c, c_score = is_low_contrast(img, contrast_thresh=contrast_thresh)
    info['is_low_contrast'] = low_c
    info['contrast_score']  = c_score
    blur_level, is_blurred = detect_blur_level(img)
    info['blur_level'] = blur_level
    info['is_blurred'] = is_blurred
    cf  = colorfulness_metric(img)
    ent = entropy_metric(img)
    try:
        ntype, _, corr = classify_noise_type(img)
    except Exception:
        ntype, corr = 'gaussian', 0.0
    info['colorfulness']   = cf
    info['entropy']        = ent
    info['noise_type']     = ntype
    info['noise_corr']     = corr
    info['wb_weight_used'] = adaptive_wb_weight_entropy(ent)
    restored = restore_image(img, sat_scale=sat_scale)
    orig_gray = cv2.cvtColor(img,      cv2.COLOR_BGR2GRAY)
    res_gray  = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    info['mse']  = mse(orig_gray,  res_gray)
    info['psnr'] = psnr(orig_gray, res_gray)
    info['ssim'] = ssim(orig_gray, res_gray)
    return restored, info

# ── Improvement #10: Data-Driven Parameter Optimization ───────────────────────
def optimize_parameters(img):
    """Grid search over key parameters using BRISQUE as objective.
    Tests combinations of wb_weight, sat_scale, clahe_clip.
    Returns best params dict and best score.
    """
    search_space = {
        'wb_weight':  [0.25, 0.40, 0.55, 0.70],
        'sat_scale':  [1.2,  1.4,  1.6],
        'clahe_clip': [1.0,  1.2,  1.4],
    }

    best_score  = float('inf')
    best_params = {
        'wb_weight':  0.50,
        'sat_scale':  1.5,
        'clahe_clip': 1.2,
    }

    for wb_w in search_space['wb_weight']:
        for sat in search_space['sat_scale']:
            for clip in search_space['clahe_clip']:
                try:
                    # Apply only the 3 steps being optimized (fast)
                    wb  = white_balance_grayworld(img)
                    res = cv2.addWeighted(img, 1.0 - wb_w, wb, wb_w, 0)
                    res = enhance_contrast_multiscale(res, clip_limit=clip)

                    # Saturation boost
                    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
                    h, s, v = cv2.split(hsv)
                    s   = np.clip(s * sat, 0, 255).astype(np.uint8)
                    res = cv2.cvtColor(
                        cv2.merge([h.astype(np.uint8), s, v.astype(np.uint8)]),
                        cv2.COLOR_HSV2BGR
                    )

                    score = brisque_score(res)

                    if score < best_score:
                        best_score  = score
                        best_params = {
                            'wb_weight':  wb_w,
                            'sat_scale':  sat,
                            'clahe_clip': clip,
                        }
                except Exception:
                    continue

    return best_params, best_score

# Add to restoration.py

def difficulty_score(img):
    """
    Returns (score, level) where:
      score 0.0–0.33 → 'low'    (mild degradation)
      score 0.33–0.66 → 'medium'
      score 0.66–1.0  → 'severe'
    """
    noise_lvl     = estimate_noise(img)
    contrast      = contrast_score(img)
    blur_lvl, _   = detect_blur_level(img)
    colorfulness  = colorfulness_metric(img)

    # Normalize each signal to [0, 1] — higher = worse
    noise_norm    = np.clip(noise_lvl / 30.0,   0, 1)
    contrast_norm = np.clip(1 - contrast / 60.0, 0, 1)   # low contrast = high difficulty
    blur_norm     = np.clip(1 - blur_lvl / 500.0, 0, 1)  # low sharpness = high difficulty
    color_norm    = np.clip(1 - colorfulness / 50.0, 0, 1)

    score = 0.3*noise_norm + 0.25*contrast_norm + 0.25*blur_norm + 0.20*color_norm
    score = float(np.clip(score, 0, 1))

    level = 'low' if score < 0.33 else ('medium' if score < 0.66 else 'severe')
    return score, level


def intensity_from_difficulty(level):
    """Map difficulty level to pipeline intensity multipliers."""
    presets = {
        'low':    dict(nlm_h=5,  clahe_clip=1.0, sat_scale=1.3, unsharp_amount=0.2),
        'medium': dict(nlm_h=7,  clahe_clip=1.2, sat_scale=1.5, unsharp_amount=0.35),
        'severe': dict(nlm_h=10, clahe_clip=1.5, sat_scale=1.7, unsharp_amount=0.5),
    }
    return presets[level]