What I changed

restoration.py:
Fold-line detection: detect_fold_lines(...) now returns per-line dicts with confidence, orientation, and thickness (estimates based on edge/gradient sampling).
Fold mask building: build_fold_mask(...) uses per-line thickness and confidence.
Fold suppression: suppress_fold_lines(...) applies orientation-aware smoothing, avoids strongly textured regions, refines the mask, then inpaints and blends.
Spot detection: detect_spots_mask(...) now uses multi-scale median residuals and suppresses detections in textured areas.
Spot inpainting: inpaint_spots(...) refines the mask before Telea inpainting and accepts inpaint_radius.
How to run the updated modules




9. Replace Heuristic Decisions with Continuous Adaptation
Current problem: Your main.py uses hard if blur_level < 100 / < 200 / < 500 branches — abrupt jumps.
Fix: Replace the 4 if/elif blocks in process_all() with a smooth interpolation function:# In main.py — replace the entire if/elif blur block with this:

def compute_params_continuous(blur_level):
    """Smoothly interpolate parameters based on blur level (no hard jumps)."""
    # Normalize blur to 0-1 range (0=very blurry, 1=sharp)
    sharpness = np.clip(blur_level / 500.0, 0.0, 1.0)

    return dict(
        nlm_h            = int(round(8 - 2 * sharpness)),     # 8 → 6
        median_k         = 3,
        clahe_clip       = round(1.5 - 0.4 * sharpness, 2),   # 1.5 → 1.1
        sat_scale_override = 1.5,
        unsharp_amount   = round(0.5 - 0.2 * sharpness, 2),   # 0.5 → 0.3
        spot_thresh      = 40,
        inpaint_radius   = 2,
        use_fold_suppression  = True,
        use_multiscale_clahe  = True,
        use_deblur       = sharpness < 0.5,   # smooth cutoff at midpoint
        use_adaptive_sharpen  = True,
    )

# Then call it:
mild_params = compute_params_continuous(blur_level)
mild_variant = restore_image(img_proc, sat_scale=1.5, **mild_params)

10. Data-Driven Parameter Optimization
Fix: Add an optimizer that searches for the best WB weight, CLAHE blend weights, and saturation scale using BRISQUE as the objective:
# Add to restoration.py

def optimize_parameters(img, search_space=None):
    """
    Grid search over key parameters using BRISQUE as objective.
    Returns best params dict. Runs fast (~1-2 sec) on downscaled image.
    """
    if search_space is None:
        search_space = {
            'wb_weight':      [0.25, 0.40, 0.55, 0.70],
            'sat_scale':      [1.2, 1.4, 1.6],
            'clahe_clip':     [1.0, 1.2, 1.4],
        }

    best_score  = float('inf')
    best_params = {}

    for wb_w in search_space['wb_weight']:
        for sat in search_space['sat_scale']:
            for clip in search_space['clahe_clip']:
                # Apply just these three steps (fast)
                wb  = white_balance_grayworld(img)
                res = cv2.addWeighted(img, 1-wb_w, wb, wb_w, 0)
                res = enhance_contrast_multiscale(res, clip_limit=clip)
                hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv)
                s = np.clip(s * sat, 0, 255).astype(np.uint8)
                res = cv2.cvtColor(
                    cv2.merge([h.astype(np.uint8), s, v.astype(np.uint8)]),
                    cv2.COLOR_HSV2BGR
                )
                score = brisque_score(res)
                if score < best_score:
                    best_score  = score
                    best_params = {'wb_weight': wb_w,
                                   'sat_scale': sat,
                                   'clahe_clip': clip}

    return best_params, best_score
    Then use it in restore_image():
    # In restore_image() — after Step 1 blur detection, add:
opt_params, _ = optimize_parameters(img)
# Override the fixed defaults with optimized values:
sat_scale    = opt_params.get('sat_scale',  sat_scale)
clahe_clip   = opt_params.get('clahe_clip', clahe_clip)
# pass opt_params['wb_weight'] into white_balance_adaptive if needed

11. Image Difficulty-Aware Processing
Fix: Add a difficulty_score() function and use it to scale pipeline intensity:
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
    Use it in main.py:
    # Replace mild_params block with:
diff_score, diff_level = difficulty_score(img_proc)
mild_params = intensity_from_difficulty(diff_level)
logging.info('Difficulty: %.2f (%s)', diff_score, diff_level)
mild_variant = restore_image(img_proc, **mild_params)


12. Improved Noise Estimation
Fix: Replace the simple estimate_noise() with patch-based + frequency-domain analysis:
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