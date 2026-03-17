# Color Restoration of Old and Damaged Photographs

This repository implements a classical Digital Image Processing (DIP) pipeline to restore faded, noisy, or dust‑spotted historical photographs using OpenCV and NumPy.

**What this project does**
- Input: scanned/photographed old images placed in [dataset/old_images](dataset/old_images).
- Output: restored images written to [results/restored_images](results/restored_images).
- Core techniques: denoising, white balance, spot detection+inpainting, local contrast enhancement (CLAHE) or MSRCR, saturation correction, and sharpening.

**Quick start**
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run the batch processor from the project root (interactive):

```powershell
cd C:/Users/nikit/Desktop/dip/color_restoration_project
python main.py
```

- Run headless (recommended for large batches) with `--no-display`:

```powershell
python main.py --no-display
```

**Files of interest**
- `main.py`: batch orchestration, presets, result saving.
- `restoration.py`: core algorithms and helpers (`restore_image()`, `restore_image_msr_wrapper()`, `analyze_and_restore()`).

**End‑to‑end flow (text flowchart)**
1. Load image
2. Analyze image (noise level, grayscale check, contrast score)
3. Denoise (Non‑Local‑Means if noisy, median blur otherwise)
4. White balance (Gray‑World by default)
5. Detect small spots (dust/specks) → inpaint with Telea if present
6. Contrast enhancement: CLAHE on L channel (or MSRCR when selected)
7. Increase saturation in HSV (small multiplier)
8. Mild sharpening (unsharp mask)
9. Save a single canonical restored image per input (`restored_{original_name}`) — currently the pipeline outputs one "mild" variant by default — and compute metrics (MSE/PSNR/SSIM)

This simple flow is implemented by `analyze_and_restore()` → `restore_image()` with optional `restore_image_msr_wrapper()` for MSRCR.

**Detailed algorithms / methods used (map to pipeline steps)**
- Noise estimation: `estimate_noise()` — high‑frequency residual stddev on grayscale.
- Denoising: `nl_means_denoise()` (OpenCV `fastNlMeansDenoisingColored`) or `cv2.medianBlur()`.
- Edge‑preserving smoothing (documented helper): bilateral filtering (`remove_noise()`).
- Spot/dust detection: `detect_spots_mask()` — median residual threshold + morphology.
- Inpainting: `inpaint_spots()` / `cv2.inpaint(..., INPAINT_TELEA)`.
- White balance: `white_balance_grayworld()` (per‑channel mean scaling) and `white_balance()` (LAB mean shift).
- Contrast: `enhance_contrast()` / CLAHE on L channel; alternate: `msrcr()` (Multi‑Scale Retinex with Color Restoration).
- Saturation: `increase_saturation()` — multiply S channel in HSV.
- Sharpening: `sharpen_image()` and unsharp mask via `cv2.addWeighted()`.
- Metrics: `mse()`, `psnr()` (OpenCV), `ssim()` (single‑channel implementation in code).

**Known issues & suggested fixes (what to change/improve now)**
- Robustness: `main.py` now includes per‑image try/except handling and safe plotting checks. Use `--no-display` for headless runs.
- Output policy: the script currently writes a single canonical restored image per input (`restored_{original_name}`) containing the "mild" preset. If you want multiple variants (balanced/aggressive/msr), either modify `main.py` or request a `--preset` option to be added.
- Parameter tuning: current presets are reasonable; add a small CLI or config file (`config.yaml`) to tweak presets without editing code.
- Inpainting: `cv2.inpaint` (Telea) handles small spots but fails on large tears; consider deep inpainting (LaMa) for large damage.

**Tuning tips (quick reference)**
- `nlm_h`: 6–12 mild, 15–25 aggressive denoising.
- `clahe_clip`: 1.0–3.0 (higher → stronger local contrast, risk amplifying noise).
- `sat_scale`: 1.05–1.30 (avoid >1.5 unless colorless image).
- `unsharp_amount`: 0.2–0.5 (higher → halos).
- `spot_thresh`: 25–40, `spot_blur`: adjust to spot size.

**How to run faster / test parameters**
- For quick experimentation downscale images by 2×, find good parameters, then reprocess full resolution.
- To batch process large sets, run headless and use Python's `multiprocessing` to process images in parallel (one process per core).

**Extensions & ML options**
- Replace NLM with a learned denoiser (e.g., DnCNN, FFDNet) for better preservation of detail.
- For grayscale or severely desaturated photographs, integrate a colorization network (DeOldify or a lightweight colorizer).
- For large hole/tear inpainting, integrate LaMa or partial convolution inpainting.
- Add super‑resolution (Real‑ESRGAN) as an optional final step to enhance detail.

**Checklist for a robust release**
- [x] Add `--no-display` flag and safe plotting checks (implemented).
- [x] Add per‑image try/except and logging (implemented).
- [ ] Add CLI flags for presets and parallel jobs (`--preset`, `--jobs`).
- [ ] Add unit tests for small pure functions (noise estimator, is_grayscale, detect_spots_mask).
- [ ] Add a reproducible example and sample images (small size) in `dataset/example/`.

**Detailed Flowchart & Explanation**

🎯 Algorithms & Methods Used (concise map)

- **1. Image Analysis (Decision Making)**
	- Noise Estimation
		- Method: High‑frequency residual standard deviation
		- Function: `estimate_noise()`
	- Grayscale Detection
		- Method: Mean difference between B/G/R channels
		- Function: `is_grayscale()`
	- Contrast Detection
		- Method: Standard deviation of luminance
		- Functions: `contrast_score()`, `is_low_contrast()`
	- Used in: `analyze_and_restore()` to decide processing steps

- **2. Noise Removal (Denoising)**
	- Non‑Local Means (best for photographic noise)
		- OpenCV: `fastNlMeansDenoisingColored()`
		- Wrapper: `nl_means_denoise()`
	- Median Filtering (good for salt‑and‑pepper / light specks)
		- OpenCV: `cv2.medianBlur()`
	- Bilateral Filtering (edge‑preserving, optional)
		- Function: `remove_noise()`

- **3. Dust / Scratch Removal**
	- Spot Detection
		- Method: Residual (median) thresholding + morphological cleanup
		- Function: `detect_spots_mask()`
	- Inpainting
		- Algorithm: Telea inpainting
		- OpenCV: `cv2.inpaint(..., cv2.INPAINT_TELEA)`
		- Function: `inpaint_spots()`

- **4. Color Correction (White Balance)**
	- Gray World
		- Assumption: Average color should be gray
		- Function: `white_balance_grayworld()`
	- LAB channel shift
		- Method: adjust a/b channel means in LAB
		- Function: `white_balance()`

- **5. Contrast Enhancement**
	- CLAHE (local adaptive histogram equalization)
		- Applied on L channel (LAB)
		- Function: `enhance_contrast()`
	- Retinex (MSRCR)
		- Multi‑Scale Retinex + Color Restoration for strong illumination/contrast correction
		- Functions: `single_scale_retinex()`, `multi_scale_retinex()`, `msrcr()`

- **6. Color Enhancement**
	- Saturation boost
		- Method: multiply S channel in HSV
		- Function: `increase_saturation()`

- **7. Image Sharpening**
	- Unsharp masking (weighted add)
		- `cv2.addWeighted()` used in pipeline (param `unsharp_amount`)
	- Kernel sharpening
		- Function: `sharpen_image()` (3×3 kernel)

- **8. Quality Evaluation Metrics**
	- MSE, PSNR, SSIM (implemented in `mse()`, `psnr()`, `ssim()`)
	- Used to log and compare restored output inside `analyze_and_restore()`

⚙️ **Final Pipeline (step‑by‑step)**
1. Load input image
2. Analyze (noise, grayscale, contrast)
3. Denoise (NLM if noisy; median otherwise)
4. White balance (Gray World by default)
5. Detect spots and inpaint small defects
6. Contrast enhancement (CLAHE; optionally MSRCR)
7. Saturation boost
8. Mild sharpening (unsharp mask)
9. Save single canonical restored image per input (`restored_{original_name}` — currently using the "mild" preset) and compute metrics (MSE/PSNR/SSIM)

📝 Notes / Rules of Thumb
- If `estimate_noise()` > ~10 → prefer NLM (`nlm_h` higher for stronger denoising).
- CLAHE `clipLimit` 1.0–3.0; reduce if you see amplified noise.
- `sat_scale` ~1.05–1.30; avoid extreme saturation values.
- Use deep inpainting for large tears (not implemented here).

If you'd like, I can render this flowchart visually (SVG or PNG) and embed it in the README, or add a `--preset` flag to choose alternatives instead of the default mild preset.

**Contact & credits**
This repository uses OpenCV and NumPy as primary dependencies. See `requirements.txt`.

If you want, I can:
- Add a `--no-display` CLI option and safe checks in `main.py` (recommended for batch runs).
- Implement the robustness fixes above (error handling, plot guard, small unit tests).

Choose what I should do next and I will apply the changes.
