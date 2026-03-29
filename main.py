"""
Color Restoration Application

Required libraries:
- OpenCV: pip install opencv-python
- NumPy: pip install numpy
- Matplotlib: pip install matplotlib

How to run:
1. Place your input images (jpg/png/bmp/tiff) into:
   color_restoration_project/dataset/old_images/
2. Run:
   python main.py
3. Restored images will be saved to:
   color_restoration_project/results/restored_images/

New features:
- Adaptive white balance using Hasler-Suesstrunk colorfulness metric
- Fold line suppression using Hough Transform
- Multi-Scale CLAHE (3 tile sizes blended)
- Ablation study: run with --ablation flag
"""

import os
import argparse
import logging
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend — fixes MemoryError on large images
import matplotlib.pyplot as plt
from restoration import (restore_image, analyze_and_restore,
                          run_ablation_study, print_ablation_table,
                          mse, psnr, ssim, colorfulness_metric, adaptive_wb_weight)


def process_all(input_dir, output_dir, display=True, run_ablation=False):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    if not files:
        print('No images found in', input_dir)
        return

    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        logging.info('Processing: %s', in_path)
        try:
            img = cv2.imread(in_path)
            if img is None:
                logging.warning('Skipping (cannot read): %s', fname)
                continue

            # Adaptive analysis
            _, info = analyze_and_restore(img)

            # Tuned parameters for warm old photo restoration
            mild_params = dict(
                nlm_h=6,
                median_k=3,
                clahe_clip=1.1,
                sat_scale_override=1.5,
                unsharp_amount=0.3,
                spot_thresh=50,
                spot_blur=9,
                spot_min_frac=1e-4,
                inpaint_radius=2,
                use_fold_suppression=True,
                use_multiscale_clahe=True,
            )

            mild_variant = restore_image(img, sat_scale=1.5, **mild_params)

            # Save restored image
            out_name = f'restored_{fname}'
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, mild_variant)
            logging.info('Saved restored image to: %s', out_path)

            # Compute metrics
            try:
                orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mild_gray = cv2.cvtColor(mild_variant, cv2.COLOR_BGR2GRAY)
                info['mse']  = mse(orig_gray, mild_gray)
                info['psnr'] = psnr(orig_gray, mild_gray)
                info['ssim'] = ssim(orig_gray, mild_gray)
            except Exception:
                logging.exception('Failed to compute metrics')

            # Log results
            cond = 'Noisy' if info.get('is_noisy') else 'Clean'
            if info.get('is_low_contrast'):
                cond += ' + Low-Contrast'
            if info.get('is_grayscale'):
                cond += ' + Grayscale'

            cf   = info.get('colorfulness', 0.0)
            wb_w = info.get('wb_weight_used', 0.0)

            logging.info('Detected condition  : %s', cond)
            logging.info('Colorfulness        : %.2f  →  WB weight used: %.2f', cf, wb_w)
            logging.info('Noise level         : %.2f,  Contrast score: %.2f',
                         info.get('noise_level', 0), info.get('contrast_score', 0))
            logging.info('MSE: %.2f,  PSNR: %.2f dB,  SSIM: %.4f',
                         info.get('mse', 0), info.get('psnr', 0), info.get('ssim', 0))

            # ── Ablation Study ──────────────────────────────────────────────
            if run_ablation:
                logging.info('Running ablation study for %s ...', fname)
                ablation_results = run_ablation_study(
                    img,
                    sat_scale=1.5,
                    nlm_h=6,
                    median_k=3,
                    clahe_clip=1.1,
                    sat_scale_override=1.5,
                    unsharp_amount=0.3,
                    spot_thresh=50,
                    spot_blur=9,
                    spot_min_frac=1e-4,
                    inpaint_radius=2,
                )
                print_ablation_table(ablation_results)

                # Save ablation comparison grid
                ablation_out = os.path.join(output_dir, f'ablation_{fname}')
                _save_ablation_grid(ablation_results, ablation_out)
                logging.info('Ablation grid saved to: %s', ablation_out)

            # ── Save comparison image (no interactive display) ───────────────
            if display:
                try:
                    _save_comparison(img, mild_variant, cf, wb_w,
                                     os.path.join(output_dir, f'comparison_{fname}'))
                    logging.info('Comparison image saved: comparison_%s', fname)
                except Exception:
                    logging.exception('Error saving comparison for %s', fname)

        except Exception:
            logging.exception('Processing failed for %s', in_path)
            continue


def _save_comparison(img, mild_variant, cf, wb_w, out_path):
    """Save side-by-side original vs restored comparison image.

    Uses Agg backend (no screen needed) — avoids MemoryError on large images.
    Downscales large images before plotting to stay within memory limits.
    """
    # Downscale if image is too large (max 1000px wide per panel)
    max_width = 1000
    h, w = img.shape[:2]
    if w > max_width:
        scale   = max_width / w
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        img_display     = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        variant_display = cv2.resize(mild_variant, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_display     = img
        variant_display = mild_variant

    orig_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    rest_rgb = cv2.cvtColor(variant_display, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    axes[0].imshow(orig_rgb)
    axes[0].set_title('Original Image', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(rest_rgb)
    axes[1].set_title(
        f'Restored (Mild)\nColorfulness={cf:.1f}  WB weight={wb_w:.2f}',
        fontsize=12)
    axes[1].axis('off')

    # Force same scale on both panels
    axes[0].set_xlim(0, orig_rgb.shape[1])
    axes[0].set_ylim(orig_rgb.shape[0], 0)
    axes[1].set_xlim(0, orig_rgb.shape[1])
    axes[1].set_ylim(orig_rgb.shape[0], 0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)  # Always close figure to free memory


def _save_ablation_grid(ablation_results, out_path):
    """Save a grid image showing all ablation variants side by side."""
    order = ['original', 'full_pipeline', 'no_denoising', 'no_white_balance',
             'no_clahe', 'no_saturation', 'no_unsharp', 'no_fold_suppression']
    labels = {
        'original':            'Original',
        'full_pipeline':       'Full Pipeline',
        'no_denoising':        'No Denoising',
        'no_white_balance':    'No White Balance',
        'no_clahe':            'No CLAHE',
        'no_saturation':       'No Saturation',
        'no_unsharp':          'No Unsharp Mask',
        'no_fold_suppression': 'No Fold Suppression',
    }
    variants = [(k, ablation_results[k]) for k in order if k in ablation_results]
    n    = len(variants)
    cols = 4
    rows = (n + cols - 1) // cols

    # Downscale each variant image before plotting to save memory
    max_width = 400
    def downscale(img_v):
        h, w = img_v.shape[:2]
        if w > max_width:
            scale = max_width / w
            return cv2.resize(img_v, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return img_v

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=80)
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (key, (img_v, b, nq)) in enumerate(variants):
        small = downscale(img_v)
        axes[i].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        axes[i].set_title(
            f'{labels.get(key, key)}\nBRISQUE={b:.2f}  NIQE={nq:.2f}',
            fontsize=9)
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Ablation Study — Each Step Removed', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)  # Free memory after saving


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch color restoration for old images')
    parser.add_argument('--input-dir',  '-i', default=None,
                        help='Input folder (default: dataset/old_images)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output folder (default: results/restored_images)')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip saving comparison image')
    parser.add_argument('--ablation',   action='store_true',
                        help='Run ablation study for each image')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    input_dir  = args.input_dir  or os.path.join(base_dir, 'dataset', 'old_images')
    output_dir = args.output_dir or os.path.join(base_dir, 'results', 'restored_images')

    os.makedirs(input_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Input folder : %s', input_dir)
    logging.info('Output folder: %s', output_dir)

    process_all(input_dir, output_dir,
                display=(not args.no_display),
                run_ablation=args.ablation)