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

This script reads all images from the input folder, applies a
restoration pipeline (see restoration.py), displays each original
and restored image side-by-side, and saves the restored output.
"""

import os
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
from restoration import restore_image, analyze_and_restore


def process_all(input_dir, output_dir, display=True):
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

            # Adaptive analysis (detect conditions). We'll produce a single 'mild' output.
            _, info = analyze_and_restore(img)

            mild_params = dict(
                nlm_h=6,
                median_k=3,
                clahe_clip=1.2,
                sat_scale_override=1.05,
                unsharp_amount=0.2,
                spot_thresh=50,
                spot_blur=9,
                spot_min_frac=1e-4,
                inpaint_radius=2,
            )

            mild_variant = restore_image(img, sat_scale=1.25, **mild_params)
            out_name = f'restored_{fname}'
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, mild_variant)
            logging.info('Saved restored image (mild) to: %s', out_path)

            # Recompute metrics comparing original and mild variant for accurate reporting
            try:
                orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mild_gray = cv2.cvtColor(mild_variant, cv2.COLOR_BGR2GRAY)
                # lazy import of helpers from restoration module
                from restoration import mse, psnr, ssim

                info['mse'] = mse(orig_gray, mild_gray)
                info['psnr'] = psnr(orig_gray, mild_gray)
                info['ssim'] = ssim(orig_gray, mild_gray)
            except Exception:
                logging.exception('Failed to compute metrics for mild variant')

            # Print detected condition and metrics
            cond = 'Noisy' if info.get('is_noisy') else 'Clean'
            if info.get('is_low_contrast'):
                cond += ' + Low-Contrast'
            if info.get('is_grayscale'):
                cond += ' + Grayscale'
            logging.info('Detected condition: %s', cond)
            logging.info('Noise level: %.2f, Contrast score: %.2f', info.get('noise_level'), info.get('contrast_score'))
            logging.info('MSE: %.2f, PSNR: %.2f, SSIM: %.4f', info.get('mse'), info.get('psnr'), info.get('ssim'))

            # Display original vs 'mild' restored for quick check (if requested)
            if display:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axes[0].set_title('Original Image')
                    mild_img = cv2.imread(out_path)
                    if mild_img is None:
                        logging.warning('Mild preset not found for display: %s', out_path)
                        axes[1].text(0.5, 0.5, 'Mild image missing', horizontalalignment='center', verticalalignment='center')
                    else:
                        axes[1].imshow(cv2.cvtColor(mild_img, cv2.COLOR_BGR2RGB))
                    axes[1].set_title('Restored (Mild)')
                    for ax in axes:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    logging.exception('Error displaying images for %s', fname)

        except Exception:
            logging.exception('Processing failed for %s', in_path)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch color restoration for old images')
    parser.add_argument('--input-dir', '-i', default=None, help='Input folder (default: dataset/old_images)')
    parser.add_argument('--output-dir', '-o', default=None, help='Output folder (default: results/restored_images)')
    parser.add_argument('--no-display', action='store_true', help='Do not show images interactively')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(base_dir, 'dataset', 'old_images')
    output_dir = args.output_dir or os.path.join(base_dir, 'results', 'restored_images')

    # Ensure folders exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Input folder: %s', input_dir)
    logging.info('Output folder: %s', output_dir)

    process_all(input_dir, output_dir, display=(not args.no_display))
