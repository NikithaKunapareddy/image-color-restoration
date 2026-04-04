import argparse
import cv2
import numpy as np
from restoration import detect_fold_lines, save_debug_overlays

parser = argparse.ArgumentParser()
parser.add_argument('--file','-f', required=True)
parser.add_argument('--out','-o', default='results/debug_test')
parser.add_argument('--mode', choices=['default','relaxed','both'], default='default',
                    help='Which detection mode to run and save')
args = parser.parse_args()

img = cv2.imread(args.file)
if img is None:
    print('Cannot read:', args.file)
    raise SystemExit(1)

h, w = img.shape[:2]
print('Image size:', w, 'x', h)

# Raw Hough with defaults from detect_fold_lines
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 180)
raw = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=180, minLineLength=200, maxLineGap=10)
print('Raw Hough lines (default params):', 0 if raw is None else len(raw))

# Call detect_fold_lines (default)
folds_default = detect_fold_lines(img)
print('Filtered fold_lines (default):', len(folds_default))
for i, fl in enumerate(folds_default):
    print(f'  {i}: conf={fl.get("confidence"):.3f} pts=({fl["x1"]},{fl["y1"]})-({fl["x2"]},{fl["y2"]})' if isinstance(fl, dict) else fl)

if args.mode in ('default','both'):
    save_debug_overlays(img, folds_default, None, args.out + '_default')

# Try relaxed params
canny_low, canny_high = 50, 150
hough_thresh = 60
min_len = 50
max_gap = 20

edges2 = cv2.Canny(gray, canny_low, canny_high)
raw2 = cv2.HoughLinesP(edges2, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_len, maxLineGap=max_gap)
print('Raw Hough lines (relaxed params):', 0 if raw2 is None else len(raw2))

from restoration import detect_fold_lines as detect_fold_lines_fn
folds_relaxed = detect_fold_lines_fn(img, canny_low=canny_low, canny_high=canny_high,
                                    hough_thresh=hough_thresh, min_line_length=min_len, max_line_gap=max_gap)
print('Filtered fold_lines (relaxed):', len(folds_relaxed))
for i, fl in enumerate(folds_relaxed):
    if isinstance(fl, dict):
        print(f'  {i}: conf={fl.get("confidence"):.3f} pts=({fl["x1"]},{fl["y1"]})-({fl["x2"]},{fl["y2"]})')
    else:
        print(' ', fl)

if args.mode in ('relaxed','both'):
    save_debug_overlays(img, folds_relaxed, None, args.out + '_relaxed')

if args.mode == 'both':
    print('Saved overlays to:', args.out + '_default.png and', args.out + '_relaxed.png')
elif args.mode == 'default':
    print('Saved overlay to:', args.out + '_default.png')
elif args.mode == 'relaxed':
    print('Saved overlay to:', args.out + '_relaxed.png')
