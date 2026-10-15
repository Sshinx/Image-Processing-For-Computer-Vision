"""
Usage:
  python Image_Processing.py              (synthetic test image) 
  python Image_Processing.py photo.jpg    (Image Input)

"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from module.colors import tag


# Charging Images

def load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def make_test_image(h=240, w=320, noise=0.04, seed=0):
    """Gradient sky + plane silhouette + Gaussian noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 0.75 + 0.15 * (yy / h) # sky: soft vertical gradient

    cx, cy = w / 2, h / 2
    # fuselage: elongated ellipse
    fuselage = ((xx - cx) / 110) ** 2 + ((yy - cy) / 12) ** 2 <= 1
    # wings: swept triangle on either side of the fuselage
    wings = (np.abs(yy - cy) <= 0.9 * (xx - (cx - 40))) & (xx <= cx + 20) & (np.abs(yy - cy) <= 85)
    # rear tail
    tail = (np.abs(yy - cy) <= 0.8 * (xx - (cx - 115))) & (xx <= cx - 80) & (np.abs(yy - cy) <= 32)
    plane = fuselage | wings | tail
    img[plane] = 0.25

    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


# Convolution Functions

def _pad(img, k):
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    return np.pad(img, ((ph, ph), (pw, pw)), mode="symmetric")   # mirrored borders


def conv2d_naive(img, k):
    """True convolution (flipped kernel), output the same size as the image."""
    k = k[::-1, ::-1]
    kh, kw = k.shape
    padded = _pad(img, k)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * k)
    return out


def conv2d(img, k):
    """Vectorized convolution: all windows at once."""
    k = k[::-1, ::-1]
    windows = sliding_window_view(_pad(img, k), k.shape) # (H, W, kh, kw)
    return np.einsum("ijkl,kl->ij", windows, k)


# Blurs

def box_kernel(size):
    return np.ones((size, size)) / size ** 2

def gaussian_1d(sigma):
    r = max(1, int(np.ceil(3 * sigma))) # ±3σ ≈ 99.7 % of the mass
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum() # sum = 1 brightness preserved

def gaussian_kernel(sigma):
    g = gaussian_1d(sigma)
    return np.outer(g, g)

def gaussian_blur_separable(img, sigma):
    """Two 1D convolutions (rows then columns) instead of one 2D."""
    g = gaussian_1d(sigma)
    tmp = conv2d(img, g[np.newaxis, :]) # noyau 1 x k
    return conv2d(tmp, g[:, np.newaxis]) # noyau k x 1

# Sobel Kernel

KX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float64)
KY = KX.T

def sobel(img):
    # With a true convolution (flipped kernel), we get -dI/dx:
    # we fix the sign so the gradient points the right way.
    gx = -conv2d(img, KX) # variation along columns (x to the right)
    gy = -conv2d(img, KY) # variation along rows (y downwards)
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    return gx, gy, mag, theta

# Canny

def non_max_suppression(mag, theta):
    ang = np.rad2deg(theta) % 180
    q = (np.round(ang / 45) % 4).astype(int) # 0°, 45°, 90°, 135°
    offsets = [(0, 1), (1, 1), (1, 0), (1, -1)] # (Δrow, Δcolumn)
    out = np.zeros_like(mag)
    H, W = mag.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            di, dj = offsets[q[i, j]]
            m = mag[i, j]
            if m >= mag[i + di, j + dj] and m >= mag[i - di, j - dj]:
                out[i, j] = m
    return out


def hysteresis(nms, low, high):
    """Double threshold + propagation from strong edges to weak ones (DFS)."""
    strong = nms >= high
    weak = (nms >= low) & ~strong
    edges = strong.copy()
    H, W = nms.shape
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and weak[ni, nj] and not edges[ni, nj]:
                    edges[ni, nj] = True
                    stack.append((ni, nj))
    return edges


def canny(img, sigma=1.4, low_ratio=0.1, high_ratio=0.3):
    blurred = gaussian_blur_separable(img, sigma)
    gx, gy, mag, theta = sobel(blurred)
    thin = non_max_suppression(mag, theta)
    high = high_ratio * thin.max()
    low = low_ratio * thin.max()
    edges = hysteresis(thin, low, high)
    return edges, {"blurred": blurred, "gx": gx, "gy": gy, "mag": mag, "nms": thin}

# Test Functions

def run_tests(img):
    print("-( Tests de vérification )-")
    from scipy.signal import convolve2d

    # 1. Convolution vs SciPy (on a crop so the naive version stays fast)
    crop = img[:60, :80]
    k = np.random.default_rng(1).normal(size=(5, 5))
    ref = convolve2d(crop, k, mode="same", boundary="symm")
    err_naive = np.abs(conv2d_naive(crop, k) - ref).max()
    err_vec = np.abs(conv2d(crop, k) - ref).max()
    print(f"{tag('conv')}  écart max naïve / SciPy     : {err_naive:.2e}")
    print(f"{tag('conv')}  écart max vectorisée / SciPy: {err_vec:.2e}")

    t0 = time.perf_counter(); conv2d_naive(img, k); t1 = time.perf_counter()
    conv2d(img, k); t2 = time.perf_counter()
    print(f"{tag('conv')}  temps naïve {t1 - t0:.3f}s  vs  vectorisée {t2 - t1:.4f}s")

    # 2. Gaussian: sum = 1 and separability
    sigma = 2.0
    G = gaussian_kernel(sigma)
    print(f"{tag('gauss')} somme du noyau = {G.sum():.6f}, taille {G.shape}")
    err_sep = np.abs(conv2d(img, G) - gaussian_blur_separable(img, sigma)).max()
    print(f"{tag('gauss')} écart 2D vs séparable : {err_sep:.2e}")

    # 3. Sobel: on a ramp I = x, the gradient must point towards +x
    ramp = np.tile(np.arange(20, dtype=float), (20, 1))
    gx, gy, _, _ = sobel(ramp)
    print(f"{tag('sobel')} rampe en x : gx centre = {gx[10, 10]:.1f} (attendu 8), gy = {gy[10, 10]:.1f} (attendu 0)")

    # 4. Canny vs OpenCV: agreement rate of edge pixels
    try:
        import cv2
        edges, _ = canny(img)
        u8 = (img * 255).astype(np.uint8)
        u8 = cv2.GaussianBlur(u8, (0, 0), 1.4)
        ref = cv2.Canny(u8, 30, 90, L2gradient=True) > 0
        # 1-pixel tolerance (dilation) because conventions differ slightly
        ref_d = cv2.dilate(ref.astype(np.uint8), np.ones((3, 3))) > 0
        ours_d = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3))) > 0
        precision = (edges & ref_d).sum() / max(edges.sum(), 1)
        recall = (ref & ours_d).sum() / max(ref.sum(), 1)
        print(f"{tag('canny')} accord avec OpenCV (±1 px) : précision {precision:.0%}, rappel {recall:.0%}")
        return ref
    except ImportError:
        print(f"{tag('canny')} OpenCV absent, comparaison sautée")
        return None

def show(img, cv_ref=None, out="resultats.png"):
    edges, s = canny(img)
    panels = [
        (img, "Image d'origine"),
        (conv2d(img, box_kernel(9)), "Flou moyenneur 9x9"),
        (s["blurred"], "Flou gaussien σ=1.4"),
        (s["gx"], "Sobel gx (contours verticaux)"),
        (s["gy"], "Sobel gy (contours horizontaux)"),
        (s["mag"], "Norme du gradient"),
        (s["nms"], "Après suppression des non-maxima"),
        (edges, "Canny (from scratch)"),
    ]
    if cv_ref is not None:
        panels.append((cv_ref, "Canny OpenCV (référence)"))
    # Sobel without blur to see the effect of noise
    panels.append((sobel(img)[2], "Sobel SANS flou (bruit)"))

    n = len(panels)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (im, title) in zip(axes.flat, panels):
        if title.startswith("Sobel g"):            # sign: red/blue centered on 0
            v = np.abs(im).max()
            ax.imshow(im, cmap="RdBu", vmin=-v, vmax=v)
        else:
            ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"Figure enregistrée : {out}")

if __name__ == "__main__":
    image = load_gray(sys.argv[1]) if len(sys.argv) > 1 else make_test_image()
    ref = run_tests(image)
    show(image, ref)
