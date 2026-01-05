import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scipy.fft import dctn, idctn


Q_LUMA = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,28,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float64)

Q_CHROMA = np.clip(Q_LUMA * 1.5, 1, 255)


def pad8(x):
    h, w = x.shape[:2]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if x.ndim == 2:
        return np.pad(x, ((0,ph),(0,pw)), mode="edge"), (h,w)
    return np.pad(x, ((0,ph),(0,pw),(0,0)), mode="edge"), (h,w)

def unpad(x, hw):
    h, w = hw
    return x[:h, :w]


def jpeg_block(block, Q, qscale):
    block = block.astype(np.float64) - 128
    Y = dctn(block, norm="ortho")
    Qs = np.clip(Q * qscale, 1, 255)
    Yq = np.round(Y / Qs)
    rec = idctn(Yq * Qs, norm="ortho") + 128
    return rec, Yq


def jpeg_gray(img, qscale=1.0, Q=Q_LUMA):
    x = img.astype(np.float64)
    xpad, hw = pad8(x)
    h, w = xpad.shape
    out = np.zeros_like(xpad)

    nnz, total = 0, 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            rec, Yq = jpeg_block(xpad[i:i+8, j:j+8], Q, qscale)
            out[i:i+8, j:j+8] = rec
            nnz += np.count_nonzero(Yq)
            total += Yq.size

    out = np.clip(out, 0, 255)
    return unpad(out, hw).astype(np.uint8), {
        "qscale": qscale,
        "coeff_total": total,
        "coeff_nnz": nnz,
        "sparsity": 1 - nnz / total
    }


def rgb_to_ycbcr(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
    Cr =  0.5*R - 0.418688*G - 0.081312*B + 128
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    Y = Y.astype(np.float64)
    Cb = Cb.astype(np.float64) - 128
    Cr = Cr.astype(np.float64) - 128
    R = Y + 1.402*Cr
    G = Y - 0.344136*Cb - 0.714136*Cr
    B = Y + 1.772*Cb
    return np.clip(np.stack([R,G,B], axis=-1), 0, 255).astype(np.uint8)


def jpeg_color(rgb, qscale=1.0):
    Y, Cb, Cr = rgb_to_ycbcr(rgb)
    Yr,_ = jpeg_gray(Y, qscale, Q_LUMA)
    Cbr,_ = jpeg_gray(Cb, qscale, Q_CHROMA)
    Crr,_ = jpeg_gray(Cr, qscale, Q_CHROMA)
    return ycbcr_to_rgb(Yr, Cbr, Crr)


def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64))**2)

def jpeg_to_mse(img, target):
    lo, hi = 0.1, 10.0
    best = None
    for _ in range(20):
        mid = (lo + hi) / 2
        rec,_ = jpeg_gray(img, mid)
        e = mse(img, rec)
        if e <= target:
            best = (rec, mid, e)
            lo = mid
        else:
            hi = mid
    return best


def jpeg_video_frames(input_mp4, output_dir, qscale=3.0):
    import imageio.v2 as imageio
    import os

    os.makedirs(output_dir, exist_ok=True)

    reader = imageio.get_reader(input_mp4, format="ffmpeg")

    for i, frame in enumerate(reader):
        frame = frame.astype(np.uint8)
        comp = jpeg_color(frame, qscale)
        out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        imageio.imwrite(out_path, comp)

    reader.close()
    print("✔ Cadre JPEG salvate în:", output_dir)


if __name__ == "__main__":
    from scipy.datasets import ascent

    X = ascent().astype(np.uint8)

    Xc, stats = jpeg_gray(X, qscale=2.0)
    print("S1 stats:", stats)

    rec, q, e = jpeg_to_mse(X, 30)
    print("S3:", {"qscale": q, "mse": e})

    jpeg_video_frames(
        r"C:\Users\Antoniu\Desktop\input.mp4",
        r"C:\Users\Antoniu\Desktop\frames_jpeg",
        qscale=3.0
    )
