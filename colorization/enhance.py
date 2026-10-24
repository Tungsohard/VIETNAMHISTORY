"""Pre- and post-processing around the colorization network.

The pipeline is

1. ``restore_luminance``: denoise, de-haze and sharpen the scanned L channel so
   the network sees a cleaner input; the restored L is also the output L.
2. ``refine_ab``: upsample the low-resolution ab prediction with edge-aware
   filtering guided by the full-resolution L, then make colour uniform inside
   SLIC superpixels whose borders follow object edges. Colour therefore stops
   at the boundary between a person, an object and the sky instead of bleeding.
3. ``modernize_ab``: remove the global colour cast (sepia/olive) estimated from
   near-neutral pixels and lift chroma towards that of modern photographs.

Every step is global or edge-driven; nothing is keyed to detected faces or to
hand-picked colour ranges, so it cannot paint rectangular patches.
"""
import cv2
import numpy as np
from scipy import sparse
from skimage.segmentation import slic


def estimate_noise(luminance):
    """Robust noise sigma (L units) from the median absolute Laplacian response."""
    laplacian = cv2.Laplacian(luminance.astype(np.float32), cv2.CV_32F)
    return float(np.median(np.abs(laplacian)) / 0.6745 / np.sqrt(20.0))


def restore_luminance(luminance, strength=1.0):
    """Return a cleaner L (0..100): adaptive denoise, contrast recovery, sharpening."""
    if strength <= 0.0:
        return luminance
    L = luminance.astype(np.float32)
    sigma = estimate_noise(L)
    if sigma > 0.6:
        h = float(np.clip(sigma * 2.55 * 1.2 * strength, 3.0, 18.0))
        denoised = cv2.fastNlMeansDenoising(np.clip(L * 2.55, 0, 255).astype(np.uint8), None, h=h,
                                            templateWindowSize=7, searchWindowSize=21)
        L = denoised.astype(np.float32) / 2.55

    # Faded scans: lift the black/white points, but keep shadow detail (hair, uniforms).
    low, high = np.percentile(L, (1.0, 99.5))
    if high - low > 10.0:
        stretched = np.clip((L - low) / (high - low) * 100.0, 0.0, 100.0)
        shadow_guard = np.clip(L / 25.0, 0.0, 1.0)
        L = L + 0.45 * strength * shadow_guard * (stretched - L)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    local = clahe.apply(np.clip(L * 2.55, 0, 255).astype(np.uint8)).astype(np.float32) / 2.55
    L = L + 0.25 * strength * (local - L)

    scale = max(L.shape) / 1000.0
    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=max(1.0, 1.2 * scale))
    L = L + 0.45 * strength * (L - blur)
    return np.clip(L, 0.0, 100.0).astype(np.float32)


def guided_filter(guide, src, radius, eps):
    """He et al. guided filter; ``guide`` in 0..1, ``src`` single channel."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, size)
    mean_p = cv2.boxFilter(src, -1, size)
    cov_ip = cv2.boxFilter(guide * src, -1, size) - mean_i * mean_p
    var_i = cv2.boxFilter(guide * guide, -1, size) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, size) * guide + cv2.boxFilter(b, -1, size)


def _segment_sums(labels, values, weights, count):
    return np.stack([np.bincount(labels, weights=values[:, c] * weights, minlength=count)
                     for c in range(values.shape[1])], axis=1)


def _segment_graph(labels, segment_l, count, sigma_l, segment_object=None):
    """Sparse adjacency between touching superpixels, weighted by luminance similarity.

    With ``segment_object`` (object id per superpixel, -1 = not covered), superpixels
    of two different objects are not connected, so colour never crosses an object border.
    """
    pairs = np.concatenate([
        np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], axis=1),
        np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()], axis=1),
    ])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    weight = np.exp(-((segment_l[pairs[:, 0]] - segment_l[pairs[:, 1]]) / sigma_l) ** 2)
    if segment_object is not None:
        first, second = segment_object[pairs[:, 0]], segment_object[pairs[:, 1]]
        weight = weight * ~((first >= 0) & (second >= 0) & (first != second))
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sparse.csr_matrix((np.concatenate([weight, weight]), (rows, cols)), shape=(count, count))


def refine_ab(luminance, ab_low, region_weight=0.65, sigma_l=6.0, iterations=20, objects=None):
    """Upsample ab (Lab units) to ``luminance`` without colour bleeding across object edges.

    A low-resolution prediction smears colour over a band around every edge whose
    width grows with the upsampling factor. Pixels inside that band are treated as
    unreliable; each superpixel takes its colour from its own reliable pixels, and
    superpixels that have none (they hug an edge) borrow colour only from
    neighbouring superpixels of similar brightness, never from across the edge.

    ``objects`` (object id per pixel from ``segmentation.ObjectSegmenter``, -1 where
    uncovered) adds borders that luminance alone cannot see, e.g. grey hair against a
    pale sky: superpixels are split along them and no colour is borrowed across them.
    """
    height, width = luminance.shape
    guide = (luminance / 100.0).astype(np.float32)
    ab = cv2.resize(ab_low.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if region_weight <= 0.0:
        return ab

    # Work on a copy capped at ~1 MP so SLIC and the graph stay fast.
    scale = min(1.0, 1000.0 / max(height, width))
    work_l = cv2.resize(luminance.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    work_ab = cv2.resize(ab, (work_l.shape[1], work_l.shape[0]), interpolation=cv2.INTER_AREA)
    labels = slic(work_l / 100.0, n_segments=max(200, int(work_l.size / 300)), compactness=0.08,
                  channel_axis=None, start_label=0, enforce_connectivity=True)
    work_objects = None
    if objects is not None:
        work_objects = cv2.resize(objects.astype(np.int32), (work_l.shape[1], work_l.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        # Split every superpixel that straddles an object border.
        _, labels = np.unique(labels.astype(np.int64) * (int(work_objects.max()) + 2) + work_objects + 1,
                              return_inverse=True)
        labels = labels.reshape(work_l.shape)
    count = int(labels.max()) + 1

    bleed = max(2.0, 2.5 * max(work_l.shape) / max(ab_low.shape[:2]))
    work_u8 = np.clip(work_l * 2.55, 0, 255).astype(np.uint8)
    edges = cv2.Canny(work_u8, 30, 90)
    if work_objects is not None:
        # Inside a known object, texture (wrinkles, foliage, cloth folds) is not a border:
        # keep only strong edges there and let SAM supply the object borders.
        strong = cv2.Canny(work_u8, 80, 200)
        edges = np.where(work_objects >= 0, strong, edges)
        border = np.zeros(work_l.shape, dtype=bool)
        border[:, 1:] |= work_objects[:, 1:] != work_objects[:, :-1]
        border[1:, :] |= work_objects[1:, :] != work_objects[:-1, :]
        edges = np.where(border, 255, edges).astype(np.uint8)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    confidence = np.clip(distance / bleed, 0.0, 1.0) ** 2 + 1e-4

    flat = labels.ravel()
    pixel_count = np.maximum(np.bincount(flat, minlength=count), 1)
    segment_l = np.bincount(flat, weights=work_l.ravel(), minlength=count) / pixel_count
    anchor = np.bincount(flat, weights=confidence.ravel(), minlength=count)
    anchor_ab = _segment_sums(flat, work_ab.reshape(-1, 2), confidence.ravel(), count) / anchor[:, None]
    anchor = anchor / pixel_count

    segment_object = None
    if work_objects is not None:
        segment_object = np.full(count, -1, dtype=np.int64)
        segment_object[flat] = work_objects.ravel()
    graph = _segment_graph(labels, segment_l, count, sigma_l, segment_object)
    degree = np.asarray(graph.sum(axis=1)).ravel() + 1e-6
    values, trust = anchor_ab.copy(), anchor.copy()
    for _ in range(iterations):
        numerator = anchor[:, None] * anchor_ab + graph @ (trust[:, None] * values)
        denominator = anchor + graph @ trust
        values = numerator / np.maximum(denominator, 1e-6)[:, None]
        trust = np.maximum(anchor, 0.9 * (graph @ trust) / degree)

    region_ab = values[labels]
    # Unreliable (near-edge) pixels take the region colour entirely; reliable ones keep texture.
    mix = np.maximum(region_weight, 1.0 - confidence)[..., None]
    work_result = work_ab + mix * (region_ab - work_ab)
    result = cv2.resize(work_result.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    # Snap the work-resolution region borders onto the full-resolution edges.
    radius = max(1, round(1.0 / scale))
    return np.stack([guided_filter(guide, result[..., c], radius, 1e-4) for c in range(2)], axis=-1)


def modernize_ab(luminance, ab, vividness=0.5, target_chroma=17.0):
    """Remove global cast and lift chroma towards modern-photo statistics (Lab units)."""
    if vividness <= 0.0:
        return ab
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    midtones = (luminance > 15.0) & (luminance < 90.0)
    pool = chroma[midtones] if midtones.any() else chroma.ravel()
    neutral = midtones & (chroma <= np.percentile(pool, 35))
    if neutral.sum() > 64:
        # Pixels the model already sees as nearly grey should be grey; their mean is the cast.
        # Only the part above a small tolerance is a cast, and it is removed from near-neutral
        # areas (walls, sky, white cloth) so genuinely coloured regions keep their chroma.
        cast = ab[neutral].mean(axis=0)
        magnitude = float(np.hypot(*cast))
        excess = cast * max(0.0, 1.0 - 3.0 / max(magnitude, 1e-6))
        neutral_weight = np.clip((25.0 - chroma) / 15.0, 0.0, 1.0)
        removal = np.clip(vividness * 1.6, 0.0, 1.0) * neutral_weight * float(np.hypot(*excess))
        # Remove the cast only as far as grey: a pixel that is already grey (white hair,
        # deep shadow) must not be pushed past neutral into the opposite, blue, hue.
        direction = excess / max(float(np.hypot(*excess)), 1e-6)
        along = ab[..., 0] * direction[0] + ab[..., 1] * direction[1]
        removal = np.minimum(removal, np.maximum(along, 0.0))
        ab = ab - removal[..., None] * direction
        chroma = np.hypot(ab[..., 0], ab[..., 1])

    mean_chroma = float(chroma[midtones].mean()) if midtones.any() else float(chroma.mean())
    # Region averaging in refine_ab costs a few percent of chroma; always give a little back.
    gain = np.clip(target_chroma / max(mean_chroma, 1e-3), 1.0 + 0.1 * vividness, 1.0 + 0.9 * vividness)
    # Vibrance: weak colours gain more than already saturated ones.
    local_gain = 1.0 + (gain - 1.0) * (1.0 - np.clip(chroma / 40.0, 0.0, 1.0))
    taper = np.clip((luminance - 4.0) / 14.0, 0.0, 1.0) * np.clip((99.0 - luminance) / 9.0, 0.0, 1.0)
    factor = 1.0 + (local_gain - 1.0) * taper
    return ab * factor[..., None]


MODES = {
    # name: (restore, region_weight, vividness)
    "modern": (1.0, 0.65, 0.6),
    "vivid": (1.0, 0.65, 1.0),
    "natural": (0.0, 0.5, 0.0),
}
