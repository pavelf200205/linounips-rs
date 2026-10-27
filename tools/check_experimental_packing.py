"""CPU oracle for the experimental packer's OIHW/NCHW reduction and row layout.
Does not execute Rust, CubeCL, GEMM, or establish GPU output parity.
"""
import random
import unittest


def resized(x, b, c, y, xx, height, width):
    ih, iw = len(x[0][0]), len(x[0][0][0])
    fy, fx = y * (ih - 1) / (height - 1), xx * (iw - 1) / (width - 1)
    y0, x0 = int(fy), int(fx)
    y1, x1 = min(y0 + 1, ih - 1), min(x0 + 1, iw - 1)
    ay, ax = fy - y0, fx - x0
    a, bb = x[b][c][y0][x0], x[b][c][y0][x1]
    cc, d = x[b][c][y1][x0], x[b][c][y1][x1]
    top, bottom = a + (bb - a) * ax, cc + (d - cc) * ax
    return top + (bottom - top) * ay


class PackingTests(unittest.TestCase):
    def test_packing_matches_direct_convolution(self):
        rng = random.Random(43)
        for upsample in (False, True):
            for batch, channels, h, w, kernel, stride, pad, chunk in (
                    (2, 3, 5, 7, 3, 1, 1, 8), (1, 2, 4, 6, 3, 2, 1, 16),
                    (2, 1, 3, 3, 2, 2, 0, 4), (1, 3, 8, 8, 8, 8, 0, 1)):
                x = [[[[rng.random() for _ in range(w)] for _ in range(h)]
                      for _ in range(channels)] for _ in range(batch)]
                hh, ww = (2 * h, 2 * w) if upsample else (h, w)
                oh, ow = (hh + 2 * pad - kernel) // stride + 1, (ww + 2 * pad - kernel) // stride + 1
                weights = [rng.random() for _ in range(channels * kernel * kernel)]
                def sample(b, c, y, xx):
                    if not (0 <= b < batch and 0 <= y < hh and 0 <= xx < ww):
                        return 0.0
                    return resized(x, b, c, y, xx, hh, ww) if upsample else x[b][c][y][xx]
                for bi in range(batch):
                    for start in range(0, oh * ow, chunk):
                        packed = []
                        for reduction in range(channels * kernel * kernel):
                            c, ky, kx = reduction // (kernel * kernel), (reduction // kernel) % kernel, reduction % kernel
                            for column in range(chunk):
                                row = bi * oh * ow + start + column
                                b = row // (oh * ow)
                                y, xx = (row // ow) % oh * stride + ky - pad, row % ow * stride + kx - pad
                                packed.append(sample(b, c, y, xx))
                        for column in range(min(chunk, oh * ow - start)):
                            y, xx = divmod(start + column, ow)
                            direct = []
                            for c in range(channels):
                                for ky in range(kernel):
                                    for kx in range(kernel):
                                        direct.append(sample(bi, c, y * stride + ky - pad, xx * stride + kx - pad))
                            gathered = [packed[k * chunk + column] for k in range(len(weights))]
                            self.assertEqual(direct, gathered)
                            self.assertEqual(sum(a * b for a, b in zip(weights, direct)),
                                             sum(a * b for a, b in zip(weights, gathered)))


if __name__ == '__main__':
    unittest.main()
