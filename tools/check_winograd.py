"""Independent matrix-algebra oracle for Winograd F(2x2,3x3), plus indexing.
CPU only; does not compile Rust or execute/validate GPU kernels or FP16 accuracy.
"""
import random
import unittest

G = ((1, 0, 0), (.5, .5, .5), (.5, -.5, .5), (0, 0, 1))
BT = ((1, 0, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, 0, -1))
AT = ((1, 1, 1, 0), (0, 1, -1, -1))


def transpose(a):
    return list(zip(*a))


def mul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in transpose(b)] for row in a]


def weights_formula(g):
    t = [[g[0][j], (g[0][j] + g[1][j] + g[2][j]) * .5,
          (g[0][j] - g[1][j] + g[2][j]) * .5, g[2][j]] for j in range(3)]
    t = transpose(t)
    return [[a, (a + b + c) * .5, (a - b + c) * .5, c] for a, b, c in t]


def input_formula(d):
    t = transpose([[d[0][j] - d[2][j], d[1][j] + d[2][j],
                    d[2][j] - d[1][j], d[1][j] - d[3][j]] for j in range(4)])
    return [[a - c, b + c, c - b, b - dd] for a, b, c, dd in t]


def inverse_formula(m):
    t = transpose([[m[0][j] + m[1][j] + m[2][j], m[1][j] - m[2][j] - m[3][j]] for j in range(4)])
    return [[a + b + c, b - c - d] for a, b, c, d in t]


class WinogradTests(unittest.TestCase):
    def assert_matrix_close(self, a, b):
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                self.assertAlmostEqual(x, y, places=11)

    def test_expanded_transform_matches_matrix_definition(self):
        rng = random.Random(12)
        for _ in range(20):
            d = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(4)]
            g = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(3)]
            m = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(4)]
            self.assert_matrix_close(weights_formula(g), mul(mul(G, g), transpose(G)))
            self.assert_matrix_close(input_formula(d), mul(mul(BT, d), transpose(BT)))
            self.assert_matrix_close(inverse_formula(m), mul(mul(AT, m), transpose(AT)))

    def test_full_convolution_with_padding_channels_batches_and_odd_tails(self):
        rng = random.Random(19)
        for batch, channels, outputs, h, w, chunk in ((2, 3, 2, 5, 7, 4), (1, 2, 3, 1, 1, 8), (2, 1, 1, 4, 6, 8)):
            x = [[[[rng.uniform(-1, 1) for _ in range(w)] for _ in range(h)] for _ in range(channels)] for _ in range(batch)]
            weights = [[[[rng.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(channels)] for _ in range(outputs)]
            u = [[weights_formula(weights[oc][ic]) for ic in range(channels)] for oc in range(outputs)]
            th, tw = (h + 1) // 2, (w + 1) // 2
            def sample(b, c, y, xx):
                return x[b][c][y][xx] if 0 <= b < batch and 0 <= y < h and 0 <= xx < w else 0
            seen = set()
            for start in range(0, batch * th * tw, chunk):
                for column in range(min(chunk, batch * th * tw - start)):
                    tile = start + column
                    b, ty, tx = tile // (th * tw), (tile // tw) % th, tile % tw
                    v = [input_formula([[sample(b, ic, ty * 2 + yy - 1, tx * 2 + xx - 1) for xx in range(4)] for yy in range(4)]) for ic in range(channels)]
                    for oc in range(outputs):
                        m = [[sum(u[oc][ic][yy][xx] * v[ic][yy][xx] for ic in range(channels)) for xx in range(4)] for yy in range(4)]
                        y = inverse_formula(m)
                        for yy in range(2):
                            for xx in range(2):
                                oy, ox = ty * 2 + yy, tx * 2 + xx
                                if oy < h and ox < w:
                                    direct = sum(sample(b, ic, oy + ky - 1, ox + kx - 1) * weights[oc][ic][ky][kx] for ic in range(channels) for ky in range(3) for kx in range(3))
                                    self.assertAlmostEqual(y[yy][xx], direct, places=10)
                                    address = ((b * outputs + oc) * h + oy) * w + ox
                                    self.assertNotIn(address, seen)
                                    seen.add(address)
            self.assertEqual(len(seen), batch * outputs * h * w)

    def test_direct_store_tail_layout(self):
        for batches, channels, h, w, rows in ((2, 3, 5, 7, 8), (1, 2, 1, 1, 16)):
            seen = set()
            for b in range(batches):
                for start in range(0, h * w, rows):
                    count = min(rows, h * w - start)
                    for i in range(channels * count):
                        ch, pixel = divmod(i, count)
                        address = (b * channels + ch) * h * w + start + pixel
                        self.assertNotIn(address, seen)
                        seen.add(address)
            self.assertEqual(seen, set(range(batches * channels * h * w)))


if __name__ == '__main__':
    unittest.main()
