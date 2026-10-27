"""CPU oracles for packed decoder layout, residual, head packing and padding.
Does not compile Rust or execute GPU kernels, subgroup reductions or low-precision GEMM.
"""
import math
import random
import unittest


def attention_reference(q, k, v, scale):
    scores = [sum(x * y for x, y in zip(q, row)) * scale for row in k]
    maximum = max(scores)
    p = [math.exp(x - maximum) for x in scores]
    total = sum(p)
    return [q[c] + sum(prob * row[c] for prob, row in zip(p, v)) / total for c in range(len(q))]


class DecoderLayoutTests(unittest.TestCase):
    def test_packed_attention_and_q_residual(self):
        rng = random.Random(71)
        for batches, queries, keys, heads, width in ((2, 9, 9, 8, 32), (2, 1, 9, 8, 48), (1, 3, 5, 4, 8)):
            c = heads * width
            q = [rng.uniform(-1, 1) for _ in range(batches * queries * c)]
            kv = [rng.uniform(-1, 1) for _ in range(batches * keys * 2 * c)]
            scale = 1 / math.sqrt(c)
            written = set()
            for row in range(batches * heads * queries):
                query, head, batch = row % queries, (row // queries) % heads, row // (queries * heads)
                qb = (batch * queries + query) * c + head * width
                qq = q[qb:qb + width]
                kk, vv = [], []
                maximum, total = -float('inf'), 0.0
                accum = [0.0] * width
                for key in range(keys):
                    base = (batch * keys + key) * 2 * c + head * width
                    k, v = kv[base:base + width], kv[base + c:base + c + width]
                    kk.append(k)
                    vv.append(v)
                    score = sum((x * scale) * y for x, y in zip(qq, k))
                    next_max = max(maximum, score)
                    correction, probability = math.exp(maximum - next_max), math.exp(score - next_max)
                    total = total * correction + probability
                    accum = [a * correction + probability * x for a, x in zip(accum, v)]
                    maximum = next_max
                expected = attention_reference(qq, kk, vv, scale)
                for ch in range(width):
                    actual = qq[ch] + accum[ch] / total
                    self.assertAlmostEqual(actual, expected[ch], places=12)
                    dst = ((batch * queries + query) * heads + head) * width + ch
                    self.assertNotIn(dst, written)
                    written.add(dst)
            self.assertEqual(written, set(range(batches * queries * c)))

    def test_broadcast_seed_strides(self):
        # Q.expand(B,1,C) has stride zero in B; K/V still select the actual batch.
        channels = 384
        for batch in range(3):
            for head in range(8):
                for ch in range(48):
                    address = batch * 0 + head * 48 + ch
                    self.assertEqual(address, head * 48 + ch)
                    self.assertLess(address, channels)

    def test_packed_pbr_head_projection(self):
        rng = random.Random(86)
        dims = (3, 3, 1, 1)
        weights = [[[rng.uniform(-1, 1) for _ in range(d)] for _ in range(192)] for d in dims]
        bias = [[rng.uniform(-1, 1) for _ in range(d)] for d in dims]
        packed_w = [sum((weights[h][c] for h in range(4)), []) for c in range(192)]
        packed_b = sum(bias, [])
        for _ in range(5):
            hidden = [[rng.uniform(0, 1) for _ in range(192)] for _ in range(4)]
            flat = sum(hidden, [])
            expected = [sum(hidden[h][c] * weights[h][c][k] for c in range(192)) + bias[h][k] for h, d in enumerate(dims) for k in range(d)]
            actual = []
            for k in range(8):
                head = 0 if k < 3 else 1 if k < 6 else 2 if k == 6 else 3
                actual.append(sum(flat[head * 192 + c] * packed_w[c][k] for c in range(192)) + packed_b[k])
            self.assertEqual(expected, actual)
            length = max(math.sqrt(sum(x * x for x in actual[:3])), 1e-12)
            normalized = [x / length for x in actual[:3]] + actual[3:]
            self.assertAlmostEqual(sum(x * x for x in normalized[:3]), 1, places=12)
            self.assertEqual(normalized[3:], expected[3:])

    def test_host_padding_retains_valid_group_and_zero_tail(self):
        for count in (1, 63, 64, 65, 2031, 2032, 2048):
            padded = (count + 63) // 64 * 64
            for lights, channels in ((1, 3), (9, 3), (9, 256)):
                original = list(range(count * lights * channels))
                uploaded = original + [0] * ((padded - count) * lights * channels)
                self.assertEqual(uploaded[:len(original)], original)
                self.assertEqual(len(uploaded), padded * lights * channels)
                self.assertFalse(any(uploaded[len(original):]))
                self.assertEqual(len(uploaded) // (lights * channels), padded)


if __name__ == '__main__':
    unittest.main()
