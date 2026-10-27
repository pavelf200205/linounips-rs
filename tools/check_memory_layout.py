"""CPU-only indexing checks; does not compile Rust or establish GPU parity.

Run: python -B tools/check_memory_layout.py
Checks mirror the tiled Gaussian address formulas and compact feature mapping.
Use `lino compare` after building to establish equality of actual GPU outputs.
"""
import random
import struct
import unittest


class MemoryLayoutTests(unittest.TestCase):
    def test_horizontal_halo_preserves_every_tap(self):
        for base in (0, 256):
            tile = [None] * 276
            for unit in range(256):
                shifted = base + unit
                if 10 <= shifted < 522:
                    tile[unit] = shifted - 10
                if unit < 20:
                    shifted = base + unit + 256
                    if 10 <= shifted < 522:
                        tile[unit + 256] = shifted - 10
            for unit in range(256):
                old = [base + unit + k - 10 for k in range(21)
                       if 0 <= base + unit + k - 10 < 512]
                new = [tile[unit + k] for k in range(21)
                       if 0 <= base + unit + k - 10 < 512]
                self.assertEqual(old, new)

    def test_shared_transpose_is_bijective_with_partial_tiles(self):
        for count in (1, 7, 31, 32, 33, 63, 65, 513):
            addresses = set()
            tiles = (count + 31) // 32
            for group in range(tiles * 8):
                pixel_base = (group % tiles) * 32
                channel_base = (group // tiles) * 32
                shared = {}
                for part in range(4):
                    for unit in range(256):
                        local = unit + part * 256
                        xx, cc = local % 32, local // 32
                        if pixel_base + xx < count:
                            shared[cc * 33 + xx] = (pixel_base + xx, channel_base + cc)
                for part in range(4):
                    for unit in range(256):
                        local = unit + part * 256
                        cc, xx = local % 32, local // 32
                        if pixel_base + xx < count:
                            pixel, channel = shared[cc * 33 + xx]
                            address = (pixel_base + xx) * 256 + channel_base + cc
                            self.assertEqual(address, pixel * 256 + channel)
                            self.assertNotIn(address, addresses)
                            addresses.add(address)
            self.assertEqual(len(addresses), count * 256)
            self.assertEqual(max(addresses), count * 256 - 1)

    def test_dense_tiles_cover_all_channel_pixel_tiles(self):
        coordinates = set()
        for group in range(65536):
            x = (group % 16) * 32
            y = (group // 16) % 512
            c = (group // 8192) * 32
            coordinates.add((x, y, c))
        self.assertEqual(len(coordinates), 16 * 512 * 8)
        self.assertIn((480, 511, 224), coordinates)

    def test_foreground_mapping_preserves_shuffle_and_native_bits(self):
        rng = random.Random(42)
        foreground = sorted(rng.sample(range(512 * 512), 2051))
        inverse = {pixel: row for row, pixel in enumerate(foreground)}
        # Arbitrary native words include signed zero and NaN payload encodings;
        # gather must copy bits, never round-trip through floating point.
        dense = {pixel: [rng.randrange(65536) for _ in range(256)] for pixel in foreground}
        compact = [dense[p] for p in foreground]
        ids = foreground.copy()
        rng.shuffle(ids)
        rng.shuffle(ids)
        groups = len(ids) // 2048 + 1
        base, extra = divmod(len(ids), groups)
        at = 0
        for group in range(groups):
            count = base + (group < extra)
            for pixel in ids[at:at + count]:
                self.assertEqual(dense[pixel], compact[inverse[pixel]])
            at += count
        self.assertEqual(at, len(ids))

    def test_fp32_indices_are_exact(self):
        for pixel in range(512 * 512):
            self.assertEqual(pixel, int(struct.unpack("f", struct.pack("f", pixel))[0]))


if __name__ == "__main__":
    unittest.main()
