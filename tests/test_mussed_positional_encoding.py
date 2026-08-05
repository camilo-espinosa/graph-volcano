import unittest

import torch
import torch.nn.functional as F

from models.MuSSED import SinusoidalPositionalEncoding


class TestMuSSEDPositionalEncoding(unittest.TestCase):
    def test_stride_aware_alignment_between_levels(self):
        d_model = 128
        fine_length = 256
        coarse_stride = 4.0
        coarse_length = fine_length // int(coarse_stride)

        pe = SinusoidalPositionalEncoding(d_model)
        fine_pe = pe(
            fine_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
            stride=1.0,
        ).squeeze(0)
        coarse_pe = pe(
            coarse_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
            stride=coarse_stride,
        ).squeeze(0)

        fine_pe = F.normalize(fine_pe, dim=-1)
        coarse_pe = F.normalize(coarse_pe, dim=-1)

        for coarse_idx in (5, 12, 31):
            coarse_token = coarse_pe[coarse_idx]
            span_start = int(coarse_idx * coarse_stride)
            span_end = int((coarse_idx + 1) * coarse_stride)
            span_vectors = fine_pe[span_start:span_end]
            similarities = span_vectors @ coarse_token

            peak_idx = torch.argmax(similarities).item()
            self.assertEqual(
                peak_idx,
                0,
                msg=(
                    "Stride-aware PE should align coarse token phase with "
                    "the fine token at the same real-time coordinate."
                ),
            )


if __name__ == "__main__":
    unittest.main()
