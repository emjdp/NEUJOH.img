"""Small end-to-end checks that do not require the optional matte model."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

import neujoh
from neujoh.cli import main


class RendererSmokeTest(TestCase):
    def test_bundled_font_and_renderer(self) -> None:
        self.assertTrue(Path(neujoh.DEFAULT_FONT).is_file())
        source = Image.fromarray(
            np.linspace(0, 255, 48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
        )
        config = neujoh.Config(
            cols=8,
            cell_w=6,
            cell_h=8,
            charset="minimal",
            matte=False,
            edges=0,
            bloom=0,
        )
        result = neujoh.convert(source, config)
        output = neujoh.composite(result, config)

        self.assertEqual(result["cols"], 8)
        self.assertEqual(output.width, result["cols"] * config.cell_w)
        self.assertEqual(output.height, result["rows"] * config.cell_h)

    def test_cli_writes_png_and_text(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            output = root / "result"
            Image.new("RGB", (32, 24), (60, 120, 180)).save(source)

            main([
                str(source),
                "--no-matte",
                "--cols",
                "6",
                "--cell",
                "6x8",
                "--charset",
                "minimal",
                "--edges",
                "0",
                "--bloom",
                "0",
                "-o",
                str(output),
            ])

            self.assertTrue(output.with_suffix(".png").is_file())
            self.assertTrue(output.with_suffix(".txt").is_file())
