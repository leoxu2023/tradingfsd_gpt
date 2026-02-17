import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_thetadata_option_snapshots.py"
spec = importlib.util.spec_from_file_location("fetch_thetadata_option_snapshots", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


class ThetaDataFetchScriptTests(unittest.TestCase):
    def test_hhmm_conversion(self):
        self.assertEqual(module.hhmm_to_ms_of_day("10:00"), 36000000)
        self.assertEqual(module.hhmm_to_ms_of_day("14:00"), 50400000)

    def test_ms_to_hhmmss(self):
        self.assertEqual(module.ms_to_hhmmss(36000000), "100000")
        self.assertEqual(module.ms_to_hhmmss(50400000), "140000")

    def test_build_bulk_url_uses_given_root(self):
        url = module.build_bulk_at_time_url(
            "http://127.0.0.1:25510",
            "SPXW",
            "20260217",
            "20260217",
            39600000,
        )
        self.assertIn("root=SPXW", url)
        self.assertIn("exp=20260217", url)
        self.assertIn("ivl=39600000", url)


if __name__ == "__main__":
    unittest.main()
