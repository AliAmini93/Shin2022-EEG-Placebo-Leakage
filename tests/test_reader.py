import sys, tempfile, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shin2022_leakage.pipeline import reader as R


class TestReader(unittest.TestCase):
    def test_roundtrip_signal_states_and_redaction(self):
        rng = np.random.default_rng(3)
        sig = rng.normal(size=(1000, 5)).astype(np.float32)
        st = {'A': (1, rng.integers(0, 2, 1000)), 'B': (12, rng.integers(0, 4096, 1000)), 'C': (24, np.arange(1000)),
              'D': (3, rng.integers(0, 8, 1000))}
        prm = ['Src int SourceCh= 5 5 1 % //', 'Src float SamplingRate= 250Hz 250Hz 0 % //',
               'Src list SourceChGain= 5 2 2 2 2 2 % % % //', 'Src list SourceChOffset= 5 0 0 0 0 1 % % % //',
               'Storage:Documentation string SubjectName= PRIVATE%20NAME % % % //',
               'App:X matrix M= { r1 r2 } { c1 c2 c3 } 1 2 3 4 5 6 // labelled matrix']
        p = Path(tempfile.mkdtemp()) / 'x.dat'
        R.write_bci2000(p, sig, st, prm)
        h = R.read_header(p)
        self.assertEqual(h.header_len, len(p.read_bytes()) - 1000 * (5 * 4 + h.statevector_len))
        s, states = R.read_data(p, h)
        np.testing.assert_allclose(s[:, :4], sig[:, :4].astype(np.float64) * 2)
        np.testing.assert_allclose(s[:, 4], (sig[:, 4].astype(np.float64) - 1) * 2)
        for k, (_, v) in st.items():
            np.testing.assert_array_equal(states[k], v)
        self.assertEqual(h.params['SubjectName'].value, '<redacted>')
        self.assertEqual(h.private_digest('SubjectName'), R._digest('PRIVATE NAME'))
        self.assertNotIn('PRIVATE', repr(h.params))
        self.assertEqual(h.params['M'].shape, (2, 3))
        self.assertEqual(h.matrix('M'), [['1', '2', '3'], ['4', '5', '6']])
        self.assertEqual(h.sampling_rate, 250.0)


if __name__ == '__main__':
    unittest.main()
