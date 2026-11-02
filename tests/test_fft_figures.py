import json
import unittest
import numpy as np
from scripts.generate_fft_figures import ROOT, SIZE, SIGMA, INDICES, compute
from map_from_fft import load_image_and_luminance, make_gaussian_window
from fft_image_explorer import make_mask

class FigureNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb, cls.y = load_image_and_luminance(str(ROOT/'beads-photo-2.jpg'))
        cls.points = np.array(json.loads((ROOT/'beads-photo-2_splines.json').read_text())['centerline_spline'])
        cls.data = [compute(cls.rgb, cls.y, cls.points, i) for i in INDICES]

    def test_source_and_scalar_correspondence(self):
        self.assertEqual(self.rgb.shape, (3182,2540,3))
        for d in self.data:
            x,y=d['origin']
            np.testing.assert_array_equal(d['patch'],self.rgb[y:y+SIZE,x:x+SIZE])
            expected=.299*d['patch'][...,0]+.587*d['patch'][...,1]+.114*d['patch'][...,2]
            np.testing.assert_array_equal(d['scalar'],expected)
            np.testing.assert_array_equal(d['windowed'],expected*d['mask'])

    def test_exact_saved_center_and_gaussian(self):
        for d in self.data:
            np.testing.assert_array_equal(d['center'],self.points[d['index']])
            yy,xx=np.indices((SIZE,SIZE));x0,y0=d['origin'];cx,cy=d['center']
            radius2=(xx+x0-cx)**2+(yy+y0-cy)**2
            expected=np.exp(-radius2/(2*SIGMA**2));expected[radius2>(SIZE/2)**2]=0
            np.testing.assert_allclose(d['mask'],expected,atol=4e-8)
            iy,ix=np.unravel_index(np.argmax(d['mask']),d['mask'].shape)
            self.assertLessEqual(abs(ix+x0-cx),.5);self.assertLessEqual(abs(iy+y0-cy),.5)

    def test_scanner_agreement_inside_circular_support(self):
        circular=make_mask((SIZE,SIZE),'Ellipse/circle','Gaussian',200,200,SIZE,SIZE,0,.2)
        scanner=make_gaussian_window(SIZE,.2)
        np.testing.assert_allclose(circular[circular>0],scanner[circular>0],rtol=3e-6)

    def test_fft_input_inverse_and_conjugacy(self):
        for d in self.data:
            f=d['spectrum'];w=d['windowed']
            np.testing.assert_allclose(np.fft.ifft2(np.fft.ifftshift(f)).real,w,atol=3e-7)
            np.testing.assert_allclose(f,np.conj(f[::-1,::-1]),atol=2e-4)
            self.assertAlmostEqual(float(f[200,200].real),float(w.sum(dtype=np.float64)),places=3)
            self.assertAlmostEqual(float((abs(f)**2).sum()/SIZE**2),float((w.astype(float)**2).sum()),places=3)

    def test_peak_bins_and_local_basis(self):
        for d in self.data:
            t=d['tangent'];n=np.array([-t[1],t[0]])
            self.assertAlmostEqual(float(t@t),1)
            for p in d['peaks']:
                col=round(p['fx']*SIZE)+200;row=round(p['fy']*SIZE)+200
                mag=abs(d['spectrum'])
                self.assertAlmostEqual(p['magnitude'],float(mag[row,col]),delta=5e-5)
                self.assertGreaterEqual(mag[row,col]+1e-4,mag[row-1:row+2,col-1:col+2].max())
                self.assertAlmostEqual(p['ft']**2+p['fn']**2,p['fx']**2+p['fy']**2)

    def test_frequency_orientation_with_known_mode(self):
        yy,xx=np.indices((SIZE,SIZE))
        signal=np.cos(2*np.pi*(11*xx+17*yy)/SIZE)
        f=np.fft.fftshift(np.fft.fft2(signal))
        self.assertAlmostEqual(abs(f[200+17,200+11]),SIZE**2/2,places=6)
        freq=np.fft.fftshift(np.fft.fftfreq(SIZE))
        self.assertAlmostEqual(freq[217],17/SIZE)

if __name__=='__main__':unittest.main()
