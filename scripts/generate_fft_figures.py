#!/usr/bin/env python3
"""Computed photograph/FFT figures. Run from any directory; see docs/fft-figures.md."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from map_from_fft import load_image_and_luminance, find_top_fft_peaks
from fft_image_explorer import make_mask
SIZE = 401
SOFTNESS = .2
SIGMA = SIZE / 2 * SOFTNESS
INDICES = [5, 80, 280]
INK = '#172c40'
CYAN = '#65efff'

def compute(rgb, luminance, points, index):
    center = points[index]
    x, y = np.rint(center).astype(int)
    x0, y0 = x-SIZE//2, y-SIZE//2
    patch = rgb[y0:y0+SIZE, x0:x0+SIZE]
    scalar = luminance[y0:y0+SIZE, x0:x0+SIZE]
    if scalar.shape != (SIZE, SIZE):
        raise ValueError('Selected window extends beyond source image')
    # Keep the saved floating-point center, with no resampling of source pixels.
    mask = make_mask(scalar.shape, 'Ellipse/circle', 'Gaussian',
                     center[0]-x0, center[1]-y0, SIZE, SIZE, 0, SOFTNESS)
    windowed = scalar * mask
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    # Secant spanning four nearby samples, in image coordinates (y down).
    tangent = points[(index+2) % (len(points)-1)]-points[(index-2) % (len(points)-1)]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    candidates = find_top_fft_peaks(np.log1p(abs(spectrum)), top_k=100, min_distance=1)[1:]
    peaks = []
    selected = [np.zeros(2)]
    for peak in candidates:
        offset = np.array([peak['x'], peak['y']])
        if min(min(np.linalg.norm(offset-q), np.linalg.norm(offset+q)) for q in selected) < 6:
            continue
        peaks.append(peak)
        selected.append(offset)
        if len(peaks) == 6:
            break
    measured = []
    for peak in peaks:
        freq = np.array([peak['x'], -peak['y']])/SIZE
        ft, fn = np.dot(freq, tangent), np.dot(freq, normal)
        if ft < 0: ft, fn = -ft, -fn
        row, col = int(SIZE//2-peak['y']), int(SIZE//2+peak['x'])
        measured.append(dict(fx=float(freq[0]), fy=float(freq[1]), ft=float(ft), fn=float(fn),
                             magnitude=float(abs(spectrum[row,col])),
                             angle_to_tangent_deg=float(np.degrees(np.arctan2(fn,ft)))))
    return dict(index=index, center=center, origin=(int(x0),int(y0)), patch=patch, scalar=scalar,
                mask=mask, windowed=windowed, spectrum=spectrum, tangent=tangent, peaks=measured)

def source(ax, d, points):
    x0,y0 = d['origin']; c=d['center']
    ax.imshow(d['patch'], extent=(x0-.5,x0+SIZE-.5,y0+SIZE-.5,y0-.5))
    overlay=np.zeros((SIZE,SIZE,4));overlay[:,:,:3]=[.2,.85,1];overlay[:,:,3]=.09*d['mask']
    ax.imshow(overlay, extent=(x0-.5,x0+SIZE-.5,y0+SIZE-.5,y0-.5))
    nearby=np.linalg.norm(points-c,axis=1)<115
    curve=points.copy();curve[~nearby]=np.nan
    line,=ax.plot(*curve.T,color='white',lw=1.1,alpha=.9)
    line.set_path_effects([pe.Stroke(linewidth=2.5,foreground=INK),pe.Normal()])
    for n,style in [(1,'-'),(2,'--')]:
        ring=Circle(c,n*SIGMA,fill=False,color=CYAN,lw=1.7,ls=style)
        ring.set_path_effects([pe.Stroke(linewidth=3,foreground=INK),pe.Normal()]);ax.add_patch(ring)
    ax.plot(*c,'+',color='white',ms=12,mew=2.3)
    ax.set_xlim(c[0]-135,c[0]+135);ax.set_ylim(c[1]+135,c[1]-135)
    ax.set_xticks([]);ax.set_yticks([])

def fft_panel(ax,d):
    freq=np.fft.fftshift(np.fft.fftfreq(SIZE));step=1/SIZE
    # Full FFT is unchanged. View only the central frequency region, saturating DC.
    im=ax.imshow(np.log1p(abs(d['spectrum'])),origin='upper',cmap='magma',vmin=1,vmax=6,
                 extent=(freq[0]-step/2,freq[-1]+step/2,freq[-1]+step/2,freq[0]-step/2),interpolation='nearest')
    ax.set_xlim(-.10,.10);ax.set_ylim(.10,-.10)
    ax.set_xticks([-.08,0,.08]);ax.set_yticks([-.08,0,.08])
    ax.set_xlabel('$f_x$ (cycles / pixel)');ax.set_ylabel('$f_y$ ↓')
    # Basis arrows indicate orientation, without rotating or resampling the FFT.
    t=d['tangent']; n=np.array([-t[1],t[0]])
    for v,label in [(t,'t'),(n,'n')]:
        ax.annotate('',xy=v*.085,xytext=v*.055,arrowprops=dict(arrowstyle='->',color=CYAN,lw=1.4))
        ax.text(*(v*.091),label,color=CYAN,ha='center',va='center',fontsize=10)
    return im

def triptych(d, points, rgb, out):
    fig=plt.figure(figsize=(16,8),facecolor='white')
    fig.text(.045,.925,'The geometry inside a photograph',fontsize=29,weight='bold',color=INK)
    fig.text(.045,.875,f'Local Fourier analysis  /  necklace centerline #{d["index"]}  /  original photograph, computed data',fontsize=14,color=INK)
    axes=[fig.add_axes([x,.25,.27,.54]) for x in [.045,.365,.685]]
    source(axes[0],d,points)
    axes[1].imshow(d['windowed'],cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[1].set_xticks([]);axes[1].set_yticks([])
    im=fft_panel(axes[2],d)
    for a,title in zip(axes,['A   Photograph + window','B   Scalar × Gaussian','C   Fourier magnitude']):a.set_title(title,loc='left',pad=15,fontsize=16,weight='bold',color=INK)
    fig.text(.045,.19,'+ saved centerline point\nSolid: 1σ   ·   Dashed: 2σ',fontsize=13,color=INK)
    fig.text(.365,.19,'Luminance Y · full 401 × 401 input\nFixed grayscale 0–1; no contrast stretch',fontsize=13,color=INK)
    cb=fig.colorbar(im,cax=fig.add_axes([.72,.14,.20,.018]),orientation='horizontal',ticks=[1,3,6]);cb.set_label('ln(1 + |FFT|) · fixed scale',fontsize=11)
    loc=fig.add_axes([.045,.025,.065,.13]);loc.imshow(rgb);loc.plot(*d['center'],'o',mfc=CYAN,mec=INK,ms=5);loc.axis('off')
    fig.text(.125,.065,'Repeated bead spacing produces paired off-center peaks.\nColor contrast changes their strength; orientation changes their direction.',fontsize=13,color=INK)
    fig.savefig(out,dpi=120);plt.close(fig)

def card_figure(d, points, photo_fraction=.48, photo_half_width=90):
    """Compose a 1600×1000 card directly from Candidate 5 data, without FFT rotation.

    Rectangular views keep equal data aspect: cropping never stretches the image.
    The optional layout arguments are for visual review, not numerical analysis.
    """
    if d['index'] != 5:
        raise ValueError('The project card is fixed to the owner-selected Candidate 5')
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor='white')
    margin, gap, bottom, height = 24, 20, 24, 872
    usable = 1600 - 2*margin - gap
    left_width = usable * photo_fraction
    right_width = usable - left_width
    left = fig.add_axes([margin/1600, bottom/1000, left_width/1600, height/1000])
    right_x = margin + left_width + gap
    right = fig.add_axes([right_x/1600, bottom/1000, right_width/1600, height/1000])
    source(left, d, points)
    cx, cy = d['center']
    photo_half_height = photo_half_width * height / left_width
    left.set_xlim(cx-photo_half_width, cx+photo_half_width)
    left.set_ylim(cy+photo_half_height, cy-photo_half_height)
    for ring in left.patches:
        ring.set_linewidth(3.4)
        ring.set_path_effects([pe.Stroke(linewidth=5, foreground=INK), pe.Normal()])
    left.lines[0].set_alpha(.65)
    left.lines[0].set_linewidth(.9)
    left.lines[1].set_markersize(22)
    left.lines[1].set_markeredgewidth(3.6)
    left.lines[1].set_path_effects([pe.Stroke(linewidth=5, foreground=INK), pe.Normal()])
    freq = np.fft.fftshift(np.fft.fftfreq(SIZE))
    half_bin = .5/SIZE
    right.imshow(np.log1p(abs(d['spectrum'])), origin='upper', cmap='magma',
                 vmin=1, vmax=6, interpolation='nearest',
                 extent=(freq[0]-half_bin, freq[-1]+half_bin,
                         freq[-1]+half_bin, freq[0]-half_bin))
    # Same scale and orientation as the scientific figure; tighter view only.
    fx_limit = .08
    fy_limit = fx_limit * height / right_width
    right.set_xlim(-fx_limit, fx_limit)
    right.set_ylim(fy_limit, -fy_limit)
    for ax in (left, right):
        ax.set_axis_off()
        ax.set_aspect('equal')
    for x, label in [(margin, 'Gaussian window'), (right_x, '2D Fourier transform')]:
        fig.text(x/1600, .943, label, fontsize=32, weight='medium',
                 fontfamily='DejaVu Sans', color=INK, va='center')
    return fig


def save_card(d, points, out):
    fig = card_figure(d, points)
    fig.savefig(out/'local-fft-card.png', dpi=100)
    plt.close(fig)
    with Image.open(out/'local-fft-card.png') as im:
        im.resize((400, 250), Image.Resampling.LANCZOS).save(out/'local-fft-card-preview.png')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/figures')
    parser.add_argument('--indices',type=int,nargs=3,default=INDICES,help='First candidate becomes hero')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#aab4be','savefig.facecolor':'white'})
    rgb,y=load_image_and_luminance(str(ROOT/'beads-photo-2.jpg'))
    points=np.array(json.loads((ROOT/'beads-photo-2_splines.json').read_text())['centerline_spline'])
    data=[compute(rgb,y,points,i) for i in args.indices]
    for d in data:triptych(d,points,rgb,out/f'candidate-{d["index"]}.png')
    (out/'local-fft-hero.png').write_bytes((out/f'candidate-{args.indices[0]}.png').read_bytes())
    thumbs=[]
    for d in data:
        with Image.open(out/f'candidate-{d["index"]}.png') as im:thumbs.append(im.copy())
    sheet=Image.new('RGB',(thumbs[0].width,thumbs[0].height*3),'white')
    for k,im in enumerate(thumbs):sheet.paste(im,(0,k*im.height))
    sheet.save(out/'local-fft-candidates.png')
    card_data = next((d for d in data if d['index'] == 5), None)
    if card_data is None:
        card_data = compute(rgb, y, points, 5)
    save_card(card_data, points, out)
    fig,axes=plt.subplots(3,3,figsize=(13,13),gridspec_kw={'width_ratios':[1,1,1.1]})
    for row,d in zip(axes,data):
        source(row[0],d,points);row[0].set_title(f'Centerline #{d["index"]} · spatial window',loc='left')
        comparison_im=fft_panel(row[1],d);row[1].set_title('FFT · original image axes',loc='left')
        a=row[2]
        for k,p in enumerate(d['peaks'][:4]):
            for sign in [-1,1]:a.scatter(sign*p['ft'],sign*p['fn'],s=25+130*p['magnitude']/400,color=plt.cm.tab10(k),alpha=.75,edgecolors='white')
            a.text(p['ft']+.003,p['fn'],str(k+1),fontsize=10)
        a.axhline(0,c='#ccd3da',lw=.8);a.axvline(0,c='#ccd3da',lw=.8)
        a.set(xlim=(-.1,.1),ylim=(-.1,.1),xlabel='$f_t$ (cycles / pixel)',ylabel='$f_n$ (cycles / pixel)')
        a.set_aspect('equal');a.set_title('Peak coordinates in local basis',loc='left')
    fig.suptitle('Same construction, different local contrast',fontsize=23,weight='bold',y=.98)
    fig.text(.07,.935,'1σ / 2σ spatial contours  ·  shared FFT scale  ·  t = tangent, n = normal  ·  spectra are not rotated',fontsize=12)
    fig.text(.07,.025,'Right: four strongest nonzero peak pairs, projected onto t/n; numbered by local amplitude rank.\nMarker area above its fixed minimum encodes |FFT| on a shared scale. Ranks do not identify the same bead-lattice mode.',fontsize=11)
    fig.subplots_adjust(left=.07,right=.97,top=.9,bottom=.11,hspace=.38,wspace=.38)
    cb=fig.colorbar(comparison_im,cax=fig.add_axes([.40,.065,.20,.009]),orientation='horizontal',ticks=[1,3,6]);fig.text(.62,.064,'ln(1 + |FFT|)',fontsize=9);cb.ax.tick_params(labelsize=8)
    fig.savefig(out/'local-fft-comparison.png',dpi=150);plt.close(fig)
    report={'source':'beads-photo-2.jpg','sha256':hashlib.sha256((ROOT/'beads-photo-2.jpg').read_bytes()).hexdigest(),'size':SIZE,'softness':SOFTNESS,'sigma_px':SIGMA,'channel':'Y = 0.299 R + 0.587 G + 0.114 B (encoded RGB)','fft':'numpy fft2, backward normalization, fftshift; no padding, filtering or mean subtraction','display':'ln(1+abs(F)), shared limits [1,6], frequency view +/-0.1 cycles/pixel; image y and fy down','candidates':[]}
    for d in data:
        report['candidates'].append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items() if k in ['index','center','origin','tangent','peaks']})
    (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
