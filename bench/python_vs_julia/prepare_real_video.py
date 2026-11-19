"""Prepare licensed Pexels footage and optional shared benchmark tensors.

Example: python prepare_real_video.py /tmp/video --cached-dir /tmp/originals
Add --tensor-size 640 360 to extract grayscale tensors at processing resolution.
Cached files must be named landscape-1080p.mp4 and portrait-1080p.mp4.
The tensor shape is (width, height, 300), Float32 little-endian, Fortran order;
pixel intensities are in [0, 1]. Requires ffmpeg, ffprobe and NumPy.
"""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

SOURCES = {
    'landscape': ('40276646', 1920, 1080),
    'portrait': ('18440087', 1080, 1920),
}


def run(*args):
    subprocess.run(args, check=True)


def probe(path):
    result = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration', '-of', 'json', str(path)
    ])
    info = json.loads(result)
    stream = info['streams'][0]
    return (stream['width'], stream['height'], float(info['format']['duration']),
            stream.get('avg_frame_rate'), stream.get('nb_frames'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--cached-dir', type=Path, help='use local originals, without downloading')
    p.add_argument('--tensor-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                   help='extract grayscale tensor at this landscape processing size; portrait dimensions are swapped')
    p.add_argument('--native-tensor', action='store_true',
                   help='extract tensor at each clip’s native dimensions')
    args = p.parse_args()
    if args.native_tensor and args.tensor_size:
        p.error('--native-tensor and --tensor-size are mutually exclusive')
    if args.tensor_size and (min(args.tensor_size) < 1 or args.tensor_size[0] <= args.tensor_size[1]):
        p.error('--tensor-size requires positive landscape WIDTH > HEIGHT')
    args.output.mkdir(parents=True, exist_ok=True)
    for orientation, (video_id, width, height) in SOURCES.items():
        name = f'{orientation}-1080p.mp4'
        source = (args.cached_dir or args.output / 'originals') / name
        if not source.exists():
            if args.cached_dir:
                p.error(f'cached original missing: {source}')
            source.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://www.pexels.com/download/video/{video_id}/'
            print(f'Downloading {url} to {source}', flush=True)
            temp = source.with_suffix('.part')
            try:
                urllib.request.urlretrieve(url, temp)
                temp.replace(source)
            finally:
                temp.unlink(missing_ok=True)
        w, h, duration, _, _ = probe(source)
        if duration < 10 or w < width or h < height or (w > h) != (width > height):
            raise ValueError(f'{source}: expected at least {width}x{height}, 10 seconds; got {w}x{h}, {duration}s')
        for short in (720, 1080):
            out_w, out_h = ((short * 16 // 9, short) if orientation == 'landscape'
                            else (short, short * 16 // 9))
            case = args.output / f'{orientation}-{short}p'
            case.mkdir(exist_ok=True)
            clip = case / 'clip.mp4'
            valid = False
            if clip.exists():
                try:
                    cw, ch, cd, rate, frames = probe(clip)
                    valid = ((cw, ch) == (out_w, out_h) and cd >= 9.99
                             and rate == '30/1' and frames == '300')
                except (subprocess.CalledProcessError, ValueError, KeyError, IndexError):
                    pass
            if not valid:
                run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                    '-vf', f'fps=30,scale={out_w}:{out_h}:flags=lanczos', '-frames:v', '300',
                    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip))
                cw, ch, cd, rate, frames = probe(clip)
                if (cw, ch) != (out_w, out_h) or cd < 9.99 or rate != '30/1' or frames != '300':
                    raise ValueError(f'Invalid output {clip}: {cw}x{ch}, {cd}s, {rate}, {frames} frames')
            if args.tensor_size or args.native_tensor:
                import numpy as np
                tw, th = ((out_w, out_h) if args.native_tensor else
                          (args.tensor_size if orientation == 'landscape' else args.tensor_size[::-1]))
                raw = case / 'gray.raw'
                tensor = case / 'tensor.f32'
                expected = tw * th * 300
                digest = hashlib.sha256()
                with clip.open('rb') as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        digest.update(block)
                manifest = case / 'tensor.json'
                identity = {'clip_sha256': digest.hexdigest(), 'shape': [tw, th, 300],
                            'format': 'gray8-scaled-float32-le-fortran-v1'}
                try:
                    cached_identity = json.loads(manifest.read_text())
                except (OSError, ValueError):
                    cached_identity = None
                if (not tensor.exists() or tensor.stat().st_size != expected * 4
                        or cached_identity != identity):
                    try:
                        run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(clip),
                            '-vf', f'scale={tw}:{th}:flags=lanczos,format=gray', '-frames:v', '300',
                            '-f', 'rawvideo', str(raw))
                        if raw.stat().st_size != expected:
                            raise ValueError(f'{raw}: expected {expected} bytes, got {raw.stat().st_size}')
                        temp = case / 'tensor.f32.tmp'
                        with raw.open('rb') as src, temp.open('wb') as dst:
                            while chunk := src.read(1024 * 1024):
                                (np.frombuffer(chunk, dtype=np.uint8).astype('<f4') / np.float32(255)).tofile(dst)
                        temp.replace(tensor)
                        manifest_tmp = case / 'tensor.json.tmp'
                        manifest_tmp.write_text(json.dumps(identity, sort_keys=True) + '\n')
                        manifest_tmp.replace(manifest)
                    finally:
                        raw.unlink(missing_ok=True)
                coordinates = tw * tw + th * th + 300 * 300
                x = case / 'x.f32'
                if not x.exists() or x.stat().st_size != coordinates * 4:
                    rng = np.random.default_rng(20261118)
                    temp = case / 'x.f32.tmp'
                    with temp.open('wb') as dst:
                        for start in range(0, coordinates, 1_000_000):
                            rng.standard_normal(min(1_000_000, coordinates - start), dtype=np.float32).astype('<f4', copy=False).tofile(dst)
                    temp.replace(x)
                print(f'{case}: tensor shape=({tw}, {th}, 300)', flush=True)
            print(f'{clip}: {cw}x{ch}, {cd:.3f}s', flush=True)


if __name__ == '__main__':
    main()
