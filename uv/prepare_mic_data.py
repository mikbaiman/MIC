"""Assemble the supplied GTA parts and Cityscapes without moving source images."""
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from zipfile import ZipFile
import zlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError('Refusing to replace existing data: {}'.format(destination))
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def crc(path):
    value = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value = zlib.crc32(chunk, value)
    return value


def main():
    images = {}
    for part in range(1, 11):
        folder = DATA / '{:02d}_images/images'.format(part)
        for path in sorted(folder.glob('*.png')):
            if path.name in images:
                raise RuntimeError('Duplicate image: ' + path.name)
            images[path.name] = path
    if len(images) != 24966:
        raise RuntimeError('Expected 24966 GTA images; found {}'.format(len(images)))
    labels = DATA / 'gta/labels'
    labels.mkdir(parents=True, exist_ok=True)
    names = set()
    for part in range(1, 11):
        archive = DATA / '{:02d}_labels.zip'.format(part)
        with ZipFile(str(archive)) as zipped:
            for item in zipped.infolist():
                if item.is_dir():
                    continue
                if not re.fullmatch(r'labels/\d{5}\.png', item.filename):
                    raise RuntimeError('Unexpected label archive entry: ' + item.filename)
                name = Path(item.filename).name
                if name in names:
                    raise RuntimeError('Duplicate label: ' + name)
                names.add(name)
                destination = labels / name
                if destination.exists():
                    if destination.stat().st_size != item.file_size or crc(destination) != item.CRC:
                        raise RuntimeError('Existing label differs: ' + str(destination))
                else:
                    with zipped.open(item) as src, destination.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
            print('Verified/extracted', archive.name, flush=True)
    if set(images) != names:
        raise RuntimeError('GTA image/label filenames do not match')
    for name, path in images.items():
        destination = DATA / 'gta/images' / name
        if destination.is_symlink() and destination.resolve() == path.resolve():
            continue
        link(path, destination)
    link(DATA / 'gta', ROOT / 'seg/data/gta')
    for name in ('gtFine', 'leftImg8bit'):
        link(DATA / name, ROOT / 'seg/data/cityscapes' / name)

    checkpoint_root = DATA / 'checkpoints'
    checkpoint_root.mkdir(exist_ok=True)
    with tarfile.open(str(DATA / 'gtaHR2csHR_mic_hrda_650a8.tar.gz')) as archive:
        for item in archive.getmembers():
            if not item.isfile():
                continue  # The regular iter_40000_relevant.pth is used, not archive links.
            destination = checkpoint_root / item.name
            if checkpoint_root.resolve() not in destination.resolve().parents:
                raise RuntimeError('Unsafe checkpoint archive entry')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != item.size:
                    raise RuntimeError('Existing checkpoint file differs: ' + str(destination))
                continue
            with archive.extractfile(item) as src, destination.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    report = dict(gta_images=len(images), gta_labels=len(names), cityscapes={})
    for split, expected in (('train', 2975), ('val', 500), ('test', 1525)):
        rgb = list((DATA / 'leftImg8bit' / split).rglob('*_leftImg8bit.png'))
        mask = list((DATA / 'gtFine' / split).rglob('*_gtFine_labelTrainIds.png'))
        if len(rgb) != expected or len(mask) != expected:
            raise RuntimeError('Cityscapes {} requires preprocessing or is incomplete'.format(split))
        for path in rgb:
            relative = path.relative_to(DATA / 'leftImg8bit' / split)
            label = DATA / 'gtFine' / split / str(relative).replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
            if not label.is_file():
                raise RuntimeError('Missing Cityscapes label: ' + str(label))
        report['cityscapes'][split] = len(rgb)
    out = ROOT / 'uv/seg/runs/mic-setup'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
