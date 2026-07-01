#!/usr/bin/env python3
"""Build Oncura Demo native app with PyInstaller."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DIST = DEMO_DIR.parent / "dist"
APP_NAME = "Oncura Demo"


def ensure_icon() -> Path | None:
    assets = DEMO_DIR / "assets"
    assets.mkdir(exist_ok=True)
    png = assets / "icon.png"
    if not png.exists():
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (512, 512), "#1e3a5f")
            draw = ImageDraw.Draw(img)
            draw.ellipse((64, 64, 448, 448), fill="#2563eb")
            draw.text((190, 200), "O", fill="white")
            img.save(png)
        except Exception:
            return None

    system = platform.system()
    if system == "Darwin":
        icns = assets / "icon.icns"
        if not icns.exists():
            iconset = assets / "icon.iconset"
            iconset.mkdir(exist_ok=True)
            subprocess.run(
                ["sips", "-z", "512", "512", str(png), "--out", str(iconset / "icon_512x512.png")],
                check=True,
            )
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
        return icns
    if system == "Windows":
        ico = assets / "icon.ico"
        if not ico.exists():
            from PIL import Image

            Image.open(png).save(ico, format="ICO", sizes=[(256, 256)])
        return ico
    return png


def build() -> Path:
    icon = ensure_icon()
    for d in (DEMO_DIR / "build", DIST):
        if d.exists() and d.name == "build":
            shutil.rmtree(d, ignore_errors=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--paths",
        str(DEMO_DIR),
        "--add-data",
        f"{DEMO_DIR / 'models'}{';models' if platform.system() == 'Windows' else ':models'}",
    ]
    if platform.system() == "Windows":
        cmd.append("--onefile")
    cmd.append(str(DEMO_DIR / "native_app.py"))
    if icon:
        cmd.extend(["--icon", str(icon)])

    subprocess.check_call(cmd, cwd=DEMO_DIR)

    system = platform.system()
    if system == "Darwin":
        app_path = DEMO_DIR / "dist" / f"{APP_NAME}.app"
        if not app_path.exists():
            raise FileNotFoundError(app_path)
        out_zip = DIST / "Oncura-Demo-mac.zip"
        DIST.mkdir(parents=True, exist_ok=True)
        if out_zip.exists():
            out_zip.unlink()
        shutil.make_archive(str(out_zip.with_suffix("")), "zip", app_path.parent, app_path.name)
        return out_zip
    if system == "Windows":
        exe = DEMO_DIR / "dist" / f"{APP_NAME}.exe"
        if not exe.exists():
            raise FileNotFoundError(exe)
        out = DIST / "Oncura-Demo-Windows.exe"
        DIST.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, out)
        return out
    raise RuntimeError(f"Unsupported build platform: {system}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    artifact = build()
    print(f"Built {artifact}")


if __name__ == "__main__":
    main()
