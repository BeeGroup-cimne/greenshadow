# -*- coding: utf-8 -*-
from PIL import Image
import os
import sys

def pngs_to_gif(png_dir, output_gif=None, duration=300, loop=0):
    """
    Merge PNGs in a directory into a GIF.

    Parameters:
        png_dir (str): Path to directory with PNG images.
        output_gif (str): Path to save the output GIF. Defaults to 'output.gif' in png_dir.
        duration (int): Time (ms) between frames.
        loop (int): Number of loops (0 = infinite).
    """
    if output_gif is None:
        output_gif = os.path.join(png_dir, "output.gif")
    
    # Get PNG files sorted
    files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
    
    if not files:
        print(f"No PNG files found in {png_dir}")
        return
    
    # Open images
    images = [Image.open(os.path.join(png_dir, f)) for f in files]

    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    print(f"GIF created: {output_gif}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python png_to_gif.py <png_directory> [output_gif] [duration] [loop]")
        sys.exit(1)
    
    png_dir = sys.argv[1]
    output_gif = sys.argv[2] if len(sys.argv) > 2 else None
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    loop = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    pngs_to_gif(png_dir, output_gif, duration, loop)
