import os
import numpy as np
import imageio
import argparse
from tqdm import tqdm


class GridVid:
    def __init__(
        self,
        dirs,
        nrows,
        ncols,
        alpha=False
    ):
        assert (len(dirs) == nrows * ncols), 'Invalid number of directories'

        # Read all images
        with tqdm(total=ncols*nrows) as pbar:
            self.in_imgs = []
            for r in range(nrows):
                for c in range(ncols):
                    dir = dirs[r * ncols + c]
                    def is_img_file(f):
                        ext = f.split('.')[-1].lower()
                        return ext == 'png' or ext == 'jpg' or ext == 'jpeg'
                    # Allows mixing zero- and one-indexed naming conventions
                    dir_img_files = sorted([f for f in os.listdir(dir) if is_img_file(f)])
                    def imread(f):
                        if not alpha:
                            return imageio.imread(f)[..., :3] # drop alpha
                        else:
                            return imageio.imread(f)
                    self.in_imgs.append([imread(os.path.join(dir, f)) for f in dir_img_files])
                    pbar.update(1)

        self.nimgs = len(self.in_imgs[0])
        for c in range(ncols):
            for r in range(nrows):
                assert (len(self.in_imgs[r * ncols + c]) == self.nimgs), 'Inconsistent number of images'

        self.dirs = dirs
        self.nrows = nrows
        self.ncols = ncols
        self.alpha = alpha

    def process(
        self,
        outdir,
        resx=-1,
        resy=-1
    ):
        rescale = (resx > 0 and resy > 0)
        if not rescale:
            # Assuming all images have the same dimensions
            self.dimy, self.dimx = self.in_imgs[0][0].shape[:2]
        else:
            self.dimx = resx
            self.dimy = resy

        # Combine the images and save them all
        os.makedirs(outdir, exist_ok=True)
        for i in tqdm(range(self.nimgs)):
            out_img = np.empty((self.nrows * self.dimy, self.ncols * self.dimx, 3 if not self.alpha else 4))
            for r in range(self.nrows):
                for c in range(self.ncols):
                    imgs = self.in_imgs[r * self.ncols + c]
                    img = imgs[i]
                    if rescale:
                        from PIL import Image
                        Img = Image.fromarray(img, 'RGB' if not self.alpha else 'RGBA')
                        Img = Img.resize((resx, resy), Image.ANTIALIAS)
                        img = np.array(Img)
                    out_img[r*self.dimy:(r+1)*self.dimy, c*self.dimx:(c+1)*self.dimx] = img
            imageio.imwrite(
                os.path.join(outdir, '{:04d}.png'.format(i)),
                out_img.astype(np.uint8)
            )

    def image_sequence_to_video(
        self,
        imgdir,
        name='vid',
        framerate=10
    ):
        ffmpeg_boilerplate = '-y -f image2 -framerate {}'.format(framerate)
        ffmpeg_boilerplate += ' -hide_banner -loglevel error'
        cmd = 'ffmpeg {} -i {}/%04d.png {}'.format(
            ffmpeg_boilerplate,
            imgdir,
            os.path.join(imgdir, '{}.mp4'.format(name))
        )
        print(cmd)
        os.system(cmd)

    def image_sequence_to_gif(
        self,
        imgdir,
        name='vid',
        framerate=10
    ):
        ffmpeg_boilerplate = '-y -f image2 -framerate {}'.format(framerate)
        ffmpeg_boilerplate += ' -hide_banner -loglevel error'
        # Heuristic reasonable scale determination
        grid_height = self.nrows * self.dimy
        grid_width = self.ncols * self.dimx
        if grid_width >= grid_height:
            gif_width = min(grid_width // 2, self.dimx * 2)
            ffmpeg_filter = '-vf \"scale={}:-1:flags=lanczos'.format(gif_width)
        else:
            gif_height = min(grid_height // 2, self.dimy * 2)
            ffmpeg_filter = '-vf \"scale=-1:{}:flags=lanczos'.format(gif_height)
        # Palette gives better quality results
        ffmpeg_filter += ',split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\"'
        cmd = 'ffmpeg {} -i {}/%04d.png {} -loop 0 {}'.format(
            ffmpeg_boilerplate,
            imgdir,
            ffmpeg_filter,
            os.path.join(imgdir, '{}.gif'.format(name))
        )
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a grid video from directories of images'
    )
    parser.add_argument(
        'dirs',
        nargs='+',
        help='directories containing images (in row-major order)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='output directory for grid video'
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=1,
        help='number of rows in grid'
    )
    parser.add_argument(
        '--ncols',
        type=int,
        default=-1,
        help='number of columns (default assumes single row)'
    )
    parser.add_argument(
        '--resx',
        type=int,
        default=-1,
        help='resolution to rescale all images (default assumes all are same dimension)'
    )
    parser.add_argument(
        '--resy',
        type=int,
        default=-1,
        help='resolution to rescale all images (default assumes all are same dimension)'
    )
    parser.add_argument(
        '--keep_alpha',
        default=False,
        action="store_true",
        help='preserve alpha channel of input'
    )
    args = parser.parse_args()

    print('Loading images...')
    if args.ncols < 0:
        gv = GridVid(args.dirs, 1, len(args.dirs), args.keep_alpha)
    else:
        gv = GridVid(args.dirs, args.nrows, args.ncols, args.keep_alpha)
    print('Compiling grid video...')
    gv.process(args.output, args.resx, args.resy)
    gv.image_sequence_to_video(args.output)
    gv.image_sequence_to_gif(args.output)
    print('Grid video completed')
