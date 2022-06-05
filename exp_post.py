import os, logging, math, glob
import numpy as np
import torch
import imageio
from pyhocon import ConfigFactory
import trimesh
import trimesh.proximity as prox
from tqdm import tqdm
from utils.args import ParseArgs
from utils.dataset import Dataset, NonrigidDataset
# N.B. modules imported in functions: matplotlib, pyrender


class PostProcessor:
    def __init__(self, conf_path, case):
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_data_dir = self.conf['dataset.data_dir']
        self.base_exp_dir = self.conf['general.base_exp_dir']
        if not os.path.isdir(self.base_data_dir):
            raise RuntimeError('please verify data directory is correct!')
        if not os.path.isdir(self.base_exp_dir):
            raise RuntimeError('please verify an experiment was run!')

        self.dataset = None
        self.nonrigid_data = False
        self.template_frames = -1
        if 'dataset.data_type' in self.conf:
            if self.conf['dataset.data_type'] == 'nonrigid':
                self.dataset = NonrigidDataset(
                    self.conf['dataset'],
                    cuda=torch.cuda.is_available(),
                    load_images=False
                )
                self.nonrigid_data = True
                self.template_frames = self.dataset.template_frames
        if self.dataset is None:
            self.dataset = Dataset(
                self.conf['dataset'],
                cuda=torch.cuda.is_available()
            )

    def process(
        self
    ):
        logging.info('Beginning processing of results for {}'.format(self.base_exp_dir))

        gtdir = os.path.join(self.base_data_dir, 'gt_meshes')
        if not os.path.isdir(gtdir):
            raise RuntimeError('cannot process without ground truth meshes!')

        for m in self._get_mesh_dirs():
            logging.info('Processing estimated directory {}'.format(m))
            # Compute the metrics (also saves to a text file in the directory)
            errors = self._compute_metrics(m, gtdir)
            try:
                # Compute the error vertex colorings
                self._error_color(m, errors)
            except Exception as e: # allow metric computation if this fails
                print('post._error_color:error', e)

        logging.info('All processing complete')

    def analyze_latents(
        self,
        animation=False
    ):
        from sklearn.decomposition import PCA

        logging.info('Beginning analysis of latent codes for {}'.format(self.base_exp_dir))

        ckptdir = os.path.join(self.base_exp_dir, 'checkpoints')
        ckpts = sorted([f for f in os.listdir(ckptdir) if f.startswith('ckpt_') and f.endswith('.pth')])
        last_latents = self._load_latents(os.path.join(ckptdir, ckpts[-1])) # n_frames x latent_dim
        latent_dim = last_latents.shape[-1]
        print('latent_dim', latent_dim)
        if latent_dim > 2:
            # We use the PCs of the last checkpoint for a stable projection over all checkpoints
            pca = PCA(n_components=2)
            pca.fit(last_latents)
        elif latent_dim < 2:
            raise RuntimeError('cannot analyze 1D latents (yet?)')

        # Project all frames
        all_projected = []
        for ckpt in ckpts:
            latents = self._load_latents(os.path.join(ckptdir, ckpt))
            if latent_dim == 2:
                all_projected.append(latents)
            else:
                all_projected.append(pca.transform(latents))
        all_projected = np.array(all_projected) # len(ckpts) x n_frames x 2

        # Determine axis limits (necessary for stable plotting)
        xmax, ymax = np.max(np.max(all_projected, axis=0), axis=0)
        xmin, ymin = np.min(np.min(all_projected, axis=0), axis=0)
        # Expand slightly to avoid points directly on the borders
        xmax = xmax + (xmax - xmin) * 0.025
        xmin = xmin - (xmax - xmin) * 0.025
        ymax = ymax + (ymax - ymin) * 0.025
        ymin = ymin - (ymax - ymin) * 0.025

        # Create figure for every checkpoint and make animation
        outdir = os.path.join(self.base_exp_dir, 'latent_analysis')
        os.makedirs(outdir, exist_ok=True)
        for i, ckpt in enumerate(ckpts):
            idx = int(ckpt.replace('ckpt_', '').replace('.pth', ''))
            self._plot_pca(
                all_projected[i],
                pca=(latent_dim != 2),
                name='{:08d}'.format(idx),
                output=os.path.join(outdir, '{:04d}.png'.format(i)),
                xlim=[xmin, xmax],
                ylim=[ymin, ymax]
            )
        self._image_sequence_to_video(
            outdir,
            framerate=2
        )
        self._image_sequence_to_gif(
            outdir,
            framerate=2
        )

        if animation:
            animdir = os.path.join(outdir, 'animation')
            os.makedirs(animdir, exist_ok=True)
            self._plot_pca_animation(
                all_projected[-1],
                pca=(latent_dim != 2),
                name='{:08d}'.format(int(ckpts[-1].replace('ckpt_', '').replace('.pth', ''))),
                outdir=animdir,
                xlim=[xmin, xmax],
                ylim=[ymin, ymax]
            )
            self._image_sequence_to_video(
                animdir,
                framerate=2
            )
            self._image_sequence_to_gif(
                animdir,
                framerate=2
            )

        logging.info('All latent analysis complete')

    def render(
        self,
        render_gt,
        render_proxies,
        novel,          # true to use completely novel view
        view            # -1 for camera view
    ):
        logging.info('Beginning rendering of results for {}'.format(self.base_exp_dir))

        # Assemble some miscellaneous videos
        for s in self._get_seq_dirs():
            self._image_sequence_to_video(s)
            self._image_sequence_to_gif(s)

        subdir = 'cam'
        poses = self.dataset.raw_poses
        # Replace the poses by the novel poses
        if novel:
            poses = self.dataset.raw_render_poses
            subdir = 'novel'
        # Replace the poses by the view pose
        if view >= 0:
            pose = poses[view]
            poses = pose[np.newaxis, ...].repeat(len(poses), axis=0)
            subdir += '_{}'.format(view)
        # May require homogeneous extension
        if poses.shape[-2] != poses.shape[-1]:
            hgext = np.repeat(np.array([0, 0, 0, 1])[np.newaxis, np.newaxis, :], poses.shape[0], axis=0)
            poses = np.hstack((poses, hgext))

        def cam_overlay(renderdir, poses):
            # Can't overlay if we don't have the images!
            if not novel and view < 0:
                logging.info('Overlaying renderings on ground truth: {}'.format(renderdir))
                overdir = os.path.join(renderdir, 'overlay')
                os.makedirs(overdir, exist_ok=True)
                self._alpha_over(
                    renderdir,
                    len(poses),
                    overdir
                )

        if render_gt:
            logging.info('Rendering ground truth meshes')
            outdir = os.path.join(self.base_exp_dir, 'gt', subdir)
            os.makedirs(outdir, exist_ok=True)
            self._render_mesh_sequence(
                os.path.join(self.base_data_dir, 'gt_meshes'),
                poses,
                outdir
            )
            cam_overlay(outdir, poses)

        if render_proxies:
            logging.info('Rendering geometry proxies')
            outdir = os.path.join(self.base_exp_dir, 'proxies', subdir)
            os.makedirs(outdir, exist_ok=True)
            self._render_mesh_sequence(
                os.path.join(self.base_data_dir, 'meshes'),
                poses,
                outdir
            )
            cam_overlay(outdir, poses)
        
        # foreach of the produced meshes folders:
        for m in self._get_mesh_dirs():
            # Render the produced meshes (with and without color)
            outdir = os.path.join(m, subdir)
            os.makedirs(outdir, exist_ok=True)
            logging.info('Rendering estimated directory without color: {}'.format(m))
            self._render_mesh_sequence(
                m,
                poses,
                outdir,
                False # without color
            )
            cam_overlay(outdir, poses)
            outdir = os.path.join(m, subdir + '_color')
            os.makedirs(outdir, exist_ok=True)
            logging.info('Rendering estimated directory with vertex colors: {}'.format(m))
            self._render_mesh_sequence(
                m,
                poses,
                outdir,
                True # with vertex colors
            )
            cam_overlay(outdir, poses)

            # Render the error meshes
            errordir = os.path.join(m, 'errors')
            if os.path.isdir(errordir):
                outdir = os.path.join(errordir, subdir)
                os.makedirs(outdir, exist_ok=True)
                logging.info('Rendering estimated directory errors: {}'.format(m))
                self._render_mesh_sequence(
                    errordir,
                    poses,
                    outdir,
                    True # doesn't make sense to render errors without color
                )
                cam_overlay(outdir, poses)

        logging.info('All rendering complete')

    def _get_mesh_dirs(
        self
    ):
        meshdirs = []
        # Output mesh directories
        patterns = [
            'meshes_*',         # default: uses object BBOX (if available)
            'full_scene_*',     # full scene (preferred)
            'custom_*',         # custom bounds
            'anim_*',           # animated canonical (legacy)
            'animated_*',       # animated canonical
            'animfull_scene_*', # animated canonical full scene
            'animcust_*'        # animated canonical custom bounds
        ]
        for p in patterns:
            for d in glob.glob(os.path.join(self.base_exp_dir, p)):
                meshdirs.append(d)
        # Returns list with path to the directory
        return meshdirs

    def _get_seq_dirs(
        self
    ):
        seqdirs = []
        # Output sequence directories
        patterns = [
            'novel_*',      # novel view seqeuence
            'sequence_*'    # camera view re-creation
        ]
        for p in patterns:
            for d in glob.glob(os.path.join(self.base_exp_dir, p)):
                seqdirs.append(os.path.join(d, 'normals'))
                seqdirs.append(os.path.join(d, 'validations_fine'))
        # Returns list with path to the directory
        return seqdirs

    def _load_latents(
        self,
        ckpt_file
    ):
        checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
        return checkpoint['bending_latents'].detach().numpy()

    def _plot_pca_animation(
        self,
        projected,
        pca=True,
        name='',
        outdir='animation',
        xlim=[],
        ylim=[]
    ):
        for i in range(len(projected)):
            self._plot_pca(
                projected,
                pca=pca,
                name=name,
                output=os.path.join(outdir, '{:04d}.png'.format(i)),
                xlim=xlim,
                ylim=ylim,
                highlight=i
            )

    def _plot_pca(
        self,
        projected,
        pca=True,
        name='',
        output='test.png',
        xlim=[],            # necessary for stability over a sequence
        ylim=[],            # necessary for stability over a sequence
        highlight=-1        # highlight an index into projected
    ):
        import matplotlib
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        colors = []
        sizes = None
        if highlight < 0:
            # Sequential coloring highlights neighbour frame similarity argument
            norm = matplotlib.colors.Normalize(vmin=0, vmax=len(projected), clip=True)
            mapper = cm.ScalarMappable(norm, cm.gist_ncar)
            for i in range(len(projected)):
                colors.append(mapper.to_rgba(i))
        else:
            sizes = []
            for _ in range(len(projected)):
                colors.append([.5, .5, .5, 1])
                sizes.append(plt.rcParams['lines.markersize'] ** 2) # default
            projected = np.vstack((projected, projected[highlight]))
            colors.append([0., 1., 0., 1.])
            sizes.append(plt.rcParams['lines.markersize'] ** 2.5) # larger size
        
        plt.clf()
        plt.scatter(projected[:, 0], projected[:, 1], c=colors, s=sizes)
        title = '2D '
        if pca: title += 'PCA of '
        title += 'Latent Codes ({})'.format(name)
        plt.title('2D PCA of Latent Codes ({})'.format(name))
        if pca:
            plt.xlabel('First PC')
            plt.ylabel('Second PC')
        else:
            plt.xlabel('First Dimension')
            plt.ylabel('Second Dimension')
        if highlight < 0:
            plt.colorbar(mapper)
        ax = plt.gca()
        ax.set_facecolor((0., 0., 0.)) # black background
        if len(xlim) == 2:
            plt.xlim(xlim)
        if len(ylim) == 2:
            plt.ylim(ylim)
        plt.savefig(output)

    def _alpha_over(
        self,
        renderdir,
        num,
        outdir
    ):
        for i in tqdm(range(num)):
            # Loading GT images on the fly allows reducing memory usage while post-processing
            # gt_img = self.dataset.images[i].numpy()
            gt_img = imageio.imread(os.path.join(
                self.dataset.images_dir,
                '{:04d}.png'.format(i+1)
            ))
            gt_img = (gt_img / 255.).astype(np.float32)

            render_img = imageio.imread(os.path.join(
                renderdir,
                '{:04d}.png'.format(i)
            ))
            render_img = (render_img / 255.).astype(np.float32)

            render_mask = np.zeros(render_img.shape[:2])
            render_mask[render_img[..., -1] > 0.5] = 1.
            render_mask = render_mask[..., np.newaxis].repeat(3, axis=-1)

            img = gt_img[..., :3] * (1. - render_mask) + render_img[..., :3] * render_mask

            imageio.imwrite(
                os.path.join(outdir, '{:04d}.png'.format(i)),
                (255. * np.clip(img, 0, 1)).astype(np.uint8)
            )
        self._image_sequence_to_video(outdir)
        self._image_sequence_to_gif(outdir)

    def _render_mesh_sequence(
        self,
        meshdir,
        poses,
        outdir,
        color=False
    ):
        for i in tqdm(range(len(poses))):
            self._render_mesh(meshdir, i, poses, outdir, color)
        self._image_sequence_to_video(outdir)
        self._image_sequence_to_gif(outdir)

    def _render_mesh(
        self,
        meshdir,
        i,
        poses,
        outdir,
        color
    ):
        import pyrender
        from scipy.spatial.transform import Rotation as R

        intrin = self.dataset.intrinsics # assumes fixed intrinsics

        scene = pyrender.Scene(
            bg_color=np.array([0., 0., 0., 0.]),
            ambient_light=np.array([.008, .008, .008])
        )

        meshfilenames = [
            '{:04d}.ply'.format(i),         # prediction output; 0-indexed
            'mesh_{:06d}.obj'.format(i+1)   # GT and proxies; 1-indexed
        ]
        for meshfilename in meshfilenames:
            meshfilepath = os.path.join(meshdir, meshfilename)
            if os.path.isfile(meshfilepath):
                break

        try:
            if not color:
                tri = trimesh.load(meshfilepath)
                if type(tri) == trimesh.base.Trimesh:
                    # Default material makes geometry easier to see
                    mat = pyrender.MetallicRoughnessMaterial()
                    mesh = pyrender.Mesh.from_trimesh(
                        tri,
                        material=mat
                    )
                    scene.add(mesh)
                elif type(tri) == trimesh.points.PointCloud:
                    import matplotlib
                    import matplotlib.cm as cm
                    # Each joint should have a different, but sequentially stable, color
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(tri.vertices), clip=True)
                    mapper = cm.ScalarMappable(norm, cm.gist_rainbow)
                    for e, v in enumerate(tri.vertices):
                        sphere = trimesh.creation.uv_sphere(radius=.05, count=[8,8])
                        sphere.visual.vertex_colors = mapper.to_rgba(e)[:3]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[0, :3, 3] = v
                        mesh = pyrender.Mesh.from_trimesh(
                            sphere,
                            poses=tfs
                        )
                        scene.add(mesh)
                else:
                    raise RuntimeError('invalid trimesh object loaded')
            else:
                # Use the per-vertex colors from the file
                mesh = pyrender.Mesh.from_trimesh(trimesh.load(meshfilepath))
        except Exception as e:
            print('render error:', e)
            mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh())

        camera = pyrender.PerspectiveCamera(
            yfov=(2. * math.atan(intrin['height'] / (2. * intrin['focal_y']))),
            aspectRatio=(intrin['width'] / intrin['height'])
        )
        scene.add(camera, pose=poses[i])

        front_light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=.25
        )
        scene.add(front_light, pose=poses[i])

        # Cross lighting for detail, warm/cool if not using vertex colors
        warm_light = pyrender.DirectionalLight(
            color=(np.array([1., 0.8, 0.75]) if not color else np.array([1., 1., 1.])),
            intensity=3.
        )
        warm_light_R = R.from_euler('y', -60, degrees=True)
        warm_light_pose = np.copy(poses[i])
        warm_light_pose[:3, :3] = np.matmul(warm_light_R.as_matrix(), warm_light_pose[:3, :3])
        scene.add(warm_light, pose=warm_light_pose)
        cool_light = pyrender.DirectionalLight(
            color=(np.array([0.75, 0.8, 1.]) if not color else np.array([1., 1., 1.])),
            intensity=.75
        )
        cool_light_R = R.from_euler('y', 45, degrees=True)
        cool_light_pose = np.copy(poses[i])
        cool_light_pose[:3, :3] = np.matmul(cool_light_R.as_matrix(), cool_light_pose[:3, :3])
        scene.add(cool_light, pose=cool_light_pose)

        render_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        r = pyrender.OffscreenRenderer(intrin['width'], intrin['height'])
        render, _ = r.render(scene, flags=render_flags)

        imageio.imwrite(
            os.path.join(outdir, '{:04d}.png'.format(i)),
            render
        )

    def _image_sequence_to_video(
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

    def _image_sequence_to_gif(
        self,
        imgdir,
        name='vid',
        framerate=10
    ):
        ffmpeg_boilerplate = '-y -f image2 -framerate {}'.format(framerate)
        ffmpeg_boilerplate += ' -hide_banner -loglevel error'
        # Palette gives better quality results
        ffmpeg_filter = '-vf \"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\"'
        cmd = 'ffmpeg {} -i {}/%04d.png {} -loop 0 {}'.format(
            ffmpeg_boilerplate,
            imgdir,
            ffmpeg_filter,
            os.path.join(imgdir, '{}.gif'.format(name))
        )
        print(cmd)
        os.system(cmd)

    def _metric_a_to_b(
        self,
        a_mesh,
        b_mesh,
        num_verts=-1,       # -1 uses all vertices and gives reproducible number
        batch_size=64*1024, # some high rez meshes need a batching
        return_dists=False  # return the distances to b_mesh for each vertex of a_mesh
    ):
        metric = 0.
        try:
            a_verts = a_mesh.vertices
            if return_dists:
                distances = np.empty(len(a_verts))
            if num_verts > 0:
                sample = np.random.choice(len(a_verts), size=num_verts, replace=False)
                a_verts = a_verts[sample]
            for i in range(0, len(a_verts), batch_size): # batching required for large meshes
                _, distances_a_to_b, _ = prox.closest_point(b_mesh, a_verts[i:(i+batch_size)])
                if return_dists:
                    distances[i:(i+batch_size)] = distances_a_to_b
            metric += np.sum(np.power(np.abs(distances_a_to_b), 2.))
        except BaseException as err:
            # This can occur if no geometry is produced
            # e.g. when ablating L_FLO and model is "switching" active canonical copy
            print('metric_a_to_b:error', err)
            if return_dists:
                return np.Inf, np.Inf
            else:
                return np.Inf
        # Per-vertex error
        if return_dists:
            return metric / float(len(a_verts)), distances
        else:
            return metric / float(len(a_verts))

    def _compute_metrics(
        self,
        estdir,
        gtdir
    ):
        with open(os.path.join(estdir, 'metrics.csv'), 'w') as w:
            w.write('idx,chamfer,est_to_gt,gt_to_est\n')
            est_meshes = sorted([f for f in os.listdir(estdir) if '.ply' in f])
            metrics = []
            dist_to_gt = [] # used for error coloring
            for m in tqdm(est_meshes):
                zero_idx = int(m.replace('.ply', ''))
                gt_mesh_filename = 'mesh_{:06d}.obj'.format(zero_idx + 1)
                gt_mesh = trimesh.load(os.path.join(gtdir, gt_mesh_filename))
                est_mesh = trimesh.load(os.path.join(estdir, m))
                est_to_gt, dists = self._metric_a_to_b(est_mesh, gt_mesh, return_dists=True)
                dist_to_gt.append(dists)
                gt_to_est = self._metric_a_to_b(gt_mesh, est_mesh)
                chamfer = gt_to_est + est_to_gt
                w.write('{},{:.6f},{:.6f},{:.6f}\n'.format(
                    zero_idx,
                    chamfer,
                    est_to_gt,
                    gt_to_est # yes, this is redundant
                ))
                w.flush()
                metrics.append(np.array([chamfer, est_to_gt, gt_to_est]))

        metrics = np.array(metrics)
        np.save(os.path.join(estdir, 'metrics.npy'), metrics)

        with open(os.path.join(estdir, 'summary.txt'), 'w') as w:
            # Collate metrics
            def write_triple(w, t):
                w.write(' chamfer = {:.6f}\n'.format(t[0]))
                w.write(' est_to_gt = {:.6f}\n'.format(t[1]))
                w.write(' gt_to_est = {:.6f}\n'.format(t[2])) # yes, this is redundant
            avg_metrics = np.average(metrics, axis=0)
            w.write('averages\n')
            write_triple(w, avg_metrics)
            w.flush()
            max_metrics = np.max(metrics, axis=0)
            w.write('max\n')
            write_triple(w, max_metrics)
            w.flush()
            std_metrics = np.std(metrics, axis=0)
            w.write('std\n')
            write_triple(w, std_metrics)
            w.flush()
            # Infinities occur for failure frames (i.e. no geometry produced by method)
            if np.any(np.isinf(metrics)):
                w.write('\nINFINITY ENCOUNTERED: computing masked results\n')
                avg_metrics = np.ma.average(np.ma.masked_array(
                    metrics,
                    np.isinf(metrics)
                ), axis=0)
                w.write('masked averages\n')
                write_triple(w, avg_metrics)
                w.flush()
                max_metrics = np.ma.max(np.ma.masked_array(
                    metrics,
                    np.isinf(metrics)
                ), axis=0)
                w.write('masked max\n')
                write_triple(w, max_metrics)
                w.flush()
                std_metrics = np.ma.std(np.ma.masked_array(
                    metrics,
                    np.isinf(metrics)
                ), axis=0)
                w.write('masked std\n')
                write_triple(w, std_metrics)
                w.flush()
            else:
                w.write('\nno infinite metric; masked results equal\n')
        return dist_to_gt

    def _error_color(
        self,
        estdir,
        errors
    ):
        import matplotlib
        import matplotlib.cm as cm

        # We want a consistent coloring over all meshes in the sequence
        all_errors = np.hstack(errors)
        # Coloring based solely on maximum gives outliers a strong influence (results in really blue meshes)
        maximum = np.ma.max(np.ma.masked_array(
            all_errors,
            np.isinf(all_errors)
        ))
        average = np.ma.average(np.ma.mask_cols(
            all_errors,
            np.isinf(all_errors)
        ))
        full_red = (maximum + average) / 2. # therefore midway between average and maximum
        norm = matplotlib.colors.Normalize(vmin=0, vmax=full_red, clip=True)
        mapper = cm.ScalarMappable(norm, cm.seismic)

        outdir = os.path.join(estdir, 'errors')
        os.makedirs(outdir, exist_ok=True)

        est_meshes = sorted([f for f in os.listdir(estdir) if '.ply' in f])
        for m in est_meshes:
            zero_idx = int(m.replace('.ply', ''))
            try:
                in_mesh = trimesh.load(os.path.join(estdir, m))
                out_mesh = trimesh.Trimesh(
                    in_mesh.vertices,
                    in_mesh.faces,
                    vertex_colors=mapper.to_rgba(errors[zero_idx])[..., :3]
                )
            except Exception as e:
                # Handles ablation cases where no geometry is produced
                print('_error_color:error', e)
                out_mesh = trimesh.Trimesh()
            out_mesh.export(os.path.join(outdir, m))

if __name__ == "__main__":
    print('Hello Solid') # very important!

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logging.getLogger('PIL').setLevel(logging.WARNING) # avoids excessive logging by PIL::PngImagePlugin.py

    args = ParseArgs(post=True)

    # Allow for running on systems without a GPU
    torch.set_default_tensor_type(
        'torch.cuda.FloatTensor' if torch.cuda.is_available() else torch.FloatTensor
    )

    post = PostProcessor(args.conf, args.case)

    if not args.skip_latents:
        try:
            post.analyze_latents(args.latent_animation)
        except Exception as e: # continue without latent analysis
            print('post.analyze_latents:error', e)

    if not args.skip_processing:
        try:
            post.process()
        except Exception as e: # continue without processing
            print('post.process:error', e)

    if not args.skip_render:
        post.render(
            args.render_gt,
            args.render_proxies,
            args.render_novel,
            args.render_view
        )
