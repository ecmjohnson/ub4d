import os, shutil, logging, random
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pyhocon import ConfigFactory
from utils.args import ParseArgs
from utils.dataset import Dataset, NonrigidDataset
from utils.fields import BendingNetwork, RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from utils.renderer import NeuSRenderer
from utils.PLYWriter import PLYWriter


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_data_dir = self.conf['dataset.data_dir']
        if not os.path.isdir(self.base_data_dir):
            raise RuntimeError('please verify data directory is correct!')
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        # Reproducibility
        seed = self.conf.get_int('train.random_seed', default=-1)
        if seed >= 0:
            print('Seeding RNGs does not guarantee reproducibility!') # see https://pytorch.org/docs/stable/notes/randomness.html
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # torch.use_deterministic_algorithms(True) # requires setting an environment variable
            # Seed all possible sources of random numbers
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Load the dataset
        self.dataset = None
        self.nonrigid_data = False
        self.template_frames = -1
        if 'dataset.data_type' in self.conf:
            if self.conf['dataset.data_type'] == 'nonrigid':
                self.dataset = NonrigidDataset(self.conf['dataset'])
                self.nonrigid_data = True
                self.template_frames = self.dataset.template_frames
        if self.dataset is None:
            self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.val_mesh_seq_freq = self.conf.get_int('train.val_mesh_seq_freq', default=-1) # never happen by default!
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.colour_weight = self.conf.get_float('train.colour_weight', default=1.0)
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.offset_weight = self.conf.get_float('train.offset_weight', default=0.0)
        self.divergence_weight = self.conf.get_float('train.divergence_weight', default=0.0)
        self.flow_weight = self.conf.get_float('train.flow_weight', default=0.0)
        # Use an increasing schedule for the bending losses: both offset and div (exponential 1/100)
        self.bending_increasing = self.conf.get_bool('train.bending_increasing', default=True)
        # Use a decreasing schedule for the flow loss (exponential 1/10)
        self.flow_decreasing = self.conf.get_bool('train.flow_decreasing', default=True)
        # Only sample frames at most self.flow_radius from current frame (entire sequence if -1)
        self.flow_radius = self.conf.get_int('train.flow_radius', default=-1)
        # Only sample deforming frames (ie. frames > self.template_frames) if true
        self.flow_deforming = self.conf.get_bool('train.flow_deforming', default=False)
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.bending_network = None
        if 'model.bending_network' in self.conf:
            if self.conf.get_bool('model.bending_network.zero_init', default=True):
                self.bending_latents_list = [
                    torch.zeros(self.conf.get_int('model.bending_network.latent_dim'))
                    for _ in range(self.dataset.n_images)
                ]
            else:
                self.bending_latents_list = [
                    torch.randn(self.conf.get_int('model.bending_network.latent_dim'))
                    for _ in range(self.dataset.n_images)
                ]
            for latent in self.bending_latents_list:
                latent.requires_grad = True
            self.bending_network = BendingNetwork(
                self.bending_latents_list,
                **self.conf['model.bending_network'],
                template_frames=self.template_frames
            ).to(self.device)
            params_to_train += self.bending_latents_list
            params_to_train += list(self.bending_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(
            self.nerf_outside,
            self.bending_network,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            **self.conf['model.neus_renderer']
        )

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        self.save_checkpoint() # save initialized state

        for iter_i in tqdm(range(res_step)):
            frame = image_perm[self.iter_step % len(image_perm)]
            data = self.dataset.gen_random_rays_at(frame, self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(
                rays_o,
                rays_d,
                near,
                far,
                frame,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio()
            )

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss * self.colour_weight \
                   + eikonal_loss * self.igr_weight \
                   + mask_loss * self.mask_weight
            
            if self.bending_network is not None:
                # Add the bending regularizer losses
                offset_loss = render_out['offset_loss']
                divergence_loss = render_out['divergence_loss']
                bending_losses = offset_loss * self.offset_weight \
                                 + divergence_loss * self.divergence_weight
                self.writer.add_scalar('Loss/offset_loss', offset_loss, self.iter_step)
                self.writer.add_scalar('Loss/divergence_loss', divergence_loss, self.iter_step)
                if self.bending_increasing:
                    # Increasing schedule for bending losses as per NR-NeRF
                    bending_losses *= ((1. / 100.) ** (1. - (self.iter_step / self.end_iter)))
                loss += bending_losses

                if self.flow_weight > 0.:
                    if not self.nonrigid_data:
                        raise RuntimeError('cannot compute flow loss without mesh prior!')
                    # Add the bending scene flow loss
                    flow_loss = self.bending_network.compute_flow_loss(
                        frame,
                        self.dataset.verts,
                        self.batch_size,
                        self.dataset.scene_min_pt,
                        self.dataset.scene_max_pt,
                        self.dataset.interp_scale,
                        self.dataset.falloff_scale,
                        frame_radius=self.flow_radius,
                        template_frame=(self.template_frames if self.flow_deforming else -1)
                    )
                    self.writer.add_scalar('Loss/flow_loss', flow_loss, self.iter_step)
                    if self.flow_decreasing:
                        # Decreasing schedule for flow loss
                        flow_loss *= ((1. / 10.) ** (self.iter_step / self.end_iter))
                    loss += flow_loss * self.flow_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            if self.val_mesh_seq_freq > 0 and self.iter_step % self.val_mesh_seq_freq == 0:
                self.mesh_sequence(
                    resolution=1024,
                    full_scene=True,
                    frustum_cull=True
                )

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    shutil.copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        shutil.copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        if self.bending_network is not None:
            self.bending_network.load_state_dict(checkpoint['bending_network'])
            # This is how NR-NeRF handles loading latent codes
            for latent, saved_latent in zip(
                self.bending_latents_list, checkpoint['bending_latents']
            ):
                latent.data[:] = saved_latent[:].detach().clone()

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        if self.bending_network is not None:
            checkpoint['bending_network'] = self.bending_network.state_dict()
            # This is how NR-NeRF handles saving latent codes
            all_latents = torch.zeros(0).cpu()
            for l in self.bending_latents_list:
                all_latents = torch.cat([all_latents, l.cpu().unsqueeze(0)], 0)
            checkpoint['bending_latents'] = all_latents # shape: n_images, latent_dim

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(
        self,
        idx=-1,
        resolution_level=-1,
        novel=False,        # use novel camera pose
        sequence=False      # put results in dedicated subdir
    ):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, novel=novel)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_pointcloud_fine = []
        out_mask_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              idx,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('frame_points') and feasible('weights'):
                out_pointcloud_fine.append(render_out['frame_points'].detach().cpu().numpy())
                out_mask_fine.append(torch.sum(render_out['weights'], dim=-1).detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        pointcloud = None
        mask = None
        if len(out_pointcloud_fine) > 0:
            pointcloud = np.concatenate(out_pointcloud_fine, axis=0).reshape(H * W, 3, -1)
            mask = np.concatenate(out_mask_fine, axis=0).reshape(H * W, -1)

        # Create a subdirectory for any novel or sequence validations
        subdir = ''
        if novel:
            subdir += 'novel_{:08d}'.format(self.iter_step)
        if sequence:
            subdir += 'sequence_{:08d}'.format(self.iter_step)

        val_dir = os.path.join(self.base_exp_dir, subdir, 'validations_fine')
        os.makedirs(val_dir, exist_ok=True)
        normal_dir = os.path.join(self.base_exp_dir, subdir, 'normals')
        os.makedirs(normal_dir, exist_ok=True)
        pc_dir = os.path.join(self.base_exp_dir, subdir, 'pointclouds')
        os.makedirs(pc_dir, exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if novel or sequence:
                outfile = '{:04d}'.format(idx)
            else:
                outfile = '{:0>8d}_{}_{}'.format(self.iter_step, i, idx)
            if len(out_rgb_fine) > 0:
                val = img_fine[..., i]
                # Nonrigid datasets use RGB byte ordering
                cv.imwrite(
                    os.path.join(
                        val_dir,
                        outfile + '.png'
                    ),
                    cv.cvtColor(val, cv.COLOR_RGB2BGR) if self.nonrigid_data else val
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        normal_dir,
                        outfile + '.png'
                    ),
                    normal_img[..., i]
                )
            if len(out_pointcloud_fine) > 0:
                colours = img_fine[..., i].reshape(H * W, 3)
                filename = os.path.join(
                    pc_dir,
                    outfile + '.ply'
                )
                def write_pointcloud(filename, mask_threshold=0.95):
                    with PLYWriter(filename, hasColours=True) as w:
                        for p in range(pointcloud.shape[0]):
                            # Only save points with a sufficient mask weight
                            if mask[p] > mask_threshold:
                                w.addPoint(pointcloud[p, :, i], colour=colours[p, :])
                write_pointcloud(filename)

    def render_novel_sequence(self):
        for i in tqdm(range(self.dataset.n_images)):
            self.validate_image(i, novel=True, resolution_level=1)

    def validate_sequence(self):
        for i in tqdm(range(self.dataset.n_images)):
            self.validate_image(i, sequence=True, resolution_level=1)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            # Uses first camera's timestep for bending network
            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                idx_0,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb
            )

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def create_unique_folder(
        self,
        pattern
    ):
        c = 0
        folderpath = pattern.format(c)
        while os.path.isdir(folderpath):
            c += 1
            folderpath = pattern.format(c)
        os.makedirs(folderpath)
        return folderpath

    def create_unique_file(
        self,
        pattern
    ):
        c = 0
        filepath = pattern.format(c)
        while os.path.isfile(filepath):
            c += 1
            filepath = pattern.format(c)
        return filepath

    def write_custom_bounds(
        self,
        root,
        bound_min,
        bound_max
    ):
        min_filepath = os.path.join(root, 'bound_min.txt')
        max_filepath = os.path.join(root, 'bound_max.txt')
        np.savetxt(min_filepath, self.dataset.scene_scale * bound_min.cpu().numpy() + self.dataset.scene_center, fmt='%.6f')
        np.savetxt(max_filepath, self.dataset.scene_scale * bound_max.cpu().numpy() + self.dataset.scene_center, fmt='%.6f')

    def read_custom_bounds(
        self,
        root
    ):
        min_filepath = os.path.join(root, 'bound_min.txt')
        max_filepath = os.path.join(root, 'bound_max.txt')
        bound_min = torch.tensor(np.loadtxt(min_filepath), dtype=torch.float32)
        bound_max = torch.tensor(np.loadtxt(max_filepath), dtype=torch.float32)
        device = bound_min.get_device()
        bound_min = (bound_min - torch.tensor(self.dataset.scene_center).to(device=device)) / self.dataset.scene_scale
        bound_max = (bound_max - torch.tensor(self.dataset.scene_center).to(device=device)) / self.dataset.scene_scale
        return bound_min, bound_max

    def get_proj_matrix(
        self,
        frame
    ):
        cam_pose = self.dataset.poses[frame]
        if cam_pose.shape[0] != cam_pose.shape[1]:
            cam_pose = torch.vstack((cam_pose, torch.tensor([0., 0., 0., 1.])))
        orientation = torch.eye(4)
        orientation[1, 1] = -1
        orientation[2, 2] = -1
        E = torch.linalg.inv(torch.matmul(cam_pose, orientation))
        K = torch.eye(4)[:3, :4]
        try:
            K[0, 0] = self.dataset.intrinsics['focal_x']
            K[1, 1] = self.dataset.intrinsics['focal_y']
            K[0, 2] = self.dataset.intrinsics['center_x']
            K[1, 2] = self.dataset.intrinsics['center_y']
        except TypeError:
            K[0, 0] = torch.tensor(self.dataset.intrinsics['focal_x']).cuda()
            K[1, 1] = torch.tensor(self.dataset.intrinsics['focal_y']).cuda()
            K[0, 2] = torch.tensor(self.dataset.intrinsics['center_x']).cuda()
            K[1, 2] = torch.tensor(self.dataset.intrinsics['center_y']).cuda()
        return torch.matmul(K, E)

    def frustum_culling(
        self,
        pts,
        proj_matrix,
        H,
        W
    ):
        hext = torch.ones((*pts.shape[:-1], 1))
        proj = torch.matmul(
            proj_matrix[None, ...].repeat_interleave(len(pts), dim=0),
            torch.hstack((pts, hext))[..., None]
        ).squeeze(dim=-1)
        proj[..., 0] = proj[..., 0] / proj[..., -1]
        proj[..., 1] = proj[..., 1] / proj[..., -1]
        inside = torch.ones(proj.shape[:-1], dtype=torch.bool)
        xy = torch.cat((proj[..., 0][None, ...], proj[..., 1][None, ...]))
        mins, _ = torch.min(xy, dim=0)
        inside[mins < 0] = False
        inside[proj[..., 0] >= W] = False
        inside[proj[..., 1] >= H] = False
        inside[proj[..., -1] < 0] = False
        return ~inside

    def validate_mesh(
        self,
        world_space=False,
        resolution=64,
        threshold=0.0,
        full_scene=False,
        custom=False,
        scale_to_scene=True,
        frame=-1,
        frustum_cull=False
    ):
        if full_scene and custom:
            print('full_scene taking priority over custom!')
        if full_scene:
            bound_min = torch.tensor(np.array([-1.01, -1.01, -1.01]), dtype=torch.float32)
            bound_max = torch.tensor(np.array([1.01, 1.01, 1.01]), dtype=torch.float32)
        elif custom:
            bound_min, bound_max = self.read_custom_bounds(self.base_data_dir)
        else:
            bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
            bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        print('bound_min', bound_min)
        print('bound_max', bound_max)

        if frustum_cull and frame >= 0:
            proj_matrix = self.get_proj_matrix(frame)
            def cull_func(pts, H=self.dataset.H, W=self.dataset.W):
                return self.frustum_culling(pts, proj_matrix, H, W)
        elif frustum_cull:
            print('need frame if frustum culling!')

        vertices, triangles, colors = self.renderer.extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            frame=frame,
            color=True,
            cull_func=(None if not frustum_cull else cull_func)
        )
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        elif scale_to_scene:
            vertices = self.dataset.scene_scale * vertices + self.dataset.scene_center

        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors, validate=True)
        if full_scene:
            folderpath = os.path.join(self.base_exp_dir, 'meshes', 'full_scene')
            os.makedirs(folderpath, exist_ok=True)
            filepath = os.path.join(folderpath, '{:0>8d}.ply'.format(self.iter_step))
        elif custom:
            folderpath = self.create_unique_folder(os.path.join(self.base_exp_dir, 'meshes', 'custom_{}'))
            self.write_custom_bounds(folderpath, bound_min, bound_max)
            filepath = os.path.join(folderpath, '{:0>8d}.ply'.format(self.iter_step))
        else:
            filepath = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step))
        mesh.export(filepath)

        logging.info('End')

    def mesh_sequence(
        self,
        world_space=False,
        resolution=64,
        threshold=0.0,
        full_scene=False,
        custom=False,
        scale_to_scene=True,
        frustum_cull=False,
        start_frame=0
    ):
        if full_scene and custom:
            print('full_scene taking priority over custom!')
        if full_scene:
            bound_min = torch.tensor(np.array([-1.01, -1.01, -1.01]), dtype=torch.float32)
            bound_max = torch.tensor(np.array([1.01, 1.01, 1.01]), dtype=torch.float32)
        elif custom:
            bound_min, bound_max = self.read_custom_bounds(self.base_data_dir)
        else:
            bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
            bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        print('bound_min', bound_min)
        print('bound_max', bound_max)

        if full_scene:
            meshes_dir = os.path.join(self.base_exp_dir, 'full_scene_{:0>8d}'.format(self.iter_step))
            os.makedirs(meshes_dir, exist_ok=True)
        elif custom:
            meshes_dir = self.create_unique_folder(os.path.join(self.base_exp_dir, 'custom_{:0>8d}'.format(self.iter_step) + '_{}'))
            self.write_custom_bounds(meshes_dir, bound_min, bound_max)
        else:
            meshes_dir = os.path.join(self.base_exp_dir, 'meshes_{:0>8d}'.format(self.iter_step))
            os.makedirs(meshes_dir, exist_ok=True)

        for f in tqdm(range(start_frame, self.dataset.n_images)):

            if frustum_cull:
                proj_matrix = self.get_proj_matrix(f)
                def cull_func(pts, H=self.dataset.H, W=self.dataset.W):
                    return self.frustum_culling(pts, proj_matrix, H, W)

            vertices, triangles, colors = self.renderer.extract_geometry(
                bound_min,
                bound_max,
                resolution=resolution,
                threshold=threshold,
                frame=f,
                color=True,
                cull_func=(None if not frustum_cull else cull_func)
            )

            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            elif scale_to_scene:
                vertices = self.dataset.scene_scale * vertices + self.dataset.scene_center
            mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors, validate=True)
            mesh.export(os.path.join(meshes_dir, '{:0>4d}.ply'.format(f)))

        logging.info('End')

    def latents_sequence(
        self,
        latents_file='',
        world_space=False,
        resolution=64,
        threshold=0.0,
        full_scene=False,
        custom=False,
        scale_to_scene=True,
        start_frame=0
    ):
        if full_scene and custom:
            print('full_scene taking priority over custom!')
        if full_scene:
            bound_min = torch.tensor(np.array([-1.01, -1.01, -1.01]), dtype=torch.float32)
            bound_max = torch.tensor(np.array([1.01, 1.01, 1.01]), dtype=torch.float32)
        elif custom:
            bound_min, bound_max = self.read_custom_bounds(self.base_data_dir)
        else:
            bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
            bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        print('bound_min', bound_min)
        print('bound_max', bound_max)

        latents_name = os.path.basename(latents_file).split('.')[0]
        print('latents_name', latents_name)
        if full_scene:
            meshes_dir = self.create_unique_folder(os.path.join(self.base_exp_dir, latents_name + '_{}'))
        elif custom:
            meshes_dir = self.create_unique_folder(os.path.join(self.base_exp_dir, latents_name + '_{}'))
            self.write_custom_bounds(meshes_dir, bound_min, bound_max)
        else:
            meshes_dir = self.create_unique_folder(os.path.join(self.base_exp_dir, latents_name + '_{}'))

        # Modify the bending network and dataset to match the latents provided
        latents = np.load(latents_file)
        d_latents = latents.shape[-1]
        self.dataset.n_images = len(latents)
        self.bending_network.n_frames = len(latents)
        assert d_latents == self.bending_network.ray_bending_latent_size, 'network must have been trained with same latent size!'
        latent_list = [torch.tensor(latent, dtype=torch.float32).cuda() for latent in latents]
        self.bending_network.latents = latent_list

        # Backup the latents used into the meshes directory
        shutil.copyfile(latents_file, os.path.join(meshes_dir, os.path.basename(latents_file)))

        for f in tqdm(range(start_frame, self.dataset.n_images)):

            vertices, triangles, colors = self.renderer.extract_geometry(
                bound_min,
                bound_max,
                resolution=resolution,
                threshold=threshold,
                frame=f,
                color=True,
                cull_func=None
            )

            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            elif scale_to_scene:
                vertices = self.dataset.scene_scale * vertices + self.dataset.scene_center
            mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors, validate=True)
            mesh.export(os.path.join(meshes_dir, '{:0>4d}.ply'.format(f)))

        logging.info('End')

    def animate_canonical(
        self,
        world_space=False,
        resolution=64,
        threshold=0.0,
        full_scene=False,
        custom=False,
        scale_to_scene=True
    ):
        # Get the canonical space mesh
        if full_scene and custom:
            print('full_scene taking priority over custom!')
        if full_scene:
            bound_min = torch.tensor(np.array([-1.01, -1.01, -1.01]), dtype=torch.float32)
            bound_max = torch.tensor(np.array([1.01, 1.01, 1.01]), dtype=torch.float32)
        elif custom:
            bound_min, bound_max = self.read_custom_bounds(self.base_data_dir)
        else:
            bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
            bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        print('bound_min', bound_min)
        print('bound_max', bound_max)

        if full_scene:
            meshes_dir = os.path.join(self.base_exp_dir, 'animfull_scene_{:0>8d}'.format(self.iter_step))
            os.makedirs(meshes_dir, exist_ok=True)
        elif custom:
            meshes_dir = self.create_unique_folder(os.path.join(self.base_exp_dir, 'animcust_{:0>8d}'.format(self.iter_step) + '_{}'))
            self.write_custom_bounds(meshes_dir, bound_min, bound_max)
        else:
            meshes_dir = os.path.join(self.base_exp_dir, 'animated_{:0>8d}'.format(self.iter_step))
            os.makedirs(meshes_dir, exist_ok=True)

        # Mesh triangles and vertex colors remain fixed throughout sequence
        canonical_vertices, triangles, colors = self.renderer.extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            color=True
        )

        # Deform vertex positions for every frame
        vertices_list = self.renderer.deform_geometry(
            canonical_vertices
        )

        for f, vertices in enumerate(vertices_list):
            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            elif scale_to_scene:
                vertices = self.dataset.scene_scale * vertices + self.dataset.scene_center
            mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors, validate=True)
            mesh.export(os.path.join(meshes_dir, '{:0>4d}.ply'.format(f)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def validate_sceneflow(self,
                           f1,
                           f2,
                           n_samples=2048,
                           n_rays=2,
                           n_ray_pts=64,
                           scale_to_scene=True):
        sceneflow_dir = os.path.join(self.base_exp_dir, 'sceneflow_{:0>8d}'.format(self.iter_step))
        os.makedirs(sceneflow_dir, exist_ok=True)

        rays_f1_o, _ = self.dataset.gen_rays_at(f1) # pinhole camera: all rays have same origin
        rays_f2_o, _ = self.dataset.gen_rays_at(f2)
        rays_o = np.vstack(( # half of rays from each camera
            np.repeat(rays_f1_o[0, 0][None, ...].cpu().numpy(), n_rays//2, axis=0),
            np.repeat(rays_f2_o[0, 0][None, ...].cpu().numpy(), n_rays - n_rays//2, axis=0)
        )) # n_rays x 3
        targets_i = np.random.randint(len(self.dataset.verts[f1]), size=n_rays) # randomly target a vertex
        ray_targets = self.dataset.verts[f1][targets_i]
        rays_d = ray_targets - rays_o
        rays_d = rays_d / np.repeat(np.linalg.norm(rays_d)[..., None], 3, axis=-1) # n_rays x 3
        near, far = self.dataset.near_far_from_sphere(torch.Tensor(rays_o), torch.Tensor(rays_d))
        near, far = near.cpu().numpy().min(axis=0), far.cpu().numpy().max(axis=0)
        rays_z = np.repeat(
            near + (far - near) * np.linspace(0., 2., n_ray_pts)[..., None],
            3,
            axis=-1
        ) # n_ray_pts x 3
        ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * np.repeat(rays_z[None, ...], n_rays, axis=0)
        ray_pts = np.reshape(ray_pts, (-1, 3))
        ray_sf_vecs = self.bending_network.extrapolate_scene_flow(
            torch.Tensor(ray_pts).cuda(),
            self.dataset.verts,
            f1,
            f2,
            self.dataset.interp_scale,
            self.dataset.falloff_scale
        ).cpu().numpy()
        if scale_to_scene:
            ray_pts = self.dataset.scene_scale * ray_pts + self.dataset.scene_center
            ray_sf_vecs = self.dataset.scene_scale * ray_sf_vecs # vectors don't need offset!
        ray_name = 'ray_{}_to_{}'.format(f1, f2)
        ray_file = self.create_unique_file(os.path.join(sceneflow_dir, ray_name + '_{}.ply'))
        with PLYWriter(ray_file, hasColours=True, hasEdges=True) as f:
            for i in range(n_rays * n_ray_pts):
                f.addPoint(ray_pts[i], colour=np.array([0,255,0]))
            for i in range(n_rays * n_ray_pts):
                f.addPoint(ray_pts[i] + ray_sf_vecs[i], colour=np.array([255,0,0]))
            for i in range(n_rays * n_ray_pts):
                f.addEdge(i, i + n_rays * n_ray_pts)

        details = {
            'f1': f1,
            'f2': f2
        }
        self.bending_network.compute_flow_loss(
            f1,
            self.dataset.verts,
            n_samples,
            self.dataset.scene_min_pt,
            self.dataset.scene_max_pt,
            self.dataset.interp_scale,
            self.dataset.falloff_scale,
            details=details
        )

        pts_f1 = details['pts']
        pts_f2 = details['pts'] + details['scene_flow_vecs']
        if scale_to_scene:
            pts_f1 = self.dataset.scene_scale * pts_f1 + self.dataset.scene_center
            pts_f2 = self.dataset.scene_scale * pts_f2 + self.dataset.scene_center
        flowname = 'sf_{}_to_{}'.format(f1, f2)
        sceneflow_file = self.create_unique_file(os.path.join(sceneflow_dir, flowname + '_{}.ply'))
        with PLYWriter(sceneflow_file, hasColours=True, hasEdges=True) as f:
            for i in range(n_samples):
                f.addPoint(pts_f1[i], colour=np.array([0,255,0]))
            for i in range(n_samples):
                f.addPoint(pts_f2[i], colour=np.array([255,0,0]))
            for i in range(n_samples):
                f.addEdge(i, i + n_samples)

        gt_f1 = details['proxy_pts']
        n_vertices = len(gt_f1)
        gt_f2 = details['proxy_pts'] + details['proxy_flow_vecs']
        if scale_to_scene:
            gt_f1 = self.dataset.scene_scale * gt_f1 + self.dataset.scene_center
            gt_f2 = self.dataset.scene_scale * gt_f2 + self.dataset.scene_center
        gt_name = 'gt_{}_to_{}'.format(f1, f2)
        gt_file = self.create_unique_file(os.path.join(sceneflow_dir, gt_name + '_{}.ply'))
        with PLYWriter(gt_file, hasColours=True, hasEdges=True) as f:
            for i in range(n_vertices):
                f.addPoint(gt_f1[i], colour=np.array([0,255,0]))
            for i in range(n_vertices):
                f.addPoint(gt_f2[i], colour=np.array([255,0,0]))
            for i in range(n_vertices):
                f.addEdge(i, i + n_vertices)

if __name__ == '__main__':
    print('Hello Wooden') # very important!

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logging.getLogger('PIL').setLevel(logging.WARNING) # avoids excessive logging by PIL::PngImagePlugin.py

    args = ParseArgs()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu)

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        # Train the model
        runner.train()

    elif args.mode == 'validate_image':
        # Render a random validation image
        runner.validate_image()

    elif args.mode == 'validate_sequence':
        # Render a validation sequence
        runner.validate_sequence()

    elif args.mode == 'novel_sequence':
        # Render a novel seqeuence
        runner.render_novel_sequence()

    elif args.mode == 'validate_mesh':
        # Validate a single frame mesh
        runner.validate_mesh(
            resolution=args.mcube_resolution,
            threshold=args.mcube_threshold,
            full_scene=args.full_scene_bounds,
            frame=args.validate_frame,
            custom=args.custom_bounds,
            frustum_cull=args.frustum_cull
        )

    elif args.mode.startswith('interpolate'):
        # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)

    elif args.mode == 'mesh_sequence':
        # Produce a mesh sequence from a trained model
        runner.mesh_sequence(
            resolution=args.mcube_resolution,
            threshold=args.mcube_threshold,
            full_scene=args.full_scene_bounds,
            custom=args.custom_bounds,
            frustum_cull=args.frustum_cull,
            start_frame=args.start_frame
        )

    elif args.mode == 'animate_canonical':
        # Animate the canonical mesh (does *NOT* work for large deformations)
        runner.animate_canonical(
            resolution=args.mcube_resolution,
            threshold=args.mcube_threshold,
            full_scene=args.full_scene_bounds,
            custom=args.custom_bounds
        )

    elif args.mode == 'validate_sceneflow':
        # Validate the extrapolated scene flow between two given frames
        runner.validate_sceneflow(args.frame1, args.frame2)

    elif args.mode == 'latents_sequence':
        # Produce a mesh sequence for given latent codes
        runner.latents_sequence(
            latents_file=args.latents_file,
            resolution=args.mcube_resolution,
            threshold=args.mcube_threshold,
            full_scene=args.full_scene_bounds,
            custom=args.custom_bounds,
            start_frame=args.start_frame
        )

    else:
        raise RuntimeError('we do humbly reqest a verification of the spelling')
