import os
import torch
import cv2 as cv
import numpy as np
import imageio
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from utils.load_blender import *
from utils.load_llff import *
from utils.PLYWriter import PLYWriter


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


# This implementation is borrowed from NR-NeRF: https://github.com/facebookresearch/nonrigid_nerf
class NonrigidDataset:
    def __init__(
        self,
        conf,
        cuda=True,          # False allows post-processing on GPU-less machines
        load_images=True    # False allows post-processing on low-memory machines
    ):
        super(NonrigidDataset, self).__init__()
        print('Load nonrigid data: Begin')
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.nonrigid_type = conf.get_string('nonrigid_type')

        self.white_bkgd = conf.get_bool('white_bkgd')
        self.scale_to_unit_sphere = conf.get_bool('scale_to_unit_sphere')
        self.bbox_expansion = 0.
        if 'bbox_expansion' in conf:
            # Expand min & max point by bbox_expansion/2.*max(extents)
            self.bbox_expansion = conf.get_float('bbox_expansion')
        self.template_frames = -1
        if 'template_frames' in conf:
            self.template_frames = conf.get_int('template_frames')
        
        # Kernel scales used when computing estimated scene flow
        # N.B. larger scales correspond to narrower kernels (inverse relation)
        self.interp_scale = conf.get_float('interp_scale', default=700) # aka extrap_scale
        self.falloff_scale = conf.get_float('falloff_scale', default=75)

        if self.nonrigid_type == 'llff':
            self.images_dir = os.path.join(self.data_dir, 'images')
            self.images, self.poses, bds, self.render_poses, _, n_imgs = load_llff_data(
                self.data_dir,
                factor=1,
                recenter=False,
                rescale=False,
                load_imgs=load_images
            )

            hwf = self.poses[0, :3, -1]
            self.poses = self.poses[:, :3, :4]

            self.near = np.ndarray.min(bds) * 0.9
            self.far = np.ndarray.max(bds) * 1.0

            self.n_images = n_imgs

            print(
                "Loaded llff",
                (self.images.shape if load_images else 'None'),
                self.render_poses.shape,
                hwf,
                self.data_dir
            )

        elif self.nonrigid_type == 'blender':
            self.images_dir = os.path.join(self.data_dir, 'train') # no data split for reconstruction problem
            self.images, self.poses, self.render_poses, hwf, _, clips = load_blender_data(
                self.data_dir,
                load_imgs=load_images
            )

            self.near, self.far = clips
            if self.near == None or self.far == None:
                self.near = 2.; self.far = 6.

            if load_images:
                if self.white_bkgd:
                    self.images = self.images[...,:3]*self.images[...,-1:] + (1. - self.images[...,-1:])
                else:
                    self.images = self.images[...,:3]

            self.n_images = self.poses.shape[0]

            print(
                "Loaded blender",
                (self.images.shape if load_images else 'None'),
                self.render_poses.shape,
                hwf,
                self.data_dir
            )

        else:
            raise RuntimeError('invalid nonrigid_type')

        static_render_pose = self.conf.get_int('static_render_pose', default=-1)
        if static_render_pose > 0:
            self.render_poses = np.copy(self.poses[static_render_pose])
            self.render_poses = self.render_poses[np.newaxis, ...].repeat(len(self.poses), axis=0)

        if not load_images:
            print('No images loaded')

        # N.B. always assuming fixed intrinsics across all images
        if self.nonrigid_type == 'llff' or self.nonrigid_type == 'blender':
            self.H = int(hwf[0])
            self.W = int(hwf[1])
            focal = hwf[2]
            self.intrinsics = {
                'width': self.W,
                'height': self.H,
                'focal_x': focal,
                'focal_y': focal,
                'center_x': self.W / 2,
                'center_y': self.H / 2
            }

        # Scale the scene into a unit sphere (necessary for correctness of positional encoding)
        self.image_pixels = self.H * self.W
        self.scene_center = np.array([0.,0.,0.])
        self.scene_scale = 1.
        self.raw_poses = np.copy(self.poses)
        self.raw_render_poses = np.copy(self.render_poses)
        print('pre-scale: near', self.near, 'far', self.far)
        if self.scale_to_unit_sphere:
            # Transform cameras so that all march points lie in the unit sphere
            cam_positions = self.poses[:, :3, -1] # n_images x 3
            cam_local_forward = np.array([0., 0., -1.]) # -z is forward
            cam_world_forwards = np.array(
                [rotation.dot(cam_local_forward) for rotation in self.poses[:, :3, :3]]
            )
            far_positions = cam_positions + cam_world_forwards * self.far
            # Write pre-transformed cameras to debug file
            unnorm_debug_file = os.path.join(self.data_dir, 'd_unnorm_cams.ply')
            with PLYWriter(unnorm_debug_file, hasColours=True, hasEdges=True) as f:
                for i in range(self.n_images):
                    f.addPoint(cam_positions[i], colour=np.array([0,255,0]))
                for i in range(self.n_images):
                    f.addPoint(far_positions[i], colour=np.array([255,0,0]))
                for i in range(self.n_images):
                    f.addEdge(i, i + self.n_images)
            # Find center of scene (average of extreme positions)
            extreme_positions = np.concatenate((
                cam_positions, far_positions
            ), axis=0)
            self.scene_center = np.mean(extreme_positions, axis=0)
            print('scene center', self.scene_center)
            # Find scale of scene (maximum extreme point distance from center)
            self.scene_scale = np.max(np.linalg.norm(
                extreme_positions - self.scene_center,
                axis=-1
            ))
            print('scene scale', self.scene_scale)
            # Apply the transformation to both the training and render poses
            self.poses[:, :3, -1] = (self.poses[:, :3, -1] - self.scene_center) / self.scene_scale
            self.render_poses[:, :3, -1] = (self.render_poses[:, :3, -1] - self.scene_center) / self.scene_scale
            # Scene should now be in unit sphere
            self.near = 0.; self.far = 2.
            # Write post-transformed cameras to debug file
            norm_cam_positions = self.poses[:, :3, -1]
            norm_far_positions = norm_cam_positions + cam_world_forwards * self.far
            norm_debug_file = os.path.join(self.data_dir, 'd_norm_cams.ply')
            with PLYWriter(norm_debug_file, hasColours=True, hasEdges=True) as f:
                for i in range(self.n_images):
                    f.addPoint(norm_cam_positions[i], colour=np.array([0,255,0]))
                for i in range(self.n_images):
                    f.addPoint(norm_far_positions[i], colour=np.array([255,0,0]))
                for i in range(self.n_images):
                    f.addEdge(i, i + self.n_images)
        # Scene now exists in the unit sphere
        self.scene_min_pt = np.array([-1.01, -1.01, -1.01])
        self.scene_max_pt = np.array([1.01, 1.01, 1.01])

        # Load segmentations if they exist, otherwise all white
        if load_images:
            self.masks = self._load_segmentations(self.data_dir) # shape: n_images, H, W
            if self.masks is not None:
                self.masks = self.masks[..., np.newaxis].repeat(3, axis=-1)
            else:
                print('No segmentations found')
                self.masks = np.ones_like(self.images)

        # Load meshes if they exist
        self.verts, self.faces = self._load_meshes(self.data_dir)
        self.triangles = None
        if self.scale_to_unit_sphere and self.verts is not None:
            # Transform mesh vertices into unit sphere
            self.verts = (self.verts - self.scene_center) / self.scene_scale
            # Write out a transformed mesh for debugging
            norm_mesh_file = os.path.join(self.data_dir, 'd_norm_mesh0.ply')
            with PLYWriter(norm_mesh_file, hasFaces=True) as f:
                for vert in self.verts[0]:
                    f.addPoint(vert)
                for face in self.faces[0]:
                    # .obj uses 1-indexed faces while .ply has 0-indexed faces
                    f.addFace(face[0] - 1, face[1] - 1, face[2] - 1)
        if self.verts is not None:
            self.object_bbox_min = np.min(np.min(self.verts, axis=1), axis=0).astype(np.float32)
            self.object_bbox_max = np.max(np.max(self.verts, axis=1), axis=0).astype(np.float32)
            if self.bbox_expansion > 0.:
                extent = self.object_bbox_max - self.object_bbox_min
                expand = np.max(extent) * self.bbox_expansion / 2.
                self.object_bbox_min -= expand
                self.object_bbox_max += expand
            # Create the triangles structure
            self.triangles = np.empty((self.faces.shape[0], self.faces.shape[1], 3, 3))
            for f in range(self.faces.shape[0]):
                for j in range(self.faces.shape[1]):
                    self.triangles[f][j][0] = self.verts[f, self.faces[f][j][0] - 1, :]
                    self.triangles[f][j][1] = self.verts[f, self.faces[f][j][1] - 1, :]
                    self.triangles[f][j][2] = self.verts[f, self.faces[f][j][2] - 1, :]
        else:
            print('No meshes found')
            # Assuming a unit sphere
            self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
            self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        # Everything should be a torch tensor (some on CPU, some on GPU)
        if load_images:
            self.images = torch.from_numpy(self.images).cpu() # shape: n_images, H, W, 3
            self.masks = torch.from_numpy(self.masks).cpu() # shape: n_images, H, W, 3
        self.poses = torch.from_numpy(self.poses).to(self.device) # shape: n_images, 3, 4
        self.render_poses = torch.from_numpy(self.render_poses).to(self.device) # shape: n_images, 3, 4
        self.pose_all = self.poses

        print('Load nonrigid data: End')

    def gen_rays_at(self, img_idx, resolution_level=1, novel=False):
        """
        Generate rays in world space from one camera.
        """
        l = resolution_level
        intrin = self.intrinsics
        H = intrin['height']
        W = intrin['width']
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W // l, device=self.device),
            torch.linspace(0, H-1, H // l, device=self.device)
        )
        i = i.t()
        j = j.t()
        focal_x = intrin['focal_x']
        focal_y = intrin['focal_y']
        center_x = intrin['center_x']
        center_y = intrin['center_y']
        dirs = torch.stack([
            (i - center_x) / focal_x,
            -(j - center_y) / focal_y,
            -torch.ones_like(i, device=self.device)
        ], -1)
        if novel:
            c2w = self.render_poses[img_idx]
        else:
            c2w = self.poses[img_idx]
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays in world space from one camera.
        """
        intrin = self.intrinsics
        H = intrin['height']
        W = intrin['width']
        pixels_x = torch.randint(low=0, high=W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        rays_o, rays_d = self.gen_rays_at(img_idx)
        rays_o = rays_o[(pixels_y, pixels_x)]
        rays_d = rays_d[(pixels_y, pixels_x)]
        return torch.cat([rays_o.cpu(), rays_d.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        raise RuntimeError('NonrigidDataset::get_rays_between -- not yet implemented!')

    def near_far_from_sphere(self, rays_o, rays_d):
        if self.scale_to_unit_sphere:
            a = torch.sum(rays_d**2, dim=-1, keepdim=True)
            b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
            mid = 0.5 * (-b) / a
            near = mid - 1.0
            far = mid + 1.0
            return near, far
        else:
            return self.near, self.far

    def image_at(self, idx, resolution_level=1):
        img = (self.images[idx].numpy() * 255.).astype(np.uint8)
        return cv.resize(img, (self.W // resolution_level, self.H // resolution_level))

    def _load_segmentations(self, datadir):
        """
        Returns the segmentations loaded from the data directory (ie. args.datadir/segmentations)
        N.B. these segmentations must have the same dimensions as the rgb images
        N.B. the segmentations should be named consistently with the rgb images
        Array has shape: n_images x h x w
        """
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
        segdir = os.path.join(datadir, 'segmentations')
        if not os.path.isdir(segdir):
            return None
        segfiles = [os.path.join(segdir, f) for f in sorted(os.listdir(segdir)) if f.endswith('.jpg') or f.endswith('.png')]
        segimgs = [imread(f)[...,:3]/255. for f in segfiles]
        segimgs = np.mean(np.stack(segimgs, 0), -1).astype(np.float32)
        return segimgs

    def _load_meshes(self, datadir, validate=False):
        """
        Returns the framewise meshes loaded from the data directory (ie. args.datadir/meshes)
        N.B. these meshes must have the same topology (ie. vertex i remains vertex i)
        Array has shape: n_images x n_verts x 3
        """
        def meshread(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            verts = []
            faces = []
            for l in lines:
                if l.startswith('v '):
                    elems = l.rstrip().split()
                    verts.append([elems[-3], elems[-2], elems[-1]])
                elif l.startswith('f '):
                    elems = l.rstrip().split()
                    faces.append([
                        elems[-3].split('/')[0],
                        elems[-2].split('/')[0],
                        elems[-1].split('/')[0]
                    ])
            return np.array(verts).astype(float), np.array(faces).astype(int)
        meshesdir = os.path.join(datadir, 'meshes')
        if not os.path.isdir(meshesdir):
            return None, None
        meshfiles = [os.path.join(meshesdir, f) for f in sorted(os.listdir(meshesdir)) if f.endswith('.obj')]
        verts = []
        faces = []
        for f in meshfiles:
            v, f = meshread(f)
            verts.append(v)
            faces.append(f)
        if validate:
            # Verify all meshes have the same number of vertices
            c = -1
            for i, m in enumerate(verts):
                if c < 0:
                    c = len(m)
                elif c != len(m):
                    print('invalid mesh in sequence!', i, c, len(m))
                    c = min(c, len(m))
        return np.array(verts), np.array(faces)


class Dataset:
    def __init__(self, conf, cuda=True):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.conf = conf
        self.scene_scale = 1.
        self.scene_center = np.array([0., 0., 0.])

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_dir = os.path.join(self.data_dir, 'image')
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        # focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        self.scene_min_pt = np.array([-1.01, -1.01, -1.01])
        self.scene_max_pt = np.array([1.01, 1.01, 1.01])

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1, novel=False):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

