import os, json
import torch
import numpy as np
import imageio, cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


# This implementation is borrowed from NeRF: https://github.com/bmild/nerf
def load_blender_data(basedir, half_res=False, testskip=1, load_imgs=True):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        try:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
        except:
            metas[s] = None

    all_imgs = []
    all_poses = []
    counts = [0]
    H, W = -1, -1
    for s in splits:
        meta = metas[s]
        if meta == None:
            # No images provided for this split
            counts.append(counts[-1] + 0)
            continue
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            if H < 0 or W < 0:
                img = imageio.imread(fname)
                H, W = img.shape[:2]
            if load_imgs:
                imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        poses = np.array(poses).astype(np.float32)
        all_poses.append(poses)
        if load_imgs:
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            all_imgs.append(imgs)
        counts.append(counts[-1] + poses.shape[0])
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    poses = np.concatenate(all_poses, 0)
    if load_imgs:
        imgs = np.concatenate(all_imgs, 0)
    
    # Get constant info from training (guaranteed to always be present)
    meta = metas[splits[0]]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if 'clip_start' in meta and 'clip_end' in meta:
        near = meta['clip_start']
        far = meta['clip_end']
    else:
        near = None
        far = None

    # Allows for rendering novel camera views
    novel_view_file = os.path.join(basedir, 'novel_view.json')
    if os.path.isfile(novel_view_file):
        print('using dataset novel views', flush=True)
        with open(novel_view_file, 'r') as fp:
            novel_json = json.load(fp)
        render_poses = []
        for frame in novel_json['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
        if render_poses.shape != poses.shape:
            print('novel views not consistent with data frames!')
            print('render_poses.shape', render_poses.shape)
            print('poses.shape', poses.shape, flush=True)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        render_poses = render_poses.cpu().numpy()

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split, [near, far]
