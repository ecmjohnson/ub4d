import argparse

def ParseArgs(post=False):
    parser = argparse.ArgumentParser()

    # Main options
    parser.add_argument(
        '--case',
        type=str,
        default='',
        help='dataset to use (replaces CASE_NAME in config file)'
    )
    parser.add_argument(
        '--conf',
        type=str,
        default='./confs/base.conf',
        help='configuration file to use'
    )
    
    if not post:
        parser.add_argument(
            '--mode',
            type=str,
            default='train',
            help='mode to run (see README or code for options)'
        )
        parser.add_argument(
            '--is_continue',
            default=False,
            action="store_true",
            help='continue with previously trained model'
        )

        # Marching cubes output options
        parser.add_argument(
            '--full_scene_bounds',
            default=False,
            action="store_true",
            help='march the complete scene (rather than object BBOX, if available)'
        )
        parser.add_argument(
            '--custom_bounds',
            default=False,
            action="store_true",
            help='use custom bounds from dataset directory (rather than object BBOX, if available)'
        )
        parser.add_argument(
            '--frustum_cull',
            default=False,
            action="store_true",
            help='cull output geometry by viewing frustum'
        )
        parser.add_argument(
            '--mcube_resolution',
            type=int,
            default=512,
            help='marching cubes resolution'
        )
        parser.add_argument(
            '--mcube_threshold',
            type=float,
            default=0.0,
            help='threshold for marching cubes (SDF=>0.0)'
        )

        # Other utility/experiment options
        parser.add_argument(
            '--validate_frame',
            type=int,
            default=-1,
            help='frame to validate (-1 for canonical space)'
        )
        parser.add_argument(
            '--start_frame',
            type=int,
            default=0,
            help='frame to start marching'
        )
        parser.add_argument(
            '--frame1',
            type=int,
            default=0,
            help='frame 1 for validate_sceneflow mode'
        )
        parser.add_argument(
            '--frame2',
            type=int,
            default=1,
            help='frame 2 for validate_sceneflow mode'
        )
        parser.add_argument(
            '--gpu',
            type=int,
            default=0,
            help='index of GPU to use'
        )
        parser.add_argument(
            '--latents_file',
            type=str,
            default='latents.npy',
            help='file of latent codes'
        )

    else:
        # Post-processing options
        parser.add_argument(
            '--render_view',
            type=int,
            default=-1,
            help='index of view from which to render (-1 is camera)'
        )
        parser.add_argument(
            '--render_novel',
            default=False,
            action="store_true",
            help='render from novel views'
        )
        parser.add_argument(
            '--latent_animation',
            default=False,
            action="store_true",
            help='output an animation of the last latent code sequence'
        )
        parser.add_argument(
            '--skip_processing',
            default=False,
            action="store_true",
            help='skip processing of results'
        )
        parser.add_argument(
            '--skip_latents',
            default=False,
            action="store_true",
            help='skip analysis of the latent codes'
        )
        parser.add_argument(
            '--skip_render',
            default=False,
            action="store_true",
            help='skip rendering of results'
        )
        parser.add_argument(
            '--render_gt',
            default=False,
            action="store_true",
            help='render the ground truth meshes'
        )
        parser.add_argument(
            '--render_proxies',
            default=False,
            action="store_true",
            help='render the proxy geometry'
        )

    return parser.parse_args()
