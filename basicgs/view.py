import argparse
import glob
from os import path as osp
from basicgs.utils.options import yaml_load, _merge_from_base
from basicgs.gaussians import build_gaussians
from basicgs.viewers import build_viewer, Editor

def parse_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str, help='Path to experiment directory.')
    parser.add_argument('--port', type=int, default=8097, help='Port for viser server.')
    parser.add_argument('--iter', type=str, default='latest', help='Iteration to view.')
    parser.add_argument('--edit', action='store_true', help='The viewer with editor.')
    args = parser.parse_args()

    yaml_file = glob.glob(osp.join(args.exp_path, '*.yaml'))
    assert len(yaml_file) == 1, f'Found {len(yaml_file)} yaml files in {args.exp_path}'
    yaml_file = yaml_file[0]
    opt = yaml_load(yaml_file)

    # convert relative path to absolute path
    if 'base' in opt:
        if isinstance(opt['base'], str) and not osp.isabs(opt['base']):
            opt['base'] = osp.join(args.exp_path, 'backup', opt['base'])
        elif isinstance(opt['base'], list):
            for idx, base_yaml in enumerate(opt['base']):
                if isinstance(base_yaml, str) and not osp.isabs(base_yaml):
                    opt['base'][idx] = osp.join(args.exp_path, 'backup', base_yaml)
                else:
                    raise ValueError(f'Invalid base option {idx}/{len(opt["base"])}: {base_yaml}')

    opt = _merge_from_base(opt, yaml_file)

    opt['path']['experiments_root'] = args.exp_path
    opt['path']['models'] = osp.join(args.exp_path, 'models')
    opt['path']['training_states'] = osp.join(args.exp_path, 'training_states')
    opt['path']['log'] = args.exp_path
    opt['path']['visualization'] = osp.join(args.exp_path, 'visualization')
    opt['path']['backup'] = osp.join(args.exp_path, 'backup')

    return opt, args

def load_pretrained_gaussians(net_g, model_path):
    net_g.load_ply(model_path)

def start_viewer(root_path):
    opt, args = parse_options(root_path)
    net_g_opt = opt['network_g']
    net_g_opt['train_dataset'] = None
    net_g = build_gaussians(net_g_opt).cuda()

    viewer_opt = opt['viewer'] if 'viewer' in opt else dict(type=net_g.default_viewer_type)
    viewer_opt['port'] = args.port
    viewer_opt['log_dir'] = args.exp_path

    model_path = osp.join(opt['path']['models'], f'net_g_{args.iter}.ply')
    load_pretrained_gaussians(net_g, model_path)

    viewer_opt['net_g'] = net_g
    viewer_opt['log_dir'] = opt['path']['log']
    viewer_opt['exp_opt'] = opt
    viewer = build_viewer(viewer_opt)
    if args.edit:
        viewer = Editor(viewer)

    while True:
        try:
            viewer.update()
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            viewer.logger.error(f'Error: {e}')
            traceback.print_exc()
            viewer.logger.error('Restarting viewer...')

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    start_viewer(root_path)
