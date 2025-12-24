import argparse
import logging
import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', default=None, type=str, required=False, help='Input file')
    parser.add_argument('--debug', action='store_true', required=False, help='Debug mode')

    return parser.parse_args()


def set_log() -> None:
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s (Line: %(lineno)d [%(filename)s])',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main() -> None:
    # args = parse_args()
    set_log()
    logging.info('Starting...')

    # 以只读模式打开文件
    file_path = '/home/ps/wx_ws/code/2512_keyframes/RoboTwin/policy/ACT/processed_data/sim-blocks_ranking_rgb/aloha_clean_500-200/episode_9.hdf5'
    with h5py.File(file_path, 'r') as f:
        print(f"正在解析文件: {file_path}")
        # visititems 会遍历文件中的每一个节点
        f.visititems(print_structure)


def print_structure(name, obj):
    """递归打印 HDF5 文件的结构"""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name} (shape={obj.shape}, type={obj.dtype})")


if __name__ == '__main__':
    main()