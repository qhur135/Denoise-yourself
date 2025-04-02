import argparse
from pathlib import Path
import os
import util
import warnings


def get_parser(name='Self-Sampling') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--iterations', default=100000, type=int)
    # Pretrain 비율. 0일 경우 기존과 같이 upsampling strategy만 수행함
    # 0 이상인 경우(ex, 0.3) iteration * ratio 만큼 pretrain을 한 뒤, iteration을 수행하도록 구현
    # 아래 1과 2를 비교하는 것이 1과 3을 비교하는 것보다 공정(?)하다고 생각해서, 실수 확률을 줄이도록 이렇게 구현한 것임
    # 혹시 참고할만한 논문(Pretrain 수행한 경우와 아닌 경우를 비교할 때 iteration 수를 어떻게 맞추어 비교하는지)이 있다면 보고 수정 의견 주세요.
    # 1) Curvature 10,000 iter
    # 2) Pretrain 3,000 iter + Curvature 7,000 iter
    # 3) Pretrain 3,000 iter + Curvature 10,000 iter
    parser.add_argument('--export-interval', default=1000, type=int)
    parser.add_argument('--D1', default=5000, type=int)
    parser.add_argument('--D2', default=5000, type=int)
    parser.add_argument('--max-points', default=-1, type=int)
    parser.add_argument('--save-path', type=Path, required=True)
    parser.add_argument('--pc', type=str)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--stn', action='store_true')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--init-var', default=-1.0, type=float)
    parser.add_argument('--sampling-mode', default='uniform', type=str)
    parser.add_argument('--inverted', default=False, type=bool)
    parser.add_argument('--noise_level', default=0.01, type=float)
    parser.add_argument('--noise_ratio', default=0.3, type=float)
    parser.add_argument('--p1', default=0.9, type=float)
    parser.add_argument('--p2', default=-1.0, type=float)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--percentile', type=float, default=-1.0)
    parser.add_argument('--ang-wt', type=float, default=0.1)
    parser.add_argument('--force-normal-estimation', action='store_true')
    parser.add_argument('--kmeans', action='store_true')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--curvature-cache', type=str, default='')
    # denoising 추가 구현 아이디어
    # parser.add_argument('--noise_offset', default=False, type=bool) # 1. [pretrain phase] noise point offset 학습
    # parser.add_argument('--classification_upgrade', default=False, type=bool) # 2. [pretrain phase] noise classification 개선
    # parser.add_argument('--noise_penalty', default=False, type=bool) # 3. [fintune phase] noise point penalty 부여

    #----------------Pretrain 실험용 추가(240207)--------------
    parser.add_argument('--do-pretrain', default=False, type=bool)
    parser.add_argument('--pretrain-lr', default=0.0005, type=float)
    parser.add_argument('--pretrain-iter', default=10010, type=int)
    parser.add_argument('--pretrain-export-interval', default=600, type=int)
    # parser.add_argument('--pretrain-mode', default='sweep', type=str) // <-- Sweep, Noise1, Noise2?
    parser.add_argument('--target-pretrain-weight', default=1200, type=int)
    parser.add_argument('--noise-pc', type=str) # noise 2

    return parser


def parse_args(parser: argparse.ArgumentParser, inference=False):
    args = parser.parse_args()

    if args.p2 == -1.0:
        args.p2 = 1 - args.p1

    if not os.path.exists(args.save_path):
        Path.mkdir(args.save_path, exist_ok=True, parents=True)
    if not inference:
        Path.mkdir(args.save_path / 'exports', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'targets', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'sources', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'generators', exist_ok=True, parents=True)

    with open(args.save_path / ('inference_args.txt' if inference else 'args.txt'), 'w+') as file:
        file.write(util.args_to_str(args))

    return args