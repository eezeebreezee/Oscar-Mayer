import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from wiener_process import run_wiener_animation
from ddm_simulation import run_ddm_animation

def main():
    parser = argparse.ArgumentParser(description='Run GPU-accelerated animations.')
    parser.add_argument('--simulation', choices=['wiener', 'ddm'], required=True,
                        help='Choose simulation: wiener or ddm')
    args = parser.parse_args()

    if args.simulation == 'wiener':
        run_wiener_animation()
    elif args.simulation == 'ddm':
        run_ddm_animation()

if __name__ == '__main__':
    main()
