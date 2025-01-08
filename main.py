import argparse
import os

from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ITMRec', help='name of models')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu number')
    args, _ = parser.parse_known_args()
    config_dict = {'gpu_id': args.gpu_id}
    datasets = ['baby','sports','clothing']

    results = []
    for dataset in datasets:
        result = quick_start(model=args.model, dataset=dataset, config_dict=config_dict, save_model=True)
        results.append((dataset, result))

    for dataset, result in results:
        print(f"Dataset: {dataset}")
        print(f"Result: {result}")
        print("-" * 40)