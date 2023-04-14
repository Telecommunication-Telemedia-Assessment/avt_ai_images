#!/usr/bin/env python3
import argparse
import sys
import multiprocessing
import json



from features import extract_features


def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(
        description="extract several image features",
        epilog="stg7 2023",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", nargs="+", type=str, help="image to be evaluated")
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help="thread/cpu count",
    )
    parser.add_argument(
        "--report_file", type=str, default="features.json", help="file to store predictions"
    )

    a = vars(parser.parse_args())

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    results = pool.map(extract_features, a["image"])
    with open(a["report_file"], "w") as xfp:
        json.dump(results, xfp, indent=4)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
