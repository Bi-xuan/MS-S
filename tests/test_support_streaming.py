import argparse
import itertools
from pathlib import Path
import resource
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from support_utils import get_all_supports


def peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def test_generator_type(n, n_edge):
    supports = get_all_supports(n, n_edge)
    print(f"type: {type(supports)}")
    first = next(supports)
    print(f"first mask shape: {first.shape}")


def test_streaming(n, n_edge, limit):
    start = time.time()
    count = 0

    for _ in itertools.islice(get_all_supports(n, n_edge), limit):
        count += 1

    print(f"streamed supports: {count}")
    print(f"peak memory MB: {peak_memory_mb():.2f}")
    print(f"seconds: {time.time() - start:.2f}")


def test_materialized_prefix(n, n_edge, limit):
    start = time.time()

    supports = list(itertools.islice(get_all_supports(n, n_edge), limit))

    print(f"materialized supports: {len(supports)}")
    print(f"peak memory MB: {peak_memory_mb():.2f}")
    print(f"seconds: {time.time() - start:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--n-edge", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument(
        "--mode",
        choices=["generator", "streaming", "materialized"],
        default="streaming",
    )
    args = parser.parse_args()

    if args.mode == "generator":
        test_generator_type(args.n, args.n_edge)
    elif args.mode == "streaming":
        test_streaming(args.n, args.n_edge, args.limit)
    elif args.mode == "materialized":
        test_materialized_prefix(args.n, args.n_edge, args.limit)


if __name__ == "__main__":
    main()
