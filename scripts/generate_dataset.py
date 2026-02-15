#!/usr/bin/env python3
"""
Unified dataset generation script for topography images.

This script provides a simple interface to generate synthetic topography datasets
with various noise types. It supports all noise controllers implemented in the
src/data_generation/noise_controllers/ module.

Examples:
    # Generate pure images (no noise)
    python scripts/generate_dataset.py --noise pure --output data/generated/pure

    # Generate with average noise
    python scripts/generate_dataset.py --noise average --output data/generated/avg \
        --path-average-noise data/average_noise/steel

    # Generate with Perlin noise
    python scripts/generate_dataset.py --noise perlin --output data/generated/perlin \
        --alpha 0.45

    # Generate with multiple noise types (perlin + triangular)
    python scripts/generate_dataset.py --noise perlin triangular --output data/generated/combo \
        --alpha 0.45 --nr-of-triangles 3 8

    # Generate with custom epsilon range
    python scripts/generate_dataset.py --noise perlin --output data/generated/test \
        --epsilon-range 0.0 0.5 --epsilon-step 0.01 --n-copies 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_generation.datasets.generator import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic topography dataset with various noise types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--noise",
        type=str,
        nargs="+",
        required=True,
        choices=["pure", "average", "blackbox", "bubble", "triangular", "fourier", "perlin"],
        help="Noise type(s) to apply. Use 'pure' for no noise. Multiple types can be combined.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path for generated images",
    )

    # Optional generation parameters
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="img",
        help="Prefix for generated image filenames (default: img)",
    )
    parser.add_argument(
        "--epsilon-range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("MIN", "MAX"),
        help="Epsilon value range (default: 0.0 1.0)",
    )
    parser.add_argument(
        "--epsilon-step",
        type=float,
        default=0.001,
        help="Step size for epsilon values (default: 0.001)",
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=50,
        help="Number of copies per epsilon value (default: 50)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Image size in pixels (default: 640 480)",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        nargs=2,
        default=[80, 210],
        metavar=("MIN", "MAX"),
        help="Brightness range (default: 80 210)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--zipfile",
        action="store_true",
        help="Save images to a zip file instead of directory",
    )
    parser.add_argument(
        "--zip-filename",
        type=str,
        default="dataset.zip",
        help="Zip file name if --zipfile is used (default: dataset.zip)",
    )

    # Noise-specific parameters
    noise_group = parser.add_argument_group("noise-specific parameters")

    # Average noise
    noise_group.add_argument(
        "--path-average-noise",
        type=str,
        help="Path to average noise images (required for 'average' noise)",
    )

    # Blackbox noise
    noise_group.add_argument(
        "--blackbox-width",
        type=int,
        nargs=2,
        default=[30, 320],
        metavar=("MIN", "MAX"),
        help="Blackbox width range (default: 30 320)",
    )
    noise_group.add_argument(
        "--blackbox-height",
        type=int,
        nargs=2,
        default=[30, 240],
        metavar=("MIN", "MAX"),
        help="Blackbox height range (default: 30 240)",
    )

    # Bubble noise
    noise_group.add_argument(
        "--spray-particles",
        type=int,
        default=None,
        help="Number of spray particles for bubble noise (default: auto)",
    )
    noise_group.add_argument(
        "--spray-diameter",
        type=int,
        default=None,
        help="Spray diameter for bubble noise (default: auto)",
    )
    noise_group.add_argument(
        "--range-of-blobs",
        type=int,
        nargs=2,
        default=[30, 40],
        metavar=("MIN", "MAX"),
        help="Range of blobs for bubble noise (default: 30 40)",
    )

    # Triangular noise
    noise_group.add_argument(
        "--nr-of-triangles",
        type=int,
        nargs=2,
        default=[3, 8],
        metavar=("MIN", "MAX"),
        help="Range of triangular segments (default: 3 8)",
    )
    noise_group.add_argument(
        "--triangular-strength",
        type=int,
        nargs=2,
        default=[10, 15],
        metavar=("MIN", "MAX"),
        help="Triangular noise strength range (default: 10 15)",
    )

    # Fourier noise
    noise_group.add_argument(
        "--fourier-domain",
        type=str,
        choices=["ampl", "freq"],
        default="ampl",
        help="Fourier domain type: amplitude or frequency (default: ampl)",
    )
    noise_group.add_argument(
        "--path-fourier-noise-freq",
        type=str,
        help="Path to Fourier frequency noise (required for 'fourier' with domain='freq')",
    )
    noise_group.add_argument(
        "--path-fourier-noise-ampl",
        type=str,
        help="Path to Fourier amplitude noise (required for 'fourier' with domain='ampl')",
    )
    noise_group.add_argument(
        "--pass-value",
        type=int,
        default=4,
        help="Pass value for Fourier frequency filtering (default: 4)",
    )
    noise_group.add_argument(
        "--noise-proportion",
        type=float,
        default=0.5,
        help="Proportion of Fourier noise to apply (default: 0.5)",
    )

    # Perlin noise
    noise_group.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Perlin noise blending alpha (0.0=no noise, 1.0=pure noise) (default: 0.45)",
    )
    noise_group.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Perlin noise scale parameter (default: 100.0)",
    )
    noise_group.add_argument(
        "--octaves",
        type=int,
        default=6,
        help="Perlin noise octaves (default: 6)",
    )
    noise_group.add_argument(
        "--persistence-range",
        type=float,
        nargs=2,
        default=[0.7, 0.9],
        metavar=("MIN", "MAX"),
        help="Perlin persistence range (default: 0.7 0.9)",
    )
    noise_group.add_argument(
        "--lacunarity",
        type=float,
        default=2.0,
        help="Perlin lacunarity parameter (default: 2.0)",
    )
    noise_group.add_argument(
        "--clip-limit-range",
        type=float,
        nargs=2,
        default=[1.0, 2.0],
        metavar=("MIN", "MAX"),
        help="CLAHE clip limit range for Perlin (default: 1.0 2.0)",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations and requirements."""
    errors = []

    # Check if 'pure' is combined with other noises
    if "pure" in args.noise and len(args.noise) > 1:
        errors.append("'pure' noise cannot be combined with other noise types")

    # Check noise-specific required parameters
    if "average" in args.noise and not args.path_average_noise:
        errors.append("--path-average-noise is required when using 'average' noise")

    if "fourier" in args.noise:
        if args.fourier_domain == "ampl" and not args.path_fourier_noise_ampl:
            errors.append(
                "--path-fourier-noise-ampl is required for 'fourier' noise with domain='ampl'"
            )
        if args.fourier_domain == "freq" and not args.path_fourier_noise_freq:
            errors.append(
                "--path-fourier-noise-freq is required for 'fourier' noise with domain='freq'"
            )

    # Validate ranges
    if args.epsilon_range[0] >= args.epsilon_range[1]:
        errors.append("Epsilon min must be less than max")

    if args.brightness[0] >= args.brightness[1]:
        errors.append("Brightness min must be less than max")

    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def build_noise_kwargs(args):
    """Build kwargs dictionary for noise controllers based on arguments."""
    kwargs = {}

    for noise_type in args.noise:
        if noise_type == "pure":
            continue

        if noise_type == "average":
            kwargs["path_average_noise"] = args.path_average_noise

        elif noise_type == "blackbox":
            kwargs["width"] = tuple(args.blackbox_width)
            kwargs["height"] = tuple(args.blackbox_height)

        elif noise_type == "bubble":
            if args.spray_particles:
                kwargs["spray_particles"] = args.spray_particles
            if args.spray_diameter:
                kwargs["spray_diameter"] = args.spray_diameter
            kwargs["range_of_blobs"] = tuple(args.range_of_blobs)

        elif noise_type == "triangular":
            kwargs["nr_of_triangles"] = tuple(args.nr_of_triangles)
            kwargs["strength"] = tuple(args.triangular_strength)
            kwargs["center_point"] = (args.size[0] // 2, args.size[1] // 2)
            kwargs["channels"] = 1

        elif noise_type == "fourier":
            kwargs["domain"] = args.fourier_domain
            kwargs["path_fourier_noise_freq"] = args.path_fourier_noise_freq or ""
            kwargs["path_fourier_noise_ampl"] = args.path_fourier_noise_ampl or ""
            kwargs["pass_value"] = args.pass_value
            kwargs["noise_proportion"] = args.noise_proportion

        elif noise_type == "perlin":
            kwargs["alpha"] = args.alpha
            kwargs["scale"] = args.scale
            kwargs["octaves"] = args.octaves
            kwargs["persistence_range"] = tuple(args.persistence_range)
            kwargs["lacunarity"] = args.lacunarity
            kwargs["clip_limit_range"] = tuple(args.clip_limit_range)

    return kwargs


def main():
    args = parse_args()
    validate_args(args)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle 'pure' noise type
    noise_types = [] if "pure" in args.noise else args.noise

    # Build noise-specific kwargs
    noise_kwargs = build_noise_kwargs(args)

    # Print configuration
    print("=" * 60)
    print("Dataset Generation Configuration")
    print("=" * 60)
    print(f"Noise type(s): {', '.join(args.noise)}")
    print(f"Output path: {args.output}")
    print(f"Epsilon range: {args.epsilon_range[0]} to {args.epsilon_range[1]}")
    print(f"Epsilon step: {args.epsilon_step}")
    print(f"Copies per epsilon: {args.n_copies}")
    print(f"Image size: {args.size[0]}x{args.size[1]}")
    print(f"Brightness range: {args.brightness[0]} to {args.brightness[1]}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Generate dataset
    try:
        generate_dataset(
            noise_type=noise_types,
            path=args.output,
            n_copies=args.n_copies,
            name_prefix=args.name_prefix,
            epsilon_range=tuple(args.epsilon_range),
            epsilon_step=args.epsilon_step,
            size=tuple(args.size),
            brightness=tuple(args.brightness),
            zipfile=args.zipfile,
            filename=args.zip_filename if args.zipfile else "",
            save_parameters=True,
            parameters_filename="parameters.csv",
            seed=args.seed,
            **noise_kwargs,
        )
        print("\n" + "=" * 60)
        print("Dataset generation completed successfully!")
        print(f"Images saved to: {args.output}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during dataset generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
