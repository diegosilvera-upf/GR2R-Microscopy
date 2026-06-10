from pathlib import Path
import tifffile


def split_tiff_sequences(input_dir, output_dir):
    """
    Splits multipage TIFF sequences into individual TIFF frames.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing TIFF sequences.
    output_dir : str or Path
        Directory where extracted frames will be saved.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]
    ]

    print(f"Found {len(tif_files)} TIFF files.")

    for tif_path in tif_files:

        print(f"Processing {tif_path.name}...")

        sequence = tifffile.imread(tif_path)

        sequence_out_dir = output_dir / tif_path.stem
        sequence_out_dir.mkdir(exist_ok=True)

        for frame_idx, frame in enumerate(sequence):

            frame_path = sequence_out_dir / f"frame_{frame_idx:04d}.tif"

            tifffile.imwrite(
                frame_path,
                frame,
            )

        print(f"Saved {len(sequence)} frames to {sequence_out_dir}")

    print("Done.")


if __name__ == "__main__":

    input_dir = "results/inference_fastdvdnet/loreal_2026_06_10-14_15_41"
    output_dir = "results/inference_fastdvdnet/loreal_2026_06_10-14_15_41splitted"

    split_tiff_sequences(input_dir, output_dir)