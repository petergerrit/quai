import numpy as np
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <file1.npy> <file2.npy> <output.npy>")
        sys.exit(1)

    file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # Load arrays
    arr1 = np.load(file1)
    arr2 = np.load(file2)

    print(f"Loaded '{file1}': shape {arr1.shape}")
    print(f"Loaded '{file2}': shape {arr2.shape}")

    # Validate both are arrays of nxn matrices (3D arrays: num_matrices x n x n)
    if arr1.ndim != 3 or arr2.ndim != 3:
        print("Error: Both files must contain 3D arrays (num_matrices x n x n).")
        sys.exit(1)

    if arr1.shape[1:] != arr2.shape[1:]:
        print(f"Error: Matrix dimensions don't match: {arr1.shape[1:]} vs {arr2.shape[1:]}")
        sys.exit(1)

    # Combine arrays
    combined = np.concatenate([arr1, arr2], axis=0)
    print(f"Combined shape: {combined.shape}")

    # Remove duplicate matrices by reshaping and using np.unique
    n_matrices = combined.shape[0]
    flat = combined.reshape(n_matrices, -1)  # flatten each matrix to a 1D row

    _, unique_indices = np.unique(flat, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)  # preserve original order
    unique_combined = combined[unique_indices]

    n_removed = n_matrices - len(unique_indices)
    print(f"Duplicates removed: {n_removed}")
    print(f"Final array shape: {unique_combined.shape}")

    # Save to output file
    np.save(output_file, unique_combined)
    print(f"Saved to '{output_file}'")

if __name__ == "__main__":
    main()
