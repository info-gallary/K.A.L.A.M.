import h5py
import numpy as np
import matplotlib.pyplot as plt

def list_all_datasets(h5_obj, path=''):
    """Recursively list all datasets in the HDF5 file."""
    datasets = []
    for key, item in h5_obj.items():
        if isinstance(item, h5py.Dataset):
            datasets.append(key)
        elif isinstance(item, h5py.Group):
            for sub_key in list_all_datasets(item, path + '/' + key):
                datasets.append(f"{key}/{sub_key}")
    return datasets

# ✅ Path to your HDF5 file
file_path = r"D:\Hackathon\ISRO\JUL\JUL\3RIMG_01JUL2024_0015_L1C_ASIA_MER_V01R00.h5"

with h5py.File(file_path, 'r') as f:
    print("📂 All available datasets:")
    all_datasets = list_all_datasets(f)
    for i, name in enumerate(all_datasets):
        print(f"{i}: {name}")

        # ✅ Pick a dataset to visualize
        dataset_index = i  # Change if needed
        selected_dataset = all_datasets[dataset_index]
        print(f"\n📌 Displaying dataset: {selected_dataset}")
        
        data = f[selected_dataset][:]
        print("Shape:", data.shape, "| Dtype:", data.dtype)

        # ✅ Normalize if needed
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # ✅ Display the image
        plt.figure(figsize=(8, 6))
        if norm_data.ndim == 2:
            plt.imshow(norm_data, cmap='gray')
            plt.title(f"INSAT Image (2D): {selected_dataset}")
            plt.axis('off')
            plt.colorbar()
            plt.show()

        elif norm_data.ndim == 3:
            plt.imshow(norm_data[0], cmap='gray')
            plt.title(f"INSAT Image (3D slice): {selected_dataset}")
            plt.axis('off')
            plt.colorbar()
            plt.show()

        elif norm_data.ndim == 1:
            plt.plot(norm_data)
            plt.title(f"1D Plot of: {selected_dataset}")
            plt.xlabel("Index")
            plt.ylabel("Normalized Value")
            plt.grid(True)
            plt.show()

        else:
            print("Unsupported dataset shape:", norm_data.shape)
