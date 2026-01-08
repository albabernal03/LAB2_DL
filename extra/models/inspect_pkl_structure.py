import pickle
import numpy as np

# Load one file to inspect its structure
filepath = 'clockwise_dataset/train/clockwise_train_01.pkl'

with open(filepath, 'rb') as f:
    data = pickle.load(f)

print("Type of data:", type(data))
print("\nData structure:")

if isinstance(data, dict):
    print("Dictionary keys:", list(data.keys()))
    for key, value in data.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Sample values: {value[:5] if len(value) > 5 else value}")
        elif isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if isinstance(value[0], np.ndarray):
                    print(f"  First element shape: {value[0].shape}")
        else:
            print(f"  Value: {value}")
elif isinstance(data, np.ndarray):
    print("Array shape:", data.shape)
    print("Array dtype:", data.dtype)
    print("Sample values:", data[:5])
elif isinstance(data, (list, tuple)):
    print("Length:", len(data))
    if len(data) > 0:
        print("First element type:", type(data[0]))
        if isinstance(data[0], np.ndarray):
            print("First element shape:", data[0].shape)
            print("First element sample:", data[0][:5] if len(data[0]) > 5 else data[0])
else:
    print("Data:", data)
