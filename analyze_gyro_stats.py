import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define dataset paths
DATASETS = [
    'clockwise_dataset',
    'horizontal_swipe_dataset',
    'vertical_updown_dataset',
    'forward_thrust_dataset',
    'wrist_twist_dataset'
]

# Define output directory
OUTPUT_DIR = Path('figures/boxplots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_pkl_file(filepath):
    """Load a pickle file and return its contents."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_gyro_stats(data):
    """
    Extract mean, std, and median for gyro_x, gyro_y, gyro_z from a recording.
    
    Args:
        data: pandas DataFrame or other format containing gyro data
        
    Returns:
        Dictionary with statistical features
    """
    stats = {}
    
    # Handle pandas DataFrame format (expected format)
    if isinstance(data, pd.DataFrame):
        # Check for Greek letter column names (ωx, ωy, ωz)
        if 'ωx' in data.columns and 'ωy' in data.columns and 'ωz' in data.columns:
            gyro_x = data['ωx'].values
            gyro_y = data['ωy'].values
            gyro_z = data['ωz'].values
        # Check for standard column names
        elif 'gyro_x' in data.columns and 'gyro_y' in data.columns and 'gyro_z' in data.columns:
            gyro_x = data['gyro_x'].values
            gyro_y = data['gyro_y'].values
            gyro_z = data['gyro_z'].values
        else:
            return None
    # Handle dictionary format
    elif isinstance(data, dict):
        if 'gyro_x' in data and 'gyro_y' in data and 'gyro_z' in data:
            gyro_x = np.array(data['gyro_x'])
            gyro_y = np.array(data['gyro_y'])
            gyro_z = np.array(data['gyro_z'])
        elif 'gyro' in data:
            gyro = np.array(data['gyro'])
            gyro_x = gyro[:, 0] if gyro.ndim > 1 else gyro
            gyro_y = gyro[:, 1] if gyro.ndim > 1 and gyro.shape[1] > 1 else np.zeros_like(gyro_x)
            gyro_z = gyro[:, 2] if gyro.ndim > 1 and gyro.shape[1] > 2 else np.zeros_like(gyro_x)
        else:
            return None
    # Handle numpy array format
    elif isinstance(data, np.ndarray):
        if data.ndim > 1 and data.shape[1] >= 3:
            gyro_x = data[:, 0]
            gyro_y = data[:, 1]
            gyro_z = data[:, 2]
        else:
            return None
    else:
        return None
    
    # Calculate statistics
    stats['mean_gyro_x'] = np.mean(gyro_x)
    stats['mean_gyro_y'] = np.mean(gyro_y)
    stats['mean_gyro_z'] = np.mean(gyro_z)
    
    stats['std_gyro_x'] = np.std(gyro_x)
    stats['std_gyro_y'] = np.std(gyro_y)
    stats['std_gyro_z'] = np.std(gyro_z)
    
    stats['median_gyro_x'] = np.median(gyro_x)
    stats['median_gyro_y'] = np.median(gyro_y)
    stats['median_gyro_z'] = np.median(gyro_z)
    
    return stats

def process_dataset(dataset_name):
    """
    Process all training files in a dataset and extract gyro statistics.
    
    Args:
        dataset_name: Name of the dataset directory
        
    Returns:
        List of dictionaries containing stats for each recording
    """
    train_dir = Path(dataset_name) / 'train'
    all_stats = []
    
    if not train_dir.exists():
        print(f"Warning: {train_dir} does not exist")
        return all_stats
    
    # Get all .pkl files in train directory
    pkl_files = sorted(train_dir.glob('*.pkl'))
    
    print(f"Processing {dataset_name}: found {len(pkl_files)} files")
    
    for pkl_file in pkl_files:
        try:
            data = load_pkl_file(pkl_file)
            stats = extract_gyro_stats(data)
            
            if stats is not None:
                stats['gesture'] = dataset_name.replace('_dataset', '')
                stats['filename'] = pkl_file.name
                all_stats.append(stats)
            else:
                print(f"  Warning: Could not extract stats from {pkl_file.name}")
        except Exception as e:
            print(f"  Error processing {pkl_file.name}: {e}")
    
    return all_stats

def create_boxplot(df, feature, output_path):
    """
    Create a boxplot for a specific feature across all gestures.
    
    Args:
        df: DataFrame containing the statistics
        feature: Name of the feature to plot
        output_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    sns.boxplot(data=df, x='gesture', y=feature, palette='Set2')
    
    # Customize plot
    plt.title(f'Distribution of {feature} across Gestures', fontsize=14, fontweight='bold')
    plt.xlabel('Gesture Type', fontsize=12)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    """Main function to process all datasets and create boxplots."""
    print("=" * 60)
    print("Gyroscope Statistics Analysis")
    print("=" * 60)
    
    # Collect statistics from all datasets
    all_data = []
    
    for dataset in DATASETS:
        stats_list = process_dataset(dataset)
        all_data.extend(stats_list)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nTotal recordings processed: {len(df)}")
    print(f"Gestures: {df['gesture'].unique()}")
    print(f"\nRecordings per gesture:")
    print(df['gesture'].value_counts().sort_index())
    
    # Define features to plot
    features = [
        'mean_gyro_x', 'mean_gyro_y', 'mean_gyro_z',
        'std_gyro_x', 'std_gyro_y', 'std_gyro_z',
        'median_gyro_x', 'median_gyro_y', 'median_gyro_z'
    ]
    
    # Create boxplots
    print(f"\n{'=' * 60}")
    print("Creating Boxplots")
    print("=" * 60)
    
    for feature in features:
        output_path = OUTPUT_DIR / f"{feature}.png"
        create_boxplot(df, feature, output_path)
    
    # Save summary statistics
    summary_path = OUTPUT_DIR / 'summary_statistics.csv'
    summary = df.groupby('gesture')[features].describe()
    summary.to_csv(summary_path)
    print(f"\nSaved summary statistics to: {summary_path}")
    
    print(f"\n{'=' * 60}")
    print("Analysis Complete!")
    print(f"All boxplots saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
