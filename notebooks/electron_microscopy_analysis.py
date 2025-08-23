#!/usr/bin/env python3
"""
Electron Microscopy Dataset Analysis

This script demonstrates how to work with the huggingface/electron_microscopy_dataset 
for analyzing electron microscopy images.

Overview:
Electron microscopy provides high-resolution images of biological samples at the nanoscale. 
This dataset contains various electron microscopy images that can be used for:
- Image segmentation
- Object detection
- Feature extraction
- Biomedical image analysis

Setup and Dependencies:
Make sure you have the required packages installed:
pip install datasets transformers torch torchvision pillow matplotlib seaborn opencv-python scikit-image
"""

# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from skimage import filters, morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def main():
    """Main function to run the electron microscopy analysis"""
    
    print("=" * 60)
    print("ELECTRON MICROSCOPY DATASET ANALYSIS")
    print("=" * 60)
    
    # 1. Load the Electron Microscopy Dataset
    print("\n1. Loading electron microscopy dataset...")
    try:
        dataset = load_dataset("huggingface/electron_microscopy_dataset")
        print(f"Dataset structure: {dataset}")
        print(f"Available splits: {list(dataset.keys())}")
        
        if 'train' in dataset:
            print(f"\nTraining set size: {len(dataset['train'])}")
            print(f"Sample features: {dataset['train'].features}")
            
            # Show first few samples
            print("\nFirst few samples:")
            for i, sample in enumerate(dataset['train'].select(range(min(3, len(dataset['train']))))):
                print(f"Sample {i+1}: {sample}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the dataset name is correct and you have internet access.")
        return
    
    # 2. Explore and Visualize the Data
    print("\n2. Exploring and visualizing data...")
    if 'train' in dataset:
        visualize_samples(dataset, 'train')
    
    # 3. Data Preprocessing and Analysis
    print("\n3. Analyzing image properties...")
    if 'train' in dataset:
        sizes, channels, dtypes = analyze_image_properties(dataset, 'train')
    
    # 4. Image Enhancement and Preprocessing
    print("\n4. Applying preprocessing pipeline...")
    if 'train' in dataset:
        apply_preprocessing_pipeline(dataset, 'train')
    
    # 5. Feature Extraction and Analysis
    print("\n5. Extracting and analyzing features...")
    if 'train' in dataset:
        features = analyze_features_across_samples(dataset, 'train')
    
    # 6. Advanced Image Processing
    print("\n6. Applying advanced processing...")
    if 'train' in dataset:
        demonstrate_advanced_processing(dataset, 'train')
    
    # 7. Data Export and Summary
    print("\n7. Exporting processed data...")
    if 'train' in dataset:
        processed_data = export_processed_data(dataset, 'train')
        
        # Generate summary
        if 'features' in locals():
            generate_analysis_summary(dataset, features)
    
    print("\nAnalysis complete! Check the generated files and plots.")

def visualize_samples(dataset, split='train', num_samples=6):
    """Visualize multiple samples from the dataset"""
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, sample in enumerate(samples):
        if i >= num_samples:
            break
            
        # Handle different possible image formats
        if 'image' in sample:
            img = sample['image']
        elif 'pixel_values' in sample:
            img = sample['pixel_values']
        else:
            # Try to find any image-like field
            img_fields = [k for k, v in sample.items() if hasattr(v, 'size') or hasattr(v, 'shape')]
            if img_fields:
                img = sample[img_fields[0]]
            else:
                print(f"No image field found in sample {i}")
                continue
        
        # Convert to numpy array if needed
        if hasattr(img, 'convert'):  # PIL Image
            img_array = np.array(img)
        elif torch.is_tensor(img):  # PyTorch tensor
            img_array = img.numpy()
        else:
            img_array = np.array(img)
        
        # Handle grayscale vs RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            axes[i].imshow(img_array)
        else:
            axes[i].imshow(img_array, cmap='gray')
        
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_image_properties(dataset, split='train', num_samples=100):
    """Analyze basic properties of images in the dataset"""
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
    
    sizes = []
    channels = []
    dtypes = []
    
    for sample in samples:
        # Find image field
        if 'image' in sample:
            img = sample['image']
        elif 'pixel_values' in sample:
            img = sample['pixel_values']
        else:
            continue
        
        # Convert to numpy array
        if hasattr(img, 'convert'):
            img_array = np.array(img)
        elif torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img)
        
        sizes.append(img_array.shape[:2])
        channels.append(img_array.shape[2] if len(img_array.shape) == 3 else 1)
        dtypes.append(str(img_array.dtype))
    
    # Analyze results
    print(f"Analyzed {len(sizes)} images from {split} split")
    print(f"\nImage sizes (height, width):")
    print(f"  Min: {min(sizes) if sizes else 'N/A'}")
    print(f"  Max: {max(sizes) if sizes else 'N/A'}")
    print(f"  Most common: {max(set(sizes), key=sizes.count) if sizes else 'N/A'}")
    
    print(f"\nChannels:")
    channel_counts = {}
    for ch in channels:
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    for ch, count in channel_counts.items():
        print(f"  {ch} channel(s): {count} images")
    
    print(f"\nData types:")
    dtype_counts = {}
    for dt in dtypes:
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    for dt, count in dtype_counts.items():
        print(f"  {dt}: {count} images")
    
    return sizes, channels, dtypes

def preprocess_em_image(image, target_size=(512, 512)):
    """Preprocess electron microscopy image"""
    # Convert to numpy array if needed
    if hasattr(image, 'convert'):
        img_array = np.array(image)
    elif torch.is_tensor(image):
        img_array = image.numpy()
    else:
        img_array = np.array(image)
    
    # Convert to grayscale if RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Normalize to 0-1 range
    img_norm = img_gray.astype(np.float32) / 255.0
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply((img_blur * 255).astype(np.uint8))
    
    # Resize to target size
    img_resized = cv2.resize(img_enhanced, target_size)
    
    return img_resized

def apply_preprocessing_pipeline(dataset, split='train', num_samples=6):
    """Apply preprocessing pipeline to multiple samples"""
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    for i, sample in enumerate(samples):
        if i >= num_samples:
            break
            
        # Get original image
        if 'image' in sample:
            img = sample['image']
        elif 'pixel_values' in sample:
            img = sample['pixel_values']
        else:
            continue
        
        # Original image
        if hasattr(img, 'convert'):
            img_orig = np.array(img)
        elif torch.is_tensor(img):
            img_orig = img.numpy()
        else:
            img_orig = np.array(img)
        
        # Preprocessed image
        img_processed = preprocess_em_image(img)
        
        # Display original
        if len(img_orig.shape) == 3 and img_orig.shape[2] == 3:
            axes[0, i].imshow(img_orig)
        else:
            axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Display processed
        axes[1, i].imshow(img_processed, cmap='gray')
        axes[1, i].set_title(f'Processed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_image_features(image):
    """Extract various features from an image"""
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(image)
    features['std'] = np.std(image)
    features['min'] = np.min(image)
    features['max'] = np.max(image)
    features['median'] = np.median(image)
    
    # Edge density
    edges = cv2.Canny(image, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    
    return features

def analyze_features_across_samples(dataset, split='train', num_samples=20):
    """Analyze features across multiple samples"""
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
    
    all_features = []
    
    for sample in samples:
        # Get image
        if 'image' in sample:
            img = sample['image']
        elif 'pixel_values' in sample:
            img = sample['pixel_values']
        else:
            continue
        
        # Preprocess
        img_processed = preprocess_em_image(img)
        
        # Extract features
        features = extract_image_features(img_processed)
        all_features.append(features)
    
    # Analyze feature distributions
    if all_features:
        print(f"Extracted features from {len(all_features)} images")
        
        # Plot feature distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mean intensity
        means = [f['mean'] for f in all_features]
        axes[0].hist(means, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_title('Mean Intensity Distribution')
        axes[0].set_xlabel('Mean Intensity')
        axes[0].set_ylabel('Frequency')
        
        # Standard deviation
        stds = [f['std'] for f in all_features]
        axes[1].hist(stds, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_title('Standard Deviation Distribution')
        axes[1].set_xlabel('Standard Deviation')
        axes[1].set_ylabel('Frequency')
        
        # Edge density
        edge_densities = [f['edge_density'] for f in all_features]
        axes[2].hist(edge_densities, bins=20, alpha=0.7, edgecolor='black')
        axes[2].set_title('Edge Density Distribution')
        axes[2].set_xlabel('Edge Density')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    return all_features

def advanced_em_processing(image):
    """Apply advanced processing techniques for EM images"""
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 1. Denoising using Non-local Means
    denoised = cv2.fastNlMeansDenoising(image)
    
    # 2. Unsharp masking for enhancement
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    # 3. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(unsharp_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 4. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return {
        'original': image,
        'denoised': denoised,
        'unsharp_masked': unsharp_mask,
        'morphological': closed,
        'thresholded': adaptive_thresh
    }

def demonstrate_advanced_processing(dataset, split='train', num_samples=3):
    """Demonstrate advanced processing on sample images"""
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
    
    for i, sample in enumerate(samples):
        if i >= num_samples:
            break
            
        # Get image
        if 'image' in sample:
            img = sample['image']
        elif 'pixel_values' in sample:
            img = sample['pixel_values']
        else:
            continue
        
        # Preprocess
        img_processed = preprocess_em_image(img)
        
        # Apply advanced processing
        results = advanced_em_processing(img_processed)
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Denoised
        axes[0, 1].imshow(results['denoised'], cmap='gray')
        axes[0, 1].set_title('Denoised')
        axes[0, 1].axis('off')
        
        # Unsharp masked
        axes[0, 2].imshow(results['unsharp_masked'], cmap='gray')
        axes[0, 2].set_title('Unsharp Masked')
        axes[0, 2].axis('off')
        
        # Morphological
        axes[1, 0].imshow(results['morphological'], cmap='gray')
        axes[1, 0].set_title('Morphological')
        axes[1, 0].axis('off')
        
        # Thresholded
        axes[1, 1].imshow(results['thresholded'], cmap='gray')
        axes[1, 1].set_title('Thresholded')
        axes[1, 1].axis('off')
        
        # Histogram comparison
        axes[1, 2].hist(results['original'].ravel(), bins=50, alpha=0.7, label='Original', density=True)
        axes[1, 2].hist(results['denoised'].ravel(), bins=50, alpha=0.7, label='Denoised', density=True)
        axes[1, 2].set_title('Histogram Comparison')
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        
        plt.suptitle(f'Advanced Processing - Sample {i+1}', fontsize=16)
        plt.tight_layout()
        plt.show()

def export_processed_data(dataset, split='train', output_dir='processed_em_data'):
    """Export processed data to files"""
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    samples = dataset[split].select(range(min(50, len(dataset[split]))))
    
    processed_data = []
    
    for i, sample in enumerate(samples):
        # Get image
        if 'image' in sample:
            img = sample['image']
        elif 'pixel_values' in sample:
            img = sample['pixel_values']
        else:
            continue
        
        # Preprocess
        img_processed = preprocess_em_image(img)
        
        # Extract features
        features = extract_image_features(img_processed)
        
        # Save processed image
        img_filename = f"processed_image_{i:03d}.png"
        img_path = os.path.join(output_dir, 'images', img_filename)
        cv2.imwrite(img_path, img_processed)
        
        # Store metadata
        sample_data = {
            'sample_id': i,
            'image_filename': img_filename,
            'features': {
                'mean': float(features['mean']),
                'std': float(features['std']),
                'edge_density': float(features['edge_density'])
            }
        }
        
        processed_data.append(sample_data)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Exported {len(processed_data)} processed images to {output_dir}/")
    print(f"Metadata saved to {metadata_path}")
    
    return processed_data

def generate_analysis_summary(dataset, features):
    """Generate a comprehensive summary of the analysis"""
    print("=" * 60)
    print("ELECTRON MICROSCOPY DATASET ANALYSIS SUMMARY")
    print("=" * 60)
    
    if 'train' in dataset:
        print(f"\nDataset Information:")
        print(f"  - Total training samples: {len(dataset['train'])}")
        print(f"  - Available splits: {list(dataset.keys())}")
    
    if features:
        print(f"\nFeature Analysis:")
        print(f"  - Images analyzed: {len(features)}")
        
        # Calculate statistics
        means = [f['mean'] for f in features]
        stds = [f['std'] for f in features]
        edge_densities = [f['edge_density'] for f in features]
        
        print(f"  - Mean intensity: {np.mean(means):.2f} ± {np.std(means):.2f}")
        print(f"  - Standard deviation: {np.mean(stds):.2f} ± {np.std(stds):.2f}")
        print(f"  - Edge density: {np.mean(edge_densities):.4f} ± {np.std(edge_densities):.4f}")
    
    print(f"\nProcessing Pipeline:")
    print(f"  1. Image loading and preprocessing")
    print(f"  2. Noise reduction and enhancement")
    print(f"  3. Feature extraction")
    print(f"  4. Advanced image processing")
    print(f"  5. Data export and analysis")
    
    print(f"\nOutput Files:")
    print(f"  - Processed images: processed_em_data/images/")
    print(f"  - Metadata: processed_em_data/metadata.json")
    
    print("=" * 60)

if __name__ == "__main__":
    main()