#!/usr/bin/env python3
"""
Image Property Comparison Tool
Compares two enhanced images and saves detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, metrics
from skimage.color import rgb2gray
from scipy import stats
import os
from datetime import datetime


def compute_brightness(img):
    """Compute mean brightness of image."""
    gray = rgb2gray(img)
    return np.mean(gray)


def compute_contrast(img):
    """Compute RMS contrast."""
    gray = rgb2gray(img)
    mean = np.mean(gray)
    return np.sqrt(np.mean((gray - mean) ** 2))


def compute_colorfulness(img):
    """Compute colorfulness metric (Hasler & Süsstrunk, 2003)."""
    # Convert to float
    img_float = img.astype(np.float64) / 255.0
    
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    
    # Compute rg and yb
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    # Compute mean and std
    mean_rg = np.mean(rg)
    std_rg = np.std(rg)
    mean_yb = np.mean(yb)
    std_yb = np.std(yb)
    
    # Colorfulness metric
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    
    return colorfulness


def compute_dark_region_stats(img, threshold_percentile=20):
    """Compute statistics for dark regions."""
    gray = rgb2gray(img)
    threshold = np.percentile(gray, threshold_percentile)
    dark_mask = gray < threshold
    
    if np.sum(dark_mask) == 0:
        return {
            'mean': 0,
            'std': 0,
            'pixel_count': 0,
            'percentage': 0
        }
    
    dark_pixels = gray[dark_mask]
    
    return {
        'mean': np.mean(dark_pixels),
        'std': np.std(dark_pixels),
        'pixel_count': np.sum(dark_mask),
        'percentage': (np.sum(dark_mask) / gray.size) * 100
    }


def compute_channel_stats(img):
    """Compute statistics for each RGB channel."""
    stats_dict = {}
    channel_names = ['R', 'G', 'B']
    
    for i, name in enumerate(channel_names):
        channel = img[:, :, i]
        stats_dict[name] = {
            'mean': np.mean(channel),
            'std': np.std(channel),
            'min': np.min(channel),
            'max': np.max(channel),
            'median': np.median(channel)
        }
    
    return stats_dict


def compute_histogram_properties(img, bins=256):
    """Compute histogram properties."""
    gray = rgb2gray(img)
    hist, bin_edges = np.histogram(gray, bins=bins, range=(0, 255))
    
    # Normalize histogram
    hist = hist / np.sum(hist)
    
    # Compute entropy
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    
    # Find peak
    peak_idx = np.argmax(hist)
    peak_value = bin_edges[peak_idx]
    
    return {
        'histogram': hist,
        'bin_edges': bin_edges,
        'entropy': entropy,
        'peak_value': peak_value
    }


def compare_images(img1_path, img2_path, output_dir=None):
    """
    Compare two images and generate comprehensive analysis.
    
    Parameters:
    -----------
    img1_path : str
        Path to first image
    img2_path : str
        Path to second image
    output_dir : str
        Directory to save results (default: same as img1)
    """
    print("=" * 70)
    print("Image Property Comparison")
    print("=" * 70)
    
    # Load images
    print(f"\nLoading images...")
    print(f"  Image 1: {os.path.basename(img1_path)}")
    print(f"  Image 2: {os.path.basename(img2_path)}")
    
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Ensure same size (resize if needed)
    if img1.shape != img2.shape:
        print(f"\n⚠ Warning: Images have different sizes!")
        print(f"  Image 1: {img1.shape}")
        print(f"  Image 2: {img2.shape}")
        print("  Resizing Image 2 to match Image 1...")
        from skimage.transform import resize
        img2 = resize(img2, img1.shape, anti_aliasing=True)
        img2 = (img2 * 255).astype(np.uint8)
    
    print(f"  Image 1 shape: {img1.shape}, dtype: {img1.dtype}, range: [{img1.min()}, {img1.max()}]")
    print(f"  Image 2 shape: {img2.shape}, dtype: {img2.dtype}, range: [{img2.min()}, {img2.max()}]")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(img1_path)
    
    base_name1 = os.path.splitext(os.path.basename(img1_path))[0]
    base_name2 = os.path.splitext(os.path.basename(img2_path))[0]
    comparison_name = f"{base_name1}_vs_{base_name2}"
    
    # Initialize results dictionary
    results = {
        'image1_name': os.path.basename(img1_path),
        'image2_name': os.path.basename(img2_path),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n" + "=" * 70)
    print("1. BASIC PROPERTIES")
    print("=" * 70)
    
    results['basic'] = {
        'image1': {
            'shape': img1.shape,
            'dtype': str(img1.dtype),
            'min': int(img1.min()),
            'max': int(img1.max()),
            'mean': float(np.mean(img1))
        },
        'image2': {
            'shape': img2.shape,
            'dtype': str(img2.dtype),
            'min': int(img2.min()),
            'max': int(img2.max()),
            'mean': float(np.mean(img2))
        }
    }
    
    print(f"\nImage 1 ({results['image1_name']}):")
    print(f"  Shape: {img1.shape}")
    print(f"  Dtype: {img1.dtype}")
    print(f"  Range: [{img1.min()}, {img1.max()}]")
    print(f"  Mean: {np.mean(img1):.2f}")
    
    print(f"\nImage 2 ({results['image2_name']}):")
    print(f"  Shape: {img2.shape}")
    print(f"  Dtype: {img2.dtype}")
    print(f"  Range: [{img2.min()}, {img2.max()}]")
    print(f"  Mean: {np.mean(img2):.2f}")
    
    print("\n" + "=" * 70)
    print("2. BRIGHTNESS & EXPOSURE")
    print("=" * 70)
    
    brightness1 = compute_brightness(img1)
    brightness2 = compute_brightness(img2)
    brightness_diff = brightness2 - brightness1
    brightness_pct = (brightness_diff / brightness1) * 100 if brightness1 > 0 else 0
    
    results['brightness'] = {
        'image1': float(brightness1),
        'image2': float(brightness2),
        'difference': float(brightness_diff),
        'percentage_change': float(brightness_pct)
    }
    
    print(f"\nMean Brightness (0-255):")
    print(f"  Image 1: {brightness1:.2f}")
    print(f"  Image 2: {brightness2:.2f}")
    print(f"  Difference: {brightness_diff:+.2f} ({brightness_pct:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("3. CONTRAST")
    print("=" * 70)
    
    contrast1 = compute_contrast(img1)
    contrast2 = compute_contrast(img2)
    contrast_diff = contrast2 - contrast1
    contrast_pct = (contrast_diff / contrast1) * 100 if contrast1 > 0 else 0
    
    results['contrast'] = {
        'image1': float(contrast1),
        'image2': float(contrast2),
        'difference': float(contrast_diff),
        'percentage_change': float(contrast_pct)
    }
    
    print(f"\nRMS Contrast:")
    print(f"  Image 1: {contrast1:.2f}")
    print(f"  Image 2: {contrast2:.2f}")
    print(f"  Difference: {contrast_diff:+.2f} ({contrast_pct:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("4. COLOR PROPERTIES")
    print("=" * 70)
    
    colorfulness1 = compute_colorfulness(img1)
    colorfulness2 = compute_colorfulness(img2)
    
    channel_stats1 = compute_channel_stats(img1)
    channel_stats2 = compute_channel_stats(img2)
    
    results['color'] = {
        'colorfulness': {
            'image1': float(colorfulness1),
            'image2': float(colorfulness2),
            'difference': float(colorfulness2 - colorfulness1)
        },
        'channels': {
            'image1': channel_stats1,
            'image2': channel_stats2
        }
    }
    
    print(f"\nColorfulness:")
    print(f"  Image 1: {colorfulness1:.4f}")
    print(f"  Image 2: {colorfulness2:.4f}")
    print(f"  Difference: {colorfulness2 - colorfulness1:+.4f}")
    
    print(f"\nChannel Statistics:")
    print(f"\n  Image 1:")
    for ch, stats in channel_stats1.items():
        print(f"    {ch}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    print(f"\n  Image 2:")
    for ch, stats in channel_stats2.items():
        print(f"    {ch}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    print("\n" + "=" * 70)
    print("5. DARK REGION ANALYSIS (Background Preservation)")
    print("=" * 70)
    
    dark_stats1 = compute_dark_region_stats(img1, threshold_percentile=20)
    dark_stats2 = compute_dark_region_stats(img2, threshold_percentile=20)
    
    results['dark_regions'] = {
        'image1': dark_stats1,
        'image2': dark_stats2,
        'improvement': {
            'mean_brightness': float(dark_stats2['mean'] - dark_stats1['mean']),
            'percentage': float(((dark_stats2['mean'] - dark_stats1['mean']) / dark_stats1['mean']) * 100) if dark_stats1['mean'] > 0 else 0
        }
    }
    
    print(f"\nDark Region Statistics (bottom 20% of pixels):")
    print(f"\n  Image 1:")
    print(f"    Mean brightness: {dark_stats1['mean']:.2f}")
    print(f"    Std deviation: {dark_stats1['std']:.2f}")
    print(f"    Pixel count: {dark_stats1['pixel_count']} ({dark_stats1['percentage']:.1f}%)")
    
    print(f"\n  Image 2:")
    print(f"    Mean brightness: {dark_stats2['mean']:.2f}")
    print(f"    Std deviation: {dark_stats2['std']:.2f}")
    print(f"    Pixel count: {dark_stats2['pixel_count']} ({dark_stats2['percentage']:.1f}%)")
    
    improvement = dark_stats2['mean'] - dark_stats1['mean']
    improvement_pct = (improvement / dark_stats1['mean']) * 100 if dark_stats1['mean'] > 0 else 0
    print(f"\n  Improvement:")
    print(f"    Brightness increase: {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("6. HISTOGRAM ANALYSIS")
    print("=" * 70)
    
    hist1 = compute_histogram_properties(img1)
    hist2 = compute_histogram_properties(img2)
    
    results['histogram'] = {
        'image1': {
            'entropy': float(hist1['entropy']),
            'peak_value': float(hist1['peak_value'])
        },
        'image2': {
            'entropy': float(hist2['entropy']),
            'peak_value': float(hist2['peak_value'])
        }
    }
    
    print(f"\nHistogram Entropy (information content):")
    print(f"  Image 1: {hist1['entropy']:.4f}")
    print(f"  Image 2: {hist2['entropy']:.4f}")
    print(f"  Difference: {hist2['entropy'] - hist1['entropy']:+.4f}")
    
    print(f"\nHistogram Peak (most common brightness):")
    print(f"  Image 1: {hist1['peak_value']:.2f}")
    print(f"  Image 2: {hist2['peak_value']:.2f}")
    
    print("\n" + "=" * 70)
    print("7. STRUCTURAL SIMILARITY")
    print("=" * 70)
    
    # Convert to float for SSIM
    img1_float = img1.astype(np.float64) / 255.0
    img2_float = img2.astype(np.float64) / 255.0
    
    ssim_value = metrics.structural_similarity(
        rgb2gray(img1_float),
        rgb2gray(img2_float),
        data_range=1.0
    )
    
    # Compute MSE and PSNR
    mse = np.mean((img1_float - img2_float) ** 2)
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    
    results['similarity'] = {
        'ssim': float(ssim_value),
        'mse': float(mse),
        'psnr': float(psnr)
    }
    
    print(f"\nStructural Similarity Index (SSIM): {ssim_value:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    
    # Generate visualization
    print("\n" + "=" * 70)
    print("8. GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Side-by-side comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img1)
    ax1.set_title(f"Image 1: {results['image1_name']}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(img2)
    ax2.set_title(f"Image 2: {results['image2_name']}", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Difference image
    ax3 = plt.subplot(2, 3, 3)
    diff_img = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8)
    ax3.imshow(diff_img, cmap='hot')
    ax3.set_title("Absolute Difference (Hot Colormap)", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Histogram comparison
    ax4 = plt.subplot(2, 3, 4)
    gray1 = rgb2gray(img1)
    gray2 = rgb2gray(img2)
    ax4.hist(gray1.flatten(), bins=50, alpha=0.6, label='Image 1', color='blue', density=True)
    ax4.hist(gray2.flatten(), bins=50, alpha=0.6, label='Image 2', color='red', density=True)
    ax4.set_xlabel('Brightness')
    ax4.set_ylabel('Density')
    ax4.set_title('Brightness Histogram Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Channel comparison
    ax5 = plt.subplot(2, 3, 5)
    channels = ['R', 'G', 'B']
    img1_means = [np.mean(img1[:, :, i]) for i in range(3)]
    img2_means = [np.mean(img2[:, :, i]) for i in range(3)]
    x = np.arange(len(channels))
    width = 0.35
    ax5.bar(x - width/2, img1_means, width, label='Image 1', color='blue', alpha=0.7)
    ax5.bar(x + width/2, img2_means, width, label='Image 2', color='red', alpha=0.7)
    ax5.set_xlabel('Channel')
    ax5.set_ylabel('Mean Value')
    ax5.set_title('Channel Mean Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(channels)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Metrics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    COMPARISON SUMMARY
    
    Brightness:
      Image 1: {brightness1:.2f}
      Image 2: {brightness2:.2f}
      Change: {brightness_diff:+.2f} ({brightness_pct:+.1f}%)
    
    Contrast:
      Image 1: {contrast1:.2f}
      Image 2: {contrast2:.2f}
      Change: {contrast_diff:+.2f} ({contrast_pct:+.1f}%)
    
    Dark Regions:
      Image 1: {dark_stats1['mean']:.2f}
      Image 2: {dark_stats2['mean']:.2f}
      Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)
    
    Colorfulness:
      Image 1: {colorfulness1:.4f}
      Image 2: {colorfulness2:.4f}
    
    SSIM: {ssim_value:.4f}
    PSNR: {psnr:.2f} dB
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    
    # Save visualization
    comparison_plot_path = os.path.join(output_dir, f"{comparison_name}_comparison.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison plot: {comparison_plot_path}")
    plt.close()
    
    # Save histogram plot separately
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.hist(gray1.flatten(), bins=100, alpha=0.6, label='Image 1', color='blue', density=True)
    ax.hist(gray2.flatten(), bins=100, alpha=0.6, label='Image 2', color='red', density=True)
    ax.set_xlabel('Brightness Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Detailed Brightness Histogram Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    histogram_path = os.path.join(output_dir, f"{comparison_name}_histogram.png")
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    print(f"  Saved histogram plot: {histogram_path}")
    plt.close()
    
    # Save results to text file
    print("\n" + "=" * 70)
    print("9. SAVING RESULTS")
    print("=" * 70)
    
    results_path = os.path.join(output_dir, f"{comparison_name}_results.txt")
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IMAGE PROPERTY COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Image 1: {results['image1_name']}\n")
        f.write(f"Image 2: {results['image2_name']}\n\n")
        
        f.write("1. BASIC PROPERTIES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Image 1: Shape={img1.shape}, Range=[{img1.min()}, {img1.max()}], Mean={np.mean(img1):.2f}\n")
        f.write(f"Image 2: Shape={img2.shape}, Range=[{img2.min()}, {img2.max()}], Mean={np.mean(img2):.2f}\n\n")
        
        f.write("2. BRIGHTNESS & EXPOSURE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Image 1: {brightness1:.2f}\n")
        f.write(f"Image 2: {brightness2:.2f}\n")
        f.write(f"Difference: {brightness_diff:+.2f} ({brightness_pct:+.1f}%)\n\n")
        
        f.write("3. CONTRAST\n")
        f.write("-" * 70 + "\n")
        f.write(f"Image 1: {contrast1:.2f}\n")
        f.write(f"Image 2: {contrast2:.2f}\n")
        f.write(f"Difference: {contrast_diff:+.2f} ({contrast_pct:+.1f}%)\n\n")
        
        f.write("4. COLOR PROPERTIES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Colorfulness - Image 1: {colorfulness1:.4f}, Image 2: {colorfulness2:.4f}\n")
        f.write(f"Channel Means:\n")
        f.write(f"  Image 1 - R: {channel_stats1['R']['mean']:.2f}, G: {channel_stats1['G']['mean']:.2f}, B: {channel_stats1['B']['mean']:.2f}\n")
        f.write(f"  Image 2 - R: {channel_stats2['R']['mean']:.2f}, G: {channel_stats2['G']['mean']:.2f}, B: {channel_stats2['B']['mean']:.2f}\n\n")
        
        f.write("5. DARK REGION ANALYSIS (Background Preservation)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Image 1 - Mean: {dark_stats1['mean']:.2f}, Std: {dark_stats1['std']:.2f}, Pixels: {dark_stats1['pixel_count']} ({dark_stats1['percentage']:.1f}%)\n")
        f.write(f"Image 2 - Mean: {dark_stats2['mean']:.2f}, Std: {dark_stats2['std']:.2f}, Pixels: {dark_stats2['pixel_count']} ({dark_stats2['percentage']:.1f}%)\n")
        f.write(f"Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)\n\n")
        
        f.write("6. HISTOGRAM ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Entropy - Image 1: {hist1['entropy']:.4f}, Image 2: {hist2['entropy']:.4f}\n")
        f.write(f"Peak - Image 1: {hist1['peak_value']:.2f}, Image 2: {hist2['peak_value']:.2f}\n\n")
        
        f.write("7. STRUCTURAL SIMILARITY\n")
        f.write("-" * 70 + "\n")
        f.write(f"SSIM: {ssim_value:.4f}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"PSNR: {psnr:.2f} dB\n\n")
    
    print(f"  Saved results text: {results_path}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - Comparison plot: {comparison_name}_comparison.png")
    print(f"  - Histogram plot: {comparison_name}_histogram.png")
    print(f"  - Results text: {comparison_name}_results.txt")
    
    return results


if __name__ == '__main__':
    # Image paths
    img1_path = "/media/hp/c587a0ea-5c63-499c-a609-e5e5362a9766/data/LIME/12_enhanced.jpg"
    img2_path = "/media/hp/c587a0ea-5c63-499c-a609-e5e5362a9766/data/LIME/12_zerodce.jpg"
    
    # Output directory
    output_dir = "/media/hp/c587a0ea-5c63-499c-a609-e5e5362a9766/data/LIME/"
    
    # Run comparison
    results = compare_images(img1_path, img2_path, output_dir)
    
    print("\n✓ Done!")

