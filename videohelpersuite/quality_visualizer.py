"""
Video Quality Visualization Module
===================================
Generates comprehensive visual reports for video quality analysis.

Supports:
- Seam discontinuity charts
- Temporal drift plots
- Quality distribution histograms
- Multi-metric dashboards
"""

import io
from typing import List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import matplotlib

# This prevents the "cannot see result" issue on servers
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image

from .lpips_analyzer import SeamAnalysis, DriftAnalysis, get_summary_statistics

class QualityVisualizer:
    """
    Creates publication-quality visualizations for video quality metrics.
    """
    
    # Color scheme
    COLOR_EXCELLENT = '#2ecc71'  # Green
    COLOR_GOOD = '#3498db'       # Blue
    COLOR_NOTICEABLE = '#f39c12' # Orange
    COLOR_POOR = '#e74c3c'       # Red
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            plt.style.use('default')
    
    def create_seam_quality_chart(
        self,
        seam_analyses: List[SeamAnalysis],
        title: str = "Video Merge Quality Analysis"
    ) -> torch.Tensor:
        """
        Create comprehensive seam quality visualization.
        
        Args:
            seam_analyses: List of seam analysis results
            title: Chart title
            
        Returns:
            Tensor representation of chart image [1, H, W, 3]
        """
        if not seam_analyses:
            return self._create_empty_chart("No seams to analyze")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extract data
        seam_indices = [s.seam_index for s in seam_analyses]
        scores = [s.lpips_score for s in seam_analyses]
        colors = [self._get_quality_color(s.quality_rating) for s in seam_analyses]
        
        # Top plot: Line chart with markers
        ax1.plot(seam_indices, scores, 
                color='#34495e', linewidth=2, alpha=0.7, 
                label='LPIPS Distance')
        ax1.scatter(seam_indices, scores, 
                   c=colors, s=100, edgecolors='black', linewidths=1.5,
                   zorder=5)
        
        # Reference lines
        ax1.axhline(y=0.05, color=self.COLOR_EXCELLENT, 
                   linestyle='--', alpha=0.6, label='Excellent (< 0.05)')
        ax1.axhline(y=0.1, color=self.COLOR_GOOD, 
                   linestyle='--', alpha=0.6, label='Good (< 0.1)')
        ax1.axhline(y=0.3, color=self.COLOR_NOTICEABLE, 
                   linestyle='--', alpha=0.6, label='Noticeable (< 0.3)')
        
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Seam Number (Batch Transition)', fontsize=11)
        ax1.set_ylabel('LPIPS Distance (Lower = Better)', fontsize=11)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_ylim(bottom=0)
        
        # Bottom plot: Bar chart with quality categories
        quality_colors_bar = colors
        bars = ax2.bar(seam_indices, scores, color=quality_colors_bar, 
                      edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels on bars
        for idx, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Seam Number', fontsize=11)
        ax2.set_ylabel('LPIPS Score', fontsize=11)
        ax2.set_title('Individual Seam Scores', fontsize=12, pad=10)
        ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        return self._fig_to_tensor(fig)
    
    def create_drift_analysis_chart(
        self,
        drift_analysis: DriftAnalysis,
        title: str = "Temporal Drift Analysis"
    ) -> torch.Tensor:
        """
        Create temporal drift visualization.
        
        Args:
            drift_analysis: DriftAnalysis object
            title: Chart title
            
        Returns:
            Tensor representation of chart image
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frames = drift_analysis.frame_numbers
        scores = drift_analysis.lpips_scores
        
        # Main drift line
        ax.plot(frames, scores, 
               color='#e74c3c', linewidth=2.5, alpha=0.8,
               label='Drift from Reference')
        ax.fill_between(frames, 0, scores, alpha=0.2, color='#e74c3c')
        
        # Highlight reference frame
        ref_frame = drift_analysis.reference_frame_index
        ax.axvline(x=ref_frame, color='#2ecc71', 
                  linestyle='--', linewidth=2, 
                  label=f'Reference Frame ({ref_frame})')
        
        # Add mean line
        ax.axhline(y=drift_analysis.mean_drift, 
                  color='#3498db', linestyle=':', linewidth=2,
                  label=f'Mean Drift ({drift_analysis.mean_drift:.3f})')
        
        # Annotate max drift point
        max_idx = np.argmax(scores)
        max_frame = frames[max_idx]
        max_score = scores[max_idx]
        ax.annotate(f'Max Drift\n{max_score:.3f}',
                   xy=(max_frame, max_score),
                   xytext=(max_frame, max_score + 0.05),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_title(f"{title} - Trend: {drift_analysis.drift_trend.upper()}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=11)
        ax.set_ylabel('LPIPS Distance from Reference', fontsize=11)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        return self._fig_to_tensor(fig)
    
    def create_summary_dashboard(
        self,
        seam_analyses: List[SeamAnalysis],
        project_name: str = "Video Project"
    ) -> torch.Tensor:
        """
        Create comprehensive quality dashboard with multiple metrics.
        
        Args:
            seam_analyses: List of seam analysis results
            project_name: Project name for title
            
        Returns:
            Tensor representation of dashboard image
        """
        if not seam_analyses:
            return self._create_empty_chart("No data for dashboard")
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get statistics
        stats = get_summary_statistics(seam_analyses)
        scores = [s.lpips_score for s in seam_analyses]
        
        # 1. Main timeline (spans top row)
        ax_timeline = fig.add_subplot(gs[0, :])
        seam_indices = [s.seam_index for s in seam_analyses]
        colors = [self._get_quality_color(s.quality_rating) for s in seam_analyses]
        
        ax_timeline.plot(seam_indices, scores, 
                        color='#2c3e50', linewidth=2.5, alpha=0.8)
        ax_timeline.scatter(seam_indices, scores, 
                           c=colors, s=120, edgecolors='black', linewidths=2, zorder=5)
        ax_timeline.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax_timeline.set_title(f'{project_name} - Quality Timeline', 
                             fontsize=16, fontweight='bold')
        ax_timeline.set_xlabel('Seam Index')
        ax_timeline.set_ylabel('LPIPS Score')
        ax_timeline.grid(True, alpha=0.3)
        
        # 2. Score distribution histogram
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_hist.hist(scores, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
        ax_hist.axvline(x=stats['mean_score'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {stats['mean_score']:.3f}")
        ax_hist.set_title('Score Distribution', fontweight='bold')
        ax_hist.set_xlabel('LPIPS Score')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3, axis='y')
        
        # 3. Quality category pie chart
        ax_pie = fig.add_subplot(gs[1, 1])
        quality_dist = stats['quality_distribution']
        labels = list(quality_dist.keys())
        sizes = list(quality_dist.values())
        colors_pie = [self._get_quality_color(label) for label in labels]
        
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors_pie,
                                               autopct='%1.1f%%', startangle=90,
                                               textprops={'fontweight': 'bold'})
        ax_pie.set_title('Quality Distribution', fontweight='bold')
        
        # 4. Statistics table
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.axis('off')
        
        stats_text = f"""
        SUMMARY STATISTICS
        {'='*30}
        Total Seams: {stats['total_seams']}
        
        Mean Score: {stats['mean_score']:.4f}
        Std Dev: {stats['std_score']:.4f}
        
        Best Seam: #{stats['best_seam_index']}
          Score: {stats['min_score']:.4f}
        
        Worst Seam: #{stats['worst_seam_index']}
          Score: {stats['max_score']:.4f}
        
        Quality Range: {stats['max_score'] - stats['min_score']:.4f}
        """
        
        ax_stats.text(0.1, 0.5, stats_text, 
                     fontsize=10, verticalalignment='center',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Problematic seams highlight (bottom row)
        ax_problems = fig.add_subplot(gs[2, :])
        
        # Identify problematic seams (score > 0.2)
        problematic = [(s.seam_index, s.lpips_score, s.quality_rating) 
                      for s in seam_analyses if s.lpips_score > 0.2]
        
        if problematic:
            prob_indices, prob_scores, prob_ratings = zip(*problematic)
            prob_colors = [self._get_quality_color(r) for r in prob_ratings]
            
            bars = ax_problems.bar(prob_indices, prob_scores, 
                                  color=prob_colors, alpha=0.8, edgecolor='black', linewidth=2)
            
            for bar, score in zip(bars, prob_scores):
                height = bar.get_height()
                ax_problems.text(bar.get_x() + bar.get_width()/2., height,
                               f'{score:.3f}',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax_problems.set_title('⚠️ Problematic Seams (Score > 0.2)', 
                                 fontsize=14, fontweight='bold', color='#e74c3c')
            ax_problems.set_xlabel('Seam Index')
            ax_problems.set_ylabel('LPIPS Score')
            ax_problems.grid(True, alpha=0.3, axis='y')
        else:
            ax_problems.text(0.5, 0.5, '[OK] All Seams Look Good!', 
                           transform=ax_problems.transAxes,
                           fontsize=20, ha='center', va='center',
                           color=self.COLOR_EXCELLENT, fontweight='bold')
            ax_problems.axis('off')
        
        plt.suptitle(f'Video Quality Dashboard - {project_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        return self._fig_to_tensor(fig)
    
    def _get_quality_color(self, quality_rating: str) -> str:
        """Map quality rating to color."""
        color_map = {
            'excellent': self.COLOR_EXCELLENT,
            'good': self.COLOR_GOOD,
            'noticeable': self.COLOR_NOTICEABLE,
            'poor': self.COLOR_POOR,
            'unknown': '#95a5a6'
        }
        return color_map.get(quality_rating, '#95a5a6')
    
    def _fig_to_tensor(self, fig: plt.Figure) -> torch.Tensor:
        """
        Convert matplotlib figure to tensor.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Tensor with shape [1, H, W, 3], values in [0, 1]
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        img = Image.open(buf).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        
        return torch.from_numpy(img_np).unsqueeze(0)
    
    def _create_empty_chart(self, message: str) -> torch.Tensor:
        """Create placeholder chart when no data is available."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message,
               ha='center', va='center',
               fontsize=16, fontweight='bold',
               transform=ax.transAxes)
        ax.axis('off')
        return self._fig_to_tensor(fig)
    
    def save_report(
        self,
        seam_analyses: List[SeamAnalysis],
        output_path: str,
        project_name: str = "Video Project"
    ) -> None:
        """
        Save comprehensive quality report to file.
        
        Args:
            seam_analyses: List of seam analysis results
            output_path: Path to save report (PNG file)
            project_name: Project name
        """
        dashboard_tensor = self.create_summary_dashboard(seam_analyses, project_name)
        
        # Convert tensor back to PIL and save
        img_np = (dashboard_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(output_path, 'PNG', optimize=True)
        
        print(f"[QualityVisualizer] Report saved to: {output_path}")


def create_comparison_chart(
    analyses_dict: dict,
    title: str = "Multi-Project Comparison"
) -> torch.Tensor:
    """
    Compare quality metrics across multiple projects.
    
    Args:
        analyses_dict: Dictionary mapping project names to SeamAnalysis lists
        title: Chart title
        
    Returns:
        Tensor representation of comparison chart
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_offset = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(analyses_dict)))
    
    for (project_name, analyses), color in zip(analyses_dict.items(), colors):
        if not analyses:
            continue
        
        stats = get_summary_statistics(analyses)
        
        # Plot box plot for this project
        scores = [s.lpips_score for s in analyses]
        positions = [x_offset]
        bp = ax.boxplot([scores], positions=positions, widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Add label
        ax.text(x_offset, -0.05, project_name, 
               ha='center', fontsize=10, fontweight='bold', rotation=45)
        
        x_offset += 1
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('LPIPS Score Distribution', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([])
    
    plt.tight_layout()
    
    # Convert to tensor
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    img = Image.open(buf).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    
    return torch.from_numpy(img_np).unsqueeze(0)
