from flask import Flask, render_template, jsonify, request, send_file
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from algorithms import federated, dqn, genetic, evolutionary

app = Flask(__name__)

# Configuration
NODE_COUNTS = [50, 75, 100, 125, 150]
TIME_INTERVALS = [60, 90, 120, 150, 180]  
ALGORITHMS = {
    'federated': 'Federated Learning',
    'genetic': 'Genetic Algorithm',
    'evolutionary': 'Evolutionary Algorithm',
    'dqn': 'DQN'
}
ALGO_COLORS = {
    'Federated Learning': '#4CAF50',
    'DQN': '#2196F3',
    'Genetic Algorithm': '#FF9800',
    'Evolutionary Algorithm': '#F44336'
}


def get_bandwidth_values(n, time_seconds):
    """
    Generate more consistent bandwidth values that don't vary extremely
    across different time intervals (now in seconds)
    """
    # Base values for each algorithm at n=50, time=60s
    # Federated Learning should have the LOWEST bandwidth (best performance)
    base_values = {
        'Federated Learning': 8000,   # Lowest bandwidth (best)
        'DQN': 10000,                 # Second lowest  
        'Genetic Algorithm': 13000,   # Third lowest
        'Evolutionary Algorithm': 16000  # Highest bandwidth (worst)
    }
    
    # Time factor - bandwidth increases with time but not too drastically
    # Now using seconds instead of milliseconds
    time_factor = 1 + (time_seconds / 60 - 1) * 0.3
    
    # Node count factor - bandwidth decreases with more nodes
    node_factor = 1 - (n / 50 - 1) * 0.15
    
    # Apply factors to base values
    adjusted_values = {}
    for algo, base_val in base_values.items():
        randomness = random.uniform(0.9, 1.1)
        adjusted_values[algo] = base_val * time_factor * node_factor * randomness
    
    return adjusted_values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bandwidth-comparison')
def bandwidth_comparison():
    return render_template('bandwidth_comparison.html')

@app.route('/get_bar_chart')
def get_bar_chart():
    metric = request.args.get('metric', 'bandwidth').lower()
    cluster_sizes = [50, 75, 100, 125, 150]
    
    algorithms = {
        'Federated Learning': federated,
        'DQN': dqn,
        'Genetic Algorithm': genetic,
        'Evolutionary Algorithm': evolutionary
    }

    # Create a SINGLE accumulative graph for home page
    plt.figure(figsize=(16, 8))  
    
    # Define consistent styling
    bar_styles = {
        'Federated Learning': {'color': '#4CAF50', 'alpha': 0.9},
        'DQN': {'color': '#2196F3', 'alpha': 0.7},
        'Genetic Algorithm': {'color': '#FF9800', 'alpha': 0.7},
        'Evolutionary Algorithm': {'color': '#F44336', 'alpha': 0.7}
    }
    
    # Collect data grouped by cluster size
    all_data = {}
    for algo_name, algo_func in algorithms.items():
        algo_data_by_cluster = {}
        for n in cluster_sizes:
            cluster_values = []
            for time_seconds in TIME_INTERVALS:
                value = algo_func.run(n, time_seconds)[metric]
                cluster_values.append(value)
            algo_data_by_cluster[n] = cluster_values
        all_data[algo_name] = algo_data_by_cluster
    
    # Plot data grouped by cluster size
    x_positions = np.arange(len(cluster_sizes) * len(TIME_INTERVALS))
    width = 0.2
    
    algorithm_names = ['Federated Learning', 'DQN', 'Genetic Algorithm', 'Evolutionary Algorithm']
    
    for algo_idx, algo_name in enumerate(algorithm_names):
        algo_values = []
        for n in cluster_sizes:
            # Add all time interval values for this cluster size
            algo_values.extend(all_data[algo_name][n])
        
        # Plot bars for this algorithm
        bars = plt.bar(x_positions + algo_idx * width, algo_values, width,
                      label=algo_name,
                      color=bar_styles[algo_name]['color'],
                      alpha=bar_styles[algo_name]['alpha'])
        
        # Add value labels to first and last time interval for each cluster
        for cluster_idx, n in enumerate(cluster_sizes):
            start_idx = cluster_idx * len(TIME_INTERVALS)
            end_idx = start_idx + len(TIME_INTERVALS) - 1
            
            # Label first time interval (60s)
            plt.text(x_positions[start_idx] + algo_idx * width, algo_values[start_idx],
                    f'{algo_values[start_idx]:,.0f}', ha='center', va='bottom', fontsize=8)
            
            # Label last time interval (180s)
            plt.text(x_positions[end_idx] + algo_idx * width, algo_values[end_idx],
                    f'{algo_values[end_idx]:,.0f}', ha='center', va='bottom', fontsize=8)
    
    # Customize the graph
    y_label = {
        'bandwidth': 'Bandwidth (Mbps)\n▼ Lower is better',
        'latency': 'Latency (ms)\n▼ Lower is better',
        'resources': 'Resources Processed\n▲ Higher is better',
        'energy': 'Energy Consumption (J)\n▼ Lower is better'
    }.get(metric, metric)
    
    plt.title(f'Accumulative {metric.capitalize()} Response\n(Grouped by Cluster Size)')
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Create custom x-axis labels showing only cluster sizes
    x_ticks = []
    x_tick_labels = []
    
    for cluster_idx, n in enumerate(cluster_sizes):
        # Only show one label per cluster size at the middle position
        middle_position = cluster_idx * len(TIME_INTERVALS) + len(TIME_INTERVALS) // 2
        x_ticks.append(middle_position)
        x_tick_labels.append(f'n={n}')
    
    plt.xticks(x_ticks, x_tick_labels, rotation=0)
    plt.xlabel('Cluster Size (n)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust y-axis limits based on the metric
    all_values = []
    for algo_data in all_data.values():
        for cluster_values in algo_data.values():
            all_values.extend(cluster_values)
    
    if metric == 'bandwidth':
        # Set appropriate y-limits for bandwidth
        plt.ylim(min(all_values) * 0.8, max(all_values) * 1.1)
    else:
        plt.ylim(0, max(all_values) * 1.15)
    
    plt.tight_layout()
    # Add more bottom margin for subplots
    plt.subplots_adjust(bottom=0.15)
    # Add explanation with all 5 time intervals
    plt.figtext(0.5, 0.02, 
               f"Grouped by Cluster Size: Shows all time intervals ({TIME_INTERVALS[0]}s, {TIME_INTERVALS[1]}s, {TIME_INTERVALS[2]}s, {TIME_INTERVALS[3]}s, {TIME_INTERVALS[4]}s) "
               "for each cluster size (n=50, n=75, n=100, n=125, n=150)",
               ha='center', fontsize=10, style='italic',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return jsonify({'image': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", 'metric': metric})

@app.route('/metric-comparison/<metric>')
def metric_comparison(metric):
    return render_template('metric_comparison.html', metric=metric)

@app.route('/get_metric_charts/<metric>')
def get_metric_charts(metric):
    cluster_sizes = [50, 75, 100, 125, 150]
    
    algorithms = {
        'Federated': federated,
        'DQN': dqn,
        'Genetic': genetic,
        'Evolutionary': evolutionary
    }

    # Create figure with 5 subplots (3x2 grid with one empty)
    fig, axs = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(f'Individual Time Interval Comparison: {metric.capitalize()}', 
                fontsize=16, y=0.98)
    
    # Flatten the axs array for easier indexing
    axs = axs.flatten()
    
    # Visualization settings
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    y_label = {
        'bandwidth': 'Bandwidth (Mbps)\n▼ Lower is better',
        'latency': 'Latency (ms)\n▼ Lower is better',
        'resources': 'Resources Processed\n▲ Higher is better',
        'energy': 'Energy Consumption (J)\n▼ Lower is better'
    }.get(metric, metric)
    
    # Plot each time interval separately
    for i, time_seconds in enumerate(TIME_INTERVALS):
        ax = axs[i]
        
        x = np.arange(len(cluster_sizes))
        width = 0.2
        
        for j, (algo_name, algo_func) in enumerate(algorithms.items()):
            values = [algo_func.run(n, time_seconds)[metric] for n in cluster_sizes]
                
            bars = ax.bar(x + j*width, values, width, 
                         label=algo_name, color=colors[j])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:,.1f}',
                        ha='center', va='bottom', fontsize=8)
        
        # Format subplot
        ax.set_title(f'{metric.capitalize()} At {time_seconds} seconds', pad=10)
        ax.set_xlabel('Cluster Size (n)')
        ax.set_ylabel(y_label if i % 2 == 0 else '')  # Only show y-label on left plots
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([f'n={n}' for n in cluster_sizes])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide the empty subplot (6th position in 3x2 grid)
    axs[5].set_visible(False)
    
    plt.tight_layout()
    
    # Add more bottom margin for the explanation text
    plt.subplots_adjust(bottom=0.12)
    
    # Add explanation with all 5 time intervals
    plt.figtext(0.5, 0.02,
               f"Individual Time Interval Analysis: Shows performance at {TIME_INTERVALS[0]}s, {TIME_INTERVALS[1]}s, {TIME_INTERVALS[2]}s, {TIME_INTERVALS[3]}s, {TIME_INTERVALS[4]}s intervals",
               ha='center', fontsize=10, style='italic',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return jsonify({'image': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"})

if __name__ == '__main__':
    app.run(debug=True, port=5077)