
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import csv
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

def parse_csv_results(filepath):

    results = {'acc': [], 'bac': [], 'f1': []}
    current_metric = None

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue

            if row[0] in ['a', 'acc']:
                current_metric = 'acc'
                continue
            elif row[0] in ['b', 'bac']:
                current_metric = 'bac'
                continue
            elif row[0] in ['f', 'f1']:
                current_metric = 'f1'
                continue

            if current_metric is None:
                continue

            values = []
            stds = []
            for cell in row:
                if '±' in cell:
                    parts = cell.replace('%', '').split('±')
                    values.append(float(parts[0]))
                    stds.append(float(parts[1]))

            if values:
                results[current_metric].append({
                    'values': values[:-1],
                    'stds': stds[:-1],
                    'avg': values[-1],
                    'avg_std': stds[-1]
                })

    return results

def create_visualizations(pems04_path, dblp_path, output_dir):

    pems04 = parse_csv_results(pems04_path)
    dblp = parse_csv_results(dblp_path)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    metrics = ['acc', 'bac', 'f1']
    titles = ['Accuracy', 'Balanced Accuracy', 'F1 Score']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        pems04_final = pems04[metric][-2]
        dblp_final = dblp[metric][-2]

        tasks = list(range(10))

        ax.plot(tasks, pems04_final['values'], 'o-', color='#2ecc71',
                label='PEMS04', linewidth=2, markersize=6)
        ax.fill_between(tasks,
                       np.array(pems04_final['values']) - np.array(pems04_final['stds']),
                       np.array(pems04_final['values']) + np.array(pems04_final['stds']),
                       alpha=0.2, color='#2ecc71')

        ax.plot(tasks, dblp_final['values'], 's-', color='#3498db',
                label='DBLP', linewidth=2, markersize=6)
        ax.fill_between(tasks,
                       np.array(dblp_final['values']) - np.array(dblp_final['stds']),
                       np.array(dblp_final['values']) + np.array(dblp_final['stds']),
                       alpha=0.2, color='#3498db')

        ax.set_xlabel('Test Task')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'{title} on Each Task\n(After Training on All Tasks)')
        ax.legend(loc='lower right')
        ax.set_xticks(tasks)
        ax.set_ylim([55, 95])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_performance_by_task.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 1_performance_by_task.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        pems04_avgs = [pems04[metric][i]['avg'] for i in range(10)]
        pems04_stds = [pems04[metric][i]['avg_std'] for i in range(10)]
        dblp_avgs = [dblp[metric][i]['avg'] for i in range(10)]
        dblp_stds = [dblp[metric][i]['avg_std'] for i in range(10)]

        tasks = list(range(10))

        ax.plot(tasks, pems04_avgs, 'o-', color='#2ecc71',
                label='PEMS04', linewidth=2, markersize=6)
        ax.fill_between(tasks,
                       np.array(pems04_avgs) - np.array(pems04_stds),
                       np.array(pems04_avgs) + np.array(pems04_stds),
                       alpha=0.2, color='#2ecc71')

        ax.plot(tasks, dblp_avgs, 's-', color='#3498db',
                label='DBLP', linewidth=2, markersize=6)
        ax.fill_between(tasks,
                       np.array(dblp_avgs) - np.array(dblp_stds),
                       np.array(dblp_avgs) + np.array(dblp_stds),
                       alpha=0.2, color='#3498db')

        ax.set_xlabel('Training Task')
        ax.set_ylabel(f'Average {title} (%)')
        ax.set_title(f'Average {title}\n(Learning Curve)')
        ax.legend(loc='lower right')
        ax.set_xticks(tasks)
        ax.set_ylim([65, 90])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_learning_curve.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 2_learning_curve.png")

    fig, ax = plt.subplots(figsize=(10, 5))

    pems04_forgetting = {m: pems04[m][-1]['values'] for m in metrics}
    dblp_forgetting = {m: dblp[m][-1]['values'] for m in metrics}

    x = np.arange(10)
    width = 0.35

    bars1 = ax.bar(x - width/2, pems04_forgetting['f1'], width,
                   label='PEMS04', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, dblp_forgetting['f1'], width,
                   label='DBLP', color='#3498db', alpha=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Forgetting (%)')
    ax.set_title('Forgetting Rate by Task (F1 Score)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i}' for i in range(10)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    pems04_avg_forget = np.mean(pems04_forgetting['f1'])
    dblp_avg_forget = np.mean(dblp_forgetting['f1'])
    ax.axhline(y=pems04_avg_forget, color='#27ae60', linestyle='--',
               label=f'PEMS04 avg: {pems04_avg_forget:.2f}%')
    ax.axhline(y=dblp_avg_forget, color='#2980b9', linestyle='--',
               label=f'DBLP avg: {dblp_avg_forget:.2f}%')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_forgetting_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 3_forgetting_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['Accuracy', 'Balanced Acc', 'F1 Score', 'Forgetting']
    pems04_values = [
        pems04['acc'][-2]['avg'],
        pems04['bac'][-2]['avg'],
        pems04['f1'][-2]['avg'],
        pems04['f1'][-1]['avg']
    ]
    pems04_stds = [
        pems04['acc'][-2]['avg_std'],
        pems04['bac'][-2]['avg_std'],
        pems04['f1'][-2]['avg_std'],
        pems04['f1'][-1]['avg_std']
    ]
    dblp_values = [
        dblp['acc'][-2]['avg'],
        dblp['bac'][-2]['avg'],
        dblp['f1'][-2]['avg'],
        dblp['f1'][-1]['avg']
    ]
    dblp_stds = [
        dblp['acc'][-2]['avg_std'],
        dblp['bac'][-2]['avg_std'],
        dblp['f1'][-2]['avg_std'],
        dblp['f1'][-1]['avg_std']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, pems04_values, width, yerr=pems04_stds,
                   label='PEMS04', color='#2ecc71', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, dblp_values, width, yerr=dblp_stds,
                   label='DBLP', color='#3498db', alpha=0.8, capsize=5)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Final Performance Summary\n(PEMS04 vs DBLP)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1, pems04_values)
    add_labels(bars2, dblp_values)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_summary_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 4_summary_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pems04_matrix = np.array([pems04['f1'][i]['values'] for i in range(10)])
    im1 = axes[0].imshow(pems04_matrix, cmap='YlGn', aspect='auto', vmin=60, vmax=85)
    axes[0].set_xlabel('Test Task')
    axes[0].set_ylabel('Training Task')
    axes[0].set_title('PEMS04 - F1 Score Matrix')
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    plt.colorbar(im1, ax=axes[0], label='F1 (%)')

    dblp_matrix = np.array([dblp['f1'][i]['values'] for i in range(10)])
    im2 = axes[1].imshow(dblp_matrix, cmap='YlGnBu', aspect='auto', vmin=60, vmax=90)
    axes[1].set_xlabel('Test Task')
    axes[1].set_ylabel('Training Task')
    axes[1].set_title('DBLP - F1 Score Matrix')
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(10))
    plt.colorbar(im2, ax=axes[1], label='F1 (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_performance_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 5_performance_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    categories = ['Accuracy', 'Balanced\nAccuracy', 'F1 Score',
                  'Low\nForgetting', 'Stability']

    pems04_radar = [
        pems04['acc'][-2]['avg'] / 100,
        pems04['bac'][-2]['avg'] / 100,
        pems04['f1'][-2]['avg'] / 100,
        1 - pems04['f1'][-1]['avg'] / 10,
        1 - pems04['f1'][-2]['avg_std'] / 10
    ]
    dblp_radar = [
        dblp['acc'][-2]['avg'] / 100,
        dblp['bac'][-2]['avg'] / 100,
        dblp['f1'][-2]['avg'] / 100,
        1 - dblp['f1'][-1]['avg'] / 10,
        1 - dblp['f1'][-2]['avg_std'] / 10
    ]

    pems04_radar += pems04_radar[:1]
    dblp_radar += dblp_radar[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, pems04_radar, 'o-', linewidth=2, label='PEMS04', color='#2ecc71')
    ax.fill(angles, pems04_radar, alpha=0.25, color='#2ecc71')
    ax.plot(angles, dblp_radar, 's-', linewidth=2, label='DBLP', color='#3498db')
    ax.fill(angles, dblp_radar, alpha=0.25, color='#3498db')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Multi-dimensional Performance Comparison', y=1.08)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_radar_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 6_radar_comparison.png")

    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == '__main__':
    pems04_path = 'results/PEMS04/DYGRA_GCN_PEMS04_reduction_0.5.csv'
    dblp_path = 'results/DBLP/DYGRA_GCN_DBLP_reduction_0.5.csv'
    output_dir = 'visualizations'

    create_visualizations(pems04_path, dblp_path, output_dir)
