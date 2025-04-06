import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path, metric, ylabel, title, output_path, last_n=10):
    # Read CSV log into a DataFrame
    df = pd.read_csv(csv_path)
    
    # If no 'epoch' column exists, use the index as epoch numbers
    if 'epoch' not in df.columns:
        df['epoch'] = df.index

    # Slice DataFrame to only include the last 'last_n' epochs
    if len(df) > last_n:
        plot_df = df.tail(last_n)
    else:
        plot_df = df

    # Plot the metric
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df['epoch'], plot_df[metric], marker='o', label=metric)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main(args):
    # Paths to CSV logs
    train_csv = os.path.join(args.log_dir, 'train_logs.csv')
    valid_csv = os.path.join(args.log_dir, 'valid_logs.csv')
    
    # Create output directory for plots if it doesn't exist
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Plot Dice Loss from training logs
    dice_output_path = os.path.join(args.plot_dir, 'dice_loss_last10.png')
    plot_metrics(
        csv_path=train_csv,
        metric='dice_loss',
        ylabel='Dice Loss',
        title='Dice Loss (Last 10 Epochs)',
        output_path=dice_output_path,
        last_n=args.last_n
    )
    
    # Plot IoU Score from validation logs
    iou_output_path = os.path.join(args.plot_dir, 'iou_score_last10.png')
    plot_metrics(
        csv_path=valid_csv,
        metric='iou_score',
        ylabel='IoU Score',
        title='IoU Score (Last 10 Epochs)',
        output_path=iou_output_path,
        last_n=args.last_n
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot recent training metrics from CSV logs.")
    parser.add_argument("--log_dir", type=str, default="/root/roadmapper/model",
                        help="Directory where train_logs.csv and valid_logs.csv are located.")
    parser.add_argument("--plot_dir", type=str, default="/root/roadmapper/model",
                        help="Directory to save the generated plots.")
    parser.add_argument("--last_n", type=int, default=10,
                        help="Number of recent epochs to plot.")
    args = parser.parse_args()
    main(args)
