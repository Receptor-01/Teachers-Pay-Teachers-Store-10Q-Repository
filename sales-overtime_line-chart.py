#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import logging
import os
from datetime import datetime, timedelta
import subprocess
import numpy as np

# Set up logging
logging.basicConfig(
    filename='/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/sales_trendline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def notify(title, text):
    """Send a system notification on macOS."""
    subprocess.call(['osascript', '-e', f'display notification "{text}" with title "{title}"'])

def save_sales_trendline_pdf():
    """Read earnings data, create an earnings trendline chart, and save it as a PDF."""
    try:
        # Start logging and notify
        logging.info('☢︎ SCRIPT START ------------------------------')
        notify('GENERATING EARNINGS TRENDLINE....', 'The script has started running.')

        # Define file paths
        csv_file_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/sales-report.csv'
        pdf_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/earnings_trendline.pdf'

        # Read CSV data
        df = pd.read_csv(csv_file_path)

        # Verify required columns are present
        if 'Date' not in df.columns or 'Your Earnings' not in df.columns:
            logging.error("Required columns ('Date' and 'Your Earnings') are missing from the data.")
            print("Required columns ('Date' and 'Your Earnings') are missing from the data.")
            return

        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Your Earnings'] = df['Your Earnings'].replace(r'[\$,]', '', regex=True)
        df['Your Earnings'] = pd.to_numeric(df['Your Earnings'], errors='coerce')
        df.dropna(subset=['Date', 'Your Earnings'], inplace=True)
        df.set_index('Date', inplace=True)

        # Exclude data beyond the end of the previous month
        today = datetime.today()
        first_day_of_current_month = datetime(today.year, today.month, 1)
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        df = df[df.index <= last_day_of_previous_month]

        # Resample data every 10 days to get ~3 data points per month
        trend_data = df.resample('10D')['Your Earnings'].sum()

        # Calculate overall trend
        x_values = np.arange(len(trend_data.index))
        y_values = trend_data.values
        if len(x_values) > 1:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            trend_direction = 'upward' if slope > 0 else 'downward'
        else:
            trend_direction = 'no significant'

        # Find the month with the highest total earnings
        monthly_totals = df.resample('ME')['Your Earnings'].sum()
        top_month = monthly_totals.idxmax()
        top_month_earnings = monthly_totals.max()
        top_month_str = top_month.strftime('%B %Y')
        top_month_earnings_str = f"${top_month_earnings:,.2f}"

        # Find the date in trend_data closest to the middle of the highest earning month
        top_month_mid_date = top_month + pd.Timedelta(days=14)
        time_deltas = abs(trend_data.index - top_month_mid_date)
        closest_date = trend_data.index[time_deltas.argmin()]
        closest_value = trend_data[closest_date]

        # Plotting with dark mode settings
        fig, ax = plt.subplots(figsize=(11, 8.5))

        # Adjust margins to increase spacing from edges
        fig.subplots_adjust(left=0.12, right=0.93, top=0.88, bottom=0.32)

        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        # Plot the earnings trendline with markers
        ax.plot(trend_data.index, trend_data.values, marker='o', color='#39FF14', linestyle='-', linewidth=2)

        # Mark the highest earning month with a white marker
        ax.plot(closest_date, closest_value, marker='o', color='white', markersize=10)

        # Add a data label showing the total earnings for the highest earning month
        ax.annotate(
            top_month_earnings_str,
            xy=(closest_date, closest_value),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            color='white',
            fontsize=12,
            arrowprops=dict(facecolor='white', arrowstyle='->')
        )

        # Set titles and labels
        ax.set_title("Earnings Trendline (Every 10 Days)", fontsize=16, color='#39FF14', pad=20)
        ax.set_xlabel("Date", fontsize=14, color='#39FF14', labelpad=20)
        ax.set_ylabel("Earnings ($)", fontsize=14, color='#39FF14', labelpad=20)
        ax.grid(True, linestyle='--', color='gray', alpha=0.5)
        ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Format x-axis ticks
        ax.tick_params(axis='x', colors='white', labelsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(color='white')

        # Reduce number of x-axis labels if too crowded
        if len(trend_data.index) > 20:
            every_nth = max(1, len(trend_data.index) // 20)
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)

        # Add descriptive text at the bottom
        description_text = (
            f"The earnings trend line exemplifies an overall {trend_direction} trend, "
            f"with {top_month_str} being the highest in earnings at {top_month_earnings_str} "
            f"for the monthly earnings total."
        )
        fig.text(
            0.5, 0.13, description_text, ha='center', va='center', fontsize=12, color='white',
            wrap=True
        )

        # Save figure as PDF
        fig.savefig(pdf_output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Finish logging and notify
        logging.info(f"Script Complete! View chart at {pdf_output_path}")
        print("Earnings trendline chart has been saved successfully.")
        notify('EARNINGS TRENDLINE CHART GENERATED!', f'Chart saved at {pdf_output_path}')

    except Exception as e:
        logging.error(f"Error saving Earnings and Date Trendline PDF: {e}")
        print(f"Error saving Earnings and Date Trendline PDF: {e}")
        notify('Error', f'An error occurred: {e}')

if __name__ == '__main__':
    save_sales_trendline_pdf()
