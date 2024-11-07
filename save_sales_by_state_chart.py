#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import logging
import os
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator

# Set up logging
logging.basicConfig(
    filename='/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/10-states.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def notify(title, text):
    """Send a system notification on macOS."""
    subprocess.call(['osascript', '-e', f'display notification "{text}" with title "{title}"'])

def save_sales_by_state_charts():
    """Generate and save bar charts for Top 10 States by Products Sold and Sales Earnings."""
    try:
        logging.info('☢︎ SCRIPT START ------------------------------')
        notify('GENERATING SALES BY STATE CHARTS...', 'The script has started running.')

        # Define file paths
        csv_file_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/sales-report.csv'
        chart_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/sales_by_state_charts.pdf'

        # Read CSV data
        df = pd.read_csv(csv_file_path)

        # Verify required columns are present
        required_columns = ['State', 'Order Id', 'Your Earnings']
        for col in required_columns:
            if col not in df.columns:
                logging.error(f"Required column '{col}' is missing from the data.")
                print(f"Required column '{col}' is missing from the data.")
                return

        # Data preprocessing
        df['Your Earnings'] = df['Your Earnings'].replace(r'[\$,]', '', regex=True)
        df['Your Earnings'] = pd.to_numeric(df['Your Earnings'], errors='coerce')
        df.dropna(subset=['Your Earnings'], inplace=True)

        # Custom grid intervals
        product_interval = 5  # Interval for Products Sold chart
        earnings_interval = 25  # Interval for Earnings chart

        # Set up PDF to save multiple pages
        with PdfPages(chart_output_path) as pdf:
            # Chart 1: Top 10 States by Products Sold
            sales_by_units = df.groupby('State')['Order Id'].count().nlargest(10)

            if not sales_by_units.empty:
                top_state_units = sales_by_units.idxmax()
                top_units = sales_by_units.max()
                top_units_str = f"{top_units:,}"

                fig, ax = plt.subplots(figsize=(11, 8.5))
                fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.25)

                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                sales_by_units.sort_values(ascending=True).plot(kind='barh', color='#39FF14', ax=ax)

                ax.set_title('Top 10 States by Products Sold', fontsize=16, color='white', pad=20)
                ax.set_xlabel('Number of Products Sold', fontsize=14, color='#39FF14', labelpad=15)
                ax.set_ylabel('State', fontsize=14, color='#39FF14', labelpad=15)

                ax.tick_params(axis='both', colors='white', labelsize=10)
                plt.xticks(color='white')
                plt.yticks(color='white')

                # Add gray vertical grid lines with finer intervals
                ax.xaxis.set_major_locator(MultipleLocator(product_interval))
                ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

                description_text = (
                    f"Data shows {top_state_units} as holding the highest amount of products sold "
                    f"with a total of {top_units_str} products sold."
                )
                fig.text(
                    0.5, 0.13, description_text, ha='center', va='center', fontsize=10, color='white',
                    style='italic', wrap=True
                )

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                logging.info("Chart 1: Top 10 States by Products Sold added to PDF.")

            # Chart 2: Top 10 States by Sales Earnings
            sales_by_earnings = df.groupby('State')['Your Earnings'].sum().nlargest(10)

            if not sales_by_earnings.empty:
                top_state_earnings = sales_by_earnings.idxmax()
                top_earnings = sales_by_earnings.max()
                top_earnings_str = f"${top_earnings:,.2f}"

                fig, ax = plt.subplots(figsize=(11, 8.5))
                fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.25)

                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                sales_by_earnings.sort_values(ascending=True).plot(kind='barh', color='#39FF14', ax=ax)

                ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
                ax.set_title('Top 10 States by Sales Earnings', fontsize=16, color='white', pad=20)
                ax.set_xlabel('Total Earnings ($)', fontsize=14, color='#39FF14', labelpad=15)
                ax.set_ylabel('State', fontsize=14, color='#39FF14', labelpad=15)

                ax.tick_params(axis='both', colors='white', labelsize=10)
                plt.xticks(color='white')
                plt.yticks(color='white')

                # Add gray vertical grid lines with finer intervals
                ax.xaxis.set_major_locator(MultipleLocator(earnings_interval))
                ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

                description_text = (
                    f"Data shows {top_state_earnings} as holding the highest amount of earnings "
                    f"with a total of {top_earnings_str} dollars earned."
                )
                fig.text(
                    0.5, 0.13, description_text, ha='center', va='center', fontsize=10, color='white',
                    style='italic', wrap=True
                )

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                logging.info("Chart 2: Top 10 States by Sales Earnings added to PDF.")

            d = pdf.infodict()
            d['Title'] = 'Sales by State Charts'
            d['Author'] = 'Your Name'
            d['Subject'] = 'Top 10 States by Products Sold and Sales Earnings'
            d['CreationDate'] = pd.Timestamp.now()
            d['ModDate'] = pd.Timestamp.now()

        logging.info(f"Sales by State charts saved at {chart_output_path}.")
        print("Sales by State charts have been saved successfully.")
        notify('SALES BY STATE CHARTS GENERATED!', f'Charts saved at {chart_output_path}')

    except Exception as e:
        logging.error(f"Error generating Sales by State charts: {e}")
        print(f"Error generating Sales by State charts: {e}")
        notify('Error', f'An error occurred: {e}')

if __name__ == '__main__':
    save_sales_by_state_charts()
