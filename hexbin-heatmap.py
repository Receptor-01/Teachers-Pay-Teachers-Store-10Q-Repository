#!/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/venv/bin/python3

import pandas as pd
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib.colors import LinearSegmentedColormap

# Define paths
grade_level_data_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/product-data.csv'
product_stats_data_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/product-stats.csv'
sales_report_data_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/sales-report.csv'
log_file_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/hexbin-heatmap.log'
hexbin_image_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/hexbin-heatmap.png'
pdf_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/hexbin-headmap.pdf'

# Set up logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define custom colormap
colors = ['#39FF14', '#404040']  # Bright green to dark gray
custom_cmap = LinearSegmentedColormap.from_list('custom_green_to_gray', colors, N=256)

def expand_grade_levels(grade_levels_str):
    """Expand the grade levels column to handle multiple grades and ranges."""
    grades_order = [
        'Pre-K', 'K', '1st', '2nd', '3rd', '4th', '5th', '6th',
        '7th', '8th', '9th', '10th', '11th', '12th', 'Adult Education', 'Staff', 'Higher Education', 'Homeschool', 'Not Grade Specific'
    ]
    expanded_grades = []
    if pd.isna(grade_levels_str):
        return expanded_grades
    for part in str(grade_levels_str).split(','):
        part = part.strip()
        # Handle ranges
        if '-' in part:
            start_grade_str, end_grade_str = part.split('-')
            start_grade = start_grade_str.strip()
            end_grade = end_grade_str.strip()
            if start_grade in grades_order and end_grade in grades_order:
                start_index = grades_order.index(start_grade)
                end_index = grades_order.index(end_grade)
                expanded_grades.extend(grades_order[min(start_index, end_index):max(start_index, end_index) + 1])
        else:
            # Single grade
            if part in grades_order:
                expanded_grades.append(part)
    return expanded_grades

def generate_hexbin_heatmap(df):
    """Generate and save a hexbin heatmap for PAGE VIEWS and SOLD columns."""
    try:
        # Check if the required columns are in the dataframe
        if 'PAGE VIEWS' not in df.columns or 'SOLD' not in df.columns:
            logging.error("The required columns 'PAGE VIEWS' and 'SOLD' are missing from the product data.")
            return None

        # Convert columns to numeric, coercing errors to NaN
        df['PAGE VIEWS'] = pd.to_numeric(df['PAGE VIEWS'], errors='coerce')
        df['SOLD'] = pd.to_numeric(df['SOLD'], errors='coerce')

        # Drop rows with missing or invalid data
        df_filtered = df[['PAGE VIEWS', 'SOLD']].dropna()

        # Remove rows where PAGE VIEWS or SOLD are zero to focus on meaningful data
        df_filtered = df_filtered[(df_filtered['PAGE VIEWS'] > 0) & (df_filtered['SOLD'] > 0)]

        # Cap PAGE VIEWS at 200 for scaling purposes
        df_filtered['PAGE VIEWS'] = df_filtered['PAGE VIEWS'].clip(upper=200)

        # Set up the plot with black background
        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor='k')
        ax.set_facecolor('k')
        hb = ax.hexbin(df_filtered['PAGE VIEWS'], df_filtered['SOLD'], gridsize=30, cmap=custom_cmap, mincnt=1)
        cb = fig.colorbar(hb, ax=ax, label='Count')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklines(), color='white')
        cb.outline.set_edgecolor('white')
        cb.ax.yaxis.label.set_color('white')
        ax.set_xlabel('Page Views', color='#39FF14')
        ax.set_ylabel('Units Sold', color='#39FF14')
        ax.set_title('Page Views vs Units Sold', color='white')
        ax.tick_params(colors='white')
        
        # Save the hexbin heatmap image
        plt.savefig(hexbin_image_path, format='png', dpi=300, bbox_inches='tight', facecolor='k')
        plt.close()
        logging.info("Hexbin Heatmap saved as PNG image.")
        return hexbin_image_path
    except Exception as e:
        logging.error(f"Error generating Hexbin Heatmap: {e}")
        print(f"Error generating Hexbin Heatmap: {e}")
        return None

def generate_pdf_with_images(image_paths):
    """Generate a PDF containing the given images on black background pages."""
    try:
        with PdfPages(pdf_output_path) as pdf:
            for image_path in image_paths:
                # Create a figure for the PDF page
                fig = plt.figure(figsize=(11, 8.5), facecolor='k')
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.set_facecolor('k')
                
                # Read and plot the image
                img = plt.imread(image_path)
                ax.imshow(img, aspect='auto')
                
                # Save the page to the PDF
                pdf.savefig(fig, bbox_inches='tight', facecolor='k')
                plt.close(fig)
        logging.info("PDF containing charts saved.")
    except Exception as e:
        logging.error(f"Error generating PDF with images: {e}")
        print(f"Error generating PDF with images: {e}")

def main():
    try:
        # Load the grade level data
        grade_level_df = pd.read_csv(grade_level_data_path)
        
        # Expand the grade levels
        grade_level_df['Expanded Grade Levels'] = grade_level_df['Grade Levels'].apply(expand_grade_levels)
        
        # Load the sales report data
        sales_report_df = pd.read_csv(sales_report_data_path)
        
        # Load the product stats data
        product_stats_df = pd.read_csv(product_stats_data_path)

        # Merge sales data with grade level data
        merged_df = pd.merge(sales_report_df, grade_level_df, left_on='Product', right_on='Product Title', how='left')
        
        # Merge the product stats with the already merged dataframe
        merged_df = pd.merge(merged_df, product_stats_df, on='Product Title', how='left')

        # Generate and save the hexbin heatmap
        hexbin_path = generate_hexbin_heatmap(merged_df)
        
        # Generate PDF with the hexbin heatmap
        image_paths = [path for path in [hexbin_path] if path is not None]
        if image_paths:
            generate_pdf_with_images(image_paths)
    except FileNotFoundError:
        logging.error(f"The data file at {grade_level_data_path}, {sales_report_data_path}, or {product_stats_data_path} was not found.")
        print(f"The data file at {grade_level_data_path}, {sales_report_data_path}, or {product_stats_data_path} was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
