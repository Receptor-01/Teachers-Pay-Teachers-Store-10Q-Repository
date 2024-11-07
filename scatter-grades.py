#!/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/venv/bin/python3

import pandas as pd
import logging
import re
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Define paths
grade_level_data_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/product-data.csv'
product_stats_data_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/product-stats.csv'
sales_report_data_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/sales-report.csv'
log_file_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/grade-level-scatter-charts.log'
scatter_chart_image_path_template = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/{}-scatter-chart.png'
scatter_all_transactions_image_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/all-transactions-scatter-chart.png'
pdf_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/grade-level-scatter-charts.pdf'

# Set up logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Keywords to filter out from the data
EXCLUSION_KEYWORDS = ['paris', 'olympics', 'olympic', 'Paris','Olympics','Olympic','Summer Games']

def filter_exclusion_keywords(df, column_name):
    """Filter out rows that contain any of the exclusion keywords in the specified column."""
    pattern = '|'.join(EXCLUSION_KEYWORDS)
    return df[~df[column_name].str.contains(pattern, case=False, na=False)]

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

def generate_scatter_chart_for_each_grade(sales_df, grade):
    """Generate and save a scatter chart for individual transactions for a specific grade."""
    try:
        logging.info(f"Generating Scatter Chart for {grade} Transactions")

        # Ensure required columns exist
        if 'Date' not in sales_df.columns or 'Product' not in sales_df.columns or 'Transaction Total' not in sales_df.columns or 'Expanded Grade Levels' not in sales_df.columns:
            logging.error("Required columns ('Date', 'Product', 'Transaction Total', 'Expanded Grade Levels') are missing from the sales data.")
            print("Required columns ('Date', 'Product', 'Transaction Total', 'Expanded Grade Levels') are missing from the sales data.")
            return None

        # Filter for specific grade products
        filtered_sales_df = sales_df[sales_df['Expanded Grade Levels'].apply(lambda x: isinstance(x, list) and grade in x)]

        if filtered_sales_df.empty:
            logging.info(f"No transactions found for {grade}. Skipping scatter chart generation.")
            return None

        # Convert 'Transaction Total' to numeric, coercing errors to NaN
        filtered_sales_df['Transaction Total'] = pd.to_numeric(filtered_sales_df['Transaction Total'].replace('[\$]', '', regex=True), errors='coerce')

        # Drop rows with missing or invalid data
        filtered_sales_df = filtered_sales_df.dropna(subset=['Transaction Total']).copy()

        # Set up the scatter plot with dark mode settings
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the scatter chart using Date as X and Transaction Total as Y
        ax.scatter(filtered_sales_df['Date'], filtered_sales_df['Transaction Total'], color='#39FF14', alpha=0.7)

        # Set titles and labels with white text
        ax.set_title(f'Individual Transactions for {grade} Products', fontsize=14, color='white')
        ax.set_xlabel('Date', fontsize=12, color='#39FF14')
        ax.set_ylabel('Transaction Amount ($)', fontsize=12, color='#39FF14')

        # Set tick parameters for white text
        ax.tick_params(colors='white')

        # Adjust ticks color
        plt.yticks(color='white')
        plt.xticks(color='white', rotation=45)

        # Add grid lines for readability
        ax.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add descriptive text at the bottom
        total_transactions = filtered_sales_df.shape[0]
        total_amount = filtered_sales_df['Transaction Total'].sum()
        description_text = (
            f"This scatter chart illustrates individual transactions for {grade} products. "
            f"The total number of transactions is {total_transactions}, with a cumulative transaction amount of ${total_amount:.2f}."
        )
        plt.figtext(0.1, -0.1, description_text, wrap=True, horizontalalignment='left', fontsize=10, color='white')

        plt.tight_layout()

        # Save the scatter chart as an image
        scatter_chart_image_path = scatter_chart_image_path_template.format(grade.replace(' ', '_'))
        fig.savefig(scatter_chart_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Scatter Chart for {grade} transactions saved as PNG image.")
        return scatter_chart_image_path
    except Exception as e:
        logging.error(f"Error generating Scatter Chart for {grade} transactions: {e}")
        print(f"Error generating Scatter Chart for {grade} transactions: {e}")
        return None

def generate_all_transactions_scatter_chart(sales_df):
    """Generate and save a scatter chart for all transactions."""
    try:
        logging.info("Generating Scatter Chart for All Transactions")

        # Ensure required columns exist
        if 'Date' not in sales_df.columns or 'Transaction Total' not in sales_df.columns:
            logging.error("Required columns ('Date', 'Transaction Total') are missing from the sales data.")
            print("Required columns ('Date', 'Transaction Total') are missing from the sales data.")
            return None

        # Convert 'Transaction Total' to numeric, coercing errors to NaN
        sales_df['Transaction Total'] = pd.to_numeric(sales_df['Transaction Total'].replace('[\$]', '', regex=True), errors='coerce')

        # Drop rows with missing or invalid data
        sales_df = sales_df.dropna(subset=['Transaction Total']).copy()

        # Convert 'Date' to datetime
        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        sales_df = sales_df.dropna(subset=['Date']).copy()

        # Set up the scatter plot with dark mode settings
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the scatter chart using Date as X and Transaction Total as Y
        ax.scatter(sales_df['Date'], sales_df['Transaction Total'], color='#39FF14', alpha=0.7)

        # Set titles and labels with white text
        ax.set_title('All Transactions Over Time', fontsize=14, color='white')
        ax.set_xlabel('Date', fontsize=12, color='#39FF14')
        ax.set_ylabel('Transaction Amount ($)', fontsize=12, color='#39FF14')

        # Set tick parameters for white text
        ax.tick_params(colors='white')

        # Adjust ticks color
        plt.yticks(color='white')
        plt.xticks(color='white', rotation=45)

        # Add grid lines for readability
        ax.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add descriptive text at the bottom
        total_transactions = sales_df.shape[0]
        total_amount = sales_df['Transaction Total'].sum()
        description_text = (
            f"This scatter chart illustrates all individual transactions over time. "
            f"The total number of transactions is {total_transactions}, with a cumulative transaction amount of ${total_amount:.2f}."
        )
        plt.figtext(0.1, -0.1, description_text, wrap=True, horizontalalignment='left', fontsize=10, color='white')

        plt.tight_layout()

        # Save the scatter chart as an image
        fig.savefig(scatter_all_transactions_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info("Scatter Chart for All Transactions saved as PNG image.")
        return scatter_all_transactions_image_path
    except Exception as e:
        logging.error(f"Error generating Scatter Chart for All Transactions: {e}")
        print(f"Error generating Scatter Chart for All Transactions: {e}")
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
        logging.info("PDF containing scatter charts saved.")
    except Exception as e:
        logging.error(f"Error generating PDF with images: {e}")
        print(f"Error generating PDF with images: {e}")

def main():
    try:
        # Load the product data for grade levels
        grade_level_df = pd.read_csv(grade_level_data_path)
        
        # Load the sales report data
        sales_report_df = pd.read_csv(sales_report_data_path)
        
        # Expand the grade levels
        grade_level_df['Expanded Grade Levels'] = grade_level_df['Grade Levels'].apply(expand_grade_levels)
        
        # Merge sales data with grade level data
        merged_df = pd.merge(sales_report_df, grade_level_df, left_on='Product', right_on='Product Title', how='left')
        
        # Filter out products with exclusion keywords
        merged_df = filter_exclusion_keywords(merged_df, 'Product Title')

        # Define grades to generate charts for
        grades = [
            'Pre-K', 'K', '1st', '2nd', '3rd', '4th', '5th', '6th',
            '7th', '8th', '9th', '10th', '11th', '12th', 'Adult Education', 'Staff', 'Higher Education', 'Homeschool', 'Not Grade Specific'
        ]
        
        # Generate scatter charts for each grade level
        scatter_chart_paths = []
        for grade in grades:
            scatter_chart_path = generate_scatter_chart_for_each_grade(merged_df, grade)
            if scatter_chart_path:
                scatter_chart_paths.append(scatter_chart_path)
        
        # Generate scatter chart for all transactions
        all_transactions_chart_path = generate_all_transactions_scatter_chart(sales_report_df)
        if all_transactions_chart_path:
            scatter_chart_paths.append(all_transactions_chart_path)
        
        # Generate PDF with the scatter charts
        if scatter_chart_paths:
            generate_pdf_with_images(scatter_chart_paths)
    except FileNotFoundError:
        logging.error(f"The data file at {grade_level_data_path} or {sales_report_data_path} was not found.")
        print(f"The data file at {grade_level_data_path} or {sales_report_data_path} was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
