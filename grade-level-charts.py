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
log_file_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/grade-level-charts.log'
grade_level_chart_image_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/grade-level-chart.png'
hexbin_chart_image_path_template = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/{}-hexbin-chart.png'
pdf_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/grade-level-charts.pdf'

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

def generate_sales_by_grade_level_chart(df):
    """Generate and save a bar chart of the number of products sold by grade level with descriptive text."""
    try:
        logging.info("Generating Sales by Grade Level Chart")

        # Ensure 'Grade Levels' column exists
        if 'Grade Levels' not in df.columns:
            logging.error("Required column ('Grade Levels') is missing from the data.")
            print("Required column ('Grade Levels') is missing from the data.")
            return None

        # Filter out products with exclusion keywords
        df = filter_exclusion_keywords(df, 'Product Title')

        # Define an ordered list of grades
        grades_order = [
            'Pre-K', 'K', '1st', '2nd', '3rd', '4th', '5th', '6th',
            '7th', '8th', '9th', '10th', '11th', '12th', 'Adult Education', 'Staff'
        ]

        # Function to standardize and expand grade levels
        def expand_grade_levels(grade_levels_str):
            grade_levels = []
            if pd.isna(grade_levels_str):
                return grade_levels
            for part in grade_levels_str.split(','):
                part = part.strip()
                # Replace 'Grade ' prefix if present
                part = part.replace('Grade ', '').replace('grade ', '').strip()
                # Handle ranges
                if '-' in part:
                    start_grade_str, end_grade_str = part.split('-')
                    start_grade = standardize_grade(start_grade_str.strip())
                    end_grade = standardize_grade(end_grade_str.strip())
                    if start_grade in grades_order and end_grade in grades_order:
                        start_index = grades_order.index(start_grade)
                        end_index = grades_order.index(end_grade)
                        grade_levels.extend(grades_order[min(start_index, end_index):max(start_index, end_index) + 1])
                else:
                    # Single grade
                    grade = standardize_grade(part)
                    if grade in grades_order:
                        grade_levels.append(grade)
            return grade_levels
        
        def standardize_grade(grade_str):
            grade_str = grade_str.strip()
            # Handle Pre-K
            if grade_str.lower() in ['pre-k', 'prek', 'pre k']:
                return 'Pre-K'
            # Handle Kindergarten
            elif grade_str.lower() in ['k', 'kindergarten']:
                return 'K'
            elif grade_str.lower() == 'staff':
                return 'Staff'
            elif grade_str.lower() == 'adult education':
                return 'Adult Education'
            else:
                # Try to match numeric grades
                grade_num_str = re.sub(r'(th|st|nd|rd)', '', grade_str.lower()).strip()
                if grade_num_str.isdigit():
                    grade_num = int(grade_num_str)
                    if 1 <= grade_num <= 12:
                        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(grade_num if grade_num < 20 else grade_num % 10, 'th')
                        return f"{grade_num}{suffix}"
            return None

        # Apply the function to expand grade levels
        df['grade_levels_expanded'] = df['Grade Levels'].apply(expand_grade_levels)

        # Explode the expanded grade levels
        exploded_df = df.explode('grade_levels_expanded').copy()

        # Remove any entries where 'grade_levels_expanded' is NaN or empty
        exploded_df = exploded_df.dropna(subset=['grade_levels_expanded']).copy()
        exploded_df = exploded_df[exploded_df['grade_levels_expanded'] != '']

        # Group by 'grade_levels_expanded' and count the products
        sales_by_grade = exploded_df['grade_levels_expanded'].value_counts().reindex(grades_order, fill_value=0).reset_index()
        sales_by_grade.columns = ['grade_levels_expanded', 'sales_count']

        # Set up the plot with dark mode settings
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the bar chart
        ax.barh(
            sales_by_grade['grade_levels_expanded'],
            sales_by_grade['sales_count'],
            color='#39FF14'
        )

        # Invert y-axis so the highest values are at the top
        ax.invert_yaxis()

        # Set titles and labels with white text
        ax.set_title('Number of Products Sold by Grade Level', fontsize=16, color='white')
        ax.set_xlabel('Number of Products Sold', fontsize=14, color='#39FF14')
        ax.set_ylabel('Grade Level', fontsize=14, color='#39FF14')

        # Set tick parameters for white text
        ax.tick_params(colors='white')

        # Adjust x-axis ticks color
        plt.xticks(color='white')

        # Add grid lines for readability
        ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Set x-axis grid lines to every 50 units
        ax.xaxis.set_major_locator(plt.MultipleLocator(50))

        plt.tight_layout()

        # Save the plot as an image
        # Add descriptive text at the bottom
        total_sales_count = df['Product Title'].nunique()  # Count unique products sold, without splitting by grade level
        description_text = (
            f"This bar chart illustrates the number of products sold across different grade levels. "
            f"The total number of unique products sold is {total_sales_count}."
        )
        plt.figtext(0.1, -0.05, description_text, wrap=True, horizontalalignment='left', fontsize=10, color='white')
        
        fig.savefig(grade_level_chart_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info("Sales by Grade Level chart saved as PNG image.")
        return grade_level_chart_image_path
    except Exception as e:
        logging.error(f"Error generating Sales by Grade Level chart: {e}")
        print(f"Error generating Sales by Grade Level chart: {e}")
        return None

def generate_hexbin_chart_for_each_grade(df_stats, df_grades, grade):
    """Generate and save a hexbin chart for a specific grade showing earnings vs units sold with descriptive text."""
    try:
        logging.info(f"Generating Hexbin Chart for {grade} Products")

        # Ensure required columns exist
        if 'SOLD' not in df_stats.columns or 'EARNINGS' not in df_stats.columns or 'NAME' not in df_stats.columns:
            logging.error("Required columns ('SOLD', 'EARNINGS', 'NAME') are missing from the stats data.")
            print("Required columns ('SOLD', 'EARNINGS', 'NAME') are missing from the stats data.")
            return None

        # Ensure 'Grade Levels' and 'Product Title' columns exist in grades data
        if 'Grade Levels' not in df_grades.columns or 'Product Title' not in df_grades.columns:
            logging.error("Required columns ('Grade Levels', 'Product Title') are missing from the grades data.")
            print("Required columns ('Grade Levels', 'Product Title') are missing from the grades data.")
            return None

        # Rename 'Product Title' to 'NAME' in grades dataframe to match for merging
        df_grades = df_grades.rename(columns={'Product Title': 'NAME'})

        # Filter out products with exclusion keywords
        df_stats = filter_exclusion_keywords(df_stats, 'NAME')
        df_grades = filter_exclusion_keywords(df_grades, 'NAME')

        # Merge the dataframes on 'NAME'
        merged_df = pd.merge(df_stats, df_grades[['NAME', 'Grade Levels']], on='NAME', how='left')

        # Filter for specific grade products
        grade_df = merged_df[merged_df['Grade Levels'].str.contains(grade, case=False, na=False)]

        # Convert 'SOLD' and 'EARNINGS' to numeric, coercing errors to NaN
        grade_df.loc[:, 'SOLD'] = pd.to_numeric(grade_df['SOLD'], errors='coerce')
        grade_df.loc[:, 'EARNINGS'] = pd.to_numeric(grade_df['EARNINGS'].replace('[\$]', '', regex=True), errors='coerce')

        # Drop rows with missing or invalid data
        grade_df = grade_df.dropna(subset=['SOLD', 'EARNINGS']).copy()

        # Set up the hexbin plot with dark mode settings
        fig, ax = plt.subplots(figsize=(8, 6))  # Reduced size to fit summary
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the hexbin chart using Units Sold as X and Earnings as Y
        hb = ax.hexbin(grade_df['SOLD'], grade_df['EARNINGS'], gridsize=30, cmap='Greens', mincnt=1)
        cb = fig.colorbar(hb, ax=ax, label='Count')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

        # Set titles and labels with white text
        ax.set_title(f'Earnings vs Units Sold for {grade} Products', fontsize=14, color='white')
        ax.set_xlabel('Units Sold', fontsize=12, color='#39FF14')
        ax.set_ylabel('Earnings ($)', fontsize=12, color='#39FF14')

        # Set tick parameters for white text
        ax.tick_params(colors='white')

        # Adjust ticks color
        plt.yticks(color='white')
        plt.xticks(color='white')

        # Add grid lines for readability
        ax.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add summary text below the chart
        total_products_in_category = merged_df[merged_df['Grade Levels'].str.contains(grade, case=False, na=False)].shape[0]
        total_units_sold = grade_df['SOLD'].sum()
        total_earnings = grade_df['EARNINGS'].sum()
        if not grade_df.empty:
            hot_seller = grade_df.loc[grade_df['SOLD'].idxmax()]
            high_roller = grade_df.loc[grade_df['EARNINGS'].idxmax()]
            hot_seller_text = f"Hot Seller: {hot_seller['NAME']} ({hot_seller['SOLD']} units sold)"
            high_roller_text = f"High Roller: {high_roller['NAME']} (${high_roller['EARNINGS']:.2f} in total earnings)"
        else:
            hot_seller_text = "Hot Seller: N/A"
            high_roller_text = "High Roller: N/A"

        # Add summary text below the chart
        summary_text = (
            f"Summary for {grade} Products:\n"
            f"Total Products in {grade} Category: {total_products_in_category}\n"
            f"Total Units Sold: {total_units_sold}\n"
            f"Total Category Earnings: ${total_earnings:.2f}\n"
            f"{hot_seller_text}\n"
            f"{high_roller_text}"
        )
        plt.figtext(0.1, -0.15, summary_text, wrap=True, horizontalalignment='left', fontsize=10, color='white')

        plt.tight_layout()

        # Save the hexbin chart as an image
        hexbin_chart_image_path = hexbin_chart_image_path_template.format(grade.replace(' ', '_'))
        fig.savefig(hexbin_chart_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Hexbin Chart for {grade} products saved as PNG image.")
        return hexbin_chart_image_path
    except Exception as e:
        logging.error(f"Error generating Hexbin Chart for {grade} products: {e}")
        print(f"Error generating Hexbin Chart for {grade} products: {e}")
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
        # Load the product data for grade levels
        grade_level_df = pd.read_csv(grade_level_data_path)
        
        # Load the product stats data
        product_stats_df = pd.read_csv(product_stats_data_path)
        
        # Generate and save the sales by grade level chart
        grade_level_chart_path = generate_sales_by_grade_level_chart(grade_level_df)
        
        # Generate and save hexbin charts for each grade level
        grades = [
            'Pre-K', 'K', '1st', '2nd', '3rd', '4th', '5th', '6th',
            '7th', '8th', '9th', '10th', '11th', '12th', 'Adult Education', 'Staff'
        ]
        hexbin_chart_paths = []
        for grade in grades:
            hexbin_chart_path = generate_hexbin_chart_for_each_grade(product_stats_df, grade_level_df, grade)
            if hexbin_chart_path:
                hexbin_chart_paths.append(hexbin_chart_path)
        
        # Generate PDF with the grade level chart and all hexbin charts
        image_paths = [path for path in [grade_level_chart_path] + hexbin_chart_paths if path is not None]
        if image_paths:
            generate_pdf_with_images(image_paths)
    except FileNotFoundError:
        logging.error(f"The data file at {grade_level_data_path} or {product_stats_data_path} was not found.")
        print(f"The data file at {grade_level_data_path} or {product_stats_data_path} was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
