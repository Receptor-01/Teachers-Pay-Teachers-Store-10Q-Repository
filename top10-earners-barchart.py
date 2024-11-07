import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill

# Usage
csv_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/product-stats.csv'
output_pdf_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/top_10_products_sales.pdf'
log_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/top10-earners-barchart.log'

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def save_sales_by_product_chart(csv_path, pdf):
    """Generate and save a bar chart for Top 10 Products by Sales."""
    try:
        # Read data from CSV
        df = pd.read_csv(csv_path)
        
        # Ensure 'NAME' and 'EARNINGS' are present
        if 'NAME' not in df.columns or 'EARNINGS' not in df.columns:
            logging.error("Required columns ('NAME' and 'EARNINGS') are missing from the data.")
            print("Required columns ('NAME' and 'EARNINGS') are missing from the data.")
            return None

        # Exclude products with specific keywords
        exclude_keywords = ['paris', 'olympics', 'summer games']
        mask = df['NAME'].str.lower().apply(lambda x: not any(keyword in x for keyword in exclude_keywords))
        filtered_df = df[mask]

        # Ensure 'EARNINGS' is numeric
        filtered_df['EARNINGS'] = filtered_df['EARNINGS'].replace(r'[\$,]', '', regex=True)
        filtered_df['EARNINGS'] = pd.to_numeric(filtered_df['EARNINGS'], errors='coerce')
        filtered_df.dropna(subset=['EARNINGS'], inplace=True)

        # Group by 'NAME' and sum the 'EARNINGS'
        sales_by_product = filtered_df.groupby('NAME')['EARNINGS'].sum().nlargest(10).reset_index()

        if not sales_by_product.empty:
            # Plotting code
            fig, ax = plt.subplots(figsize=(11, 8.5))  # Landscape orientation
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            ax.barh(
                sales_by_product['NAME'],
                sales_by_product['EARNINGS'],
                color='#39FF14'
            )

            # Invert y-axis so the highest values are at the top
            ax.invert_yaxis()

            # Set titles and labels with white text
            ax.set_title('Top 10 Products by Earnings', fontsize=16, color='white')
            ax.set_xlabel('Total Earnings ($)', fontsize=14, color='#39FF14')
            ax.set_ylabel('Product Name', fontsize=14, color='#39FF14')

            # Set tick parameters for white text
            ax.tick_params(colors='white')

            # Adjust x-axis ticks color
            plt.xticks(color='white')

            # Add grid lines for readability
            ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            # Wrap product names
            ax.set_yticklabels([\
                '\n'.join([name[i:i+40] for i in range(0, len(name), 40)]) for name in sales_by_product['NAME']
            ])

            plt.tight_layout()

            # Save the plot as a PDF page in landscape orientation
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            logging.info("Sales by Product chart saved as PDF page.")
            print("Sales by Product chart added to PDF.")
        else:
            logging.warning("No data available for 'Top 10 Products by Sales' chart.")
            print("No data available for 'Top 10 Products by Sales' chart.")
    except Exception as e:
        logging.error(f"Error generating Sales by Product chart: {e}")
        print(f"Error generating Sales by Product chart: {e}")

def save_units_sold_by_product_chart(csv_path, pdf):
    """Generate and save a bar chart for Top 10 Products by Units Sold."""
    try:
        # Read data from CSV
        df = pd.read_csv(csv_path)
        
        # Ensure 'NAME' and 'SOLD' are present
        if 'NAME' not in df.columns or 'SOLD' not in df.columns:
            logging.error("Required columns ('NAME' and 'SOLD') are missing from the data.")
            print("Required columns ('NAME' and 'SOLD') are missing from the data.")
            return None

        # Exclude products with specific keywords
        exclude_keywords = ['paris', 'olympics', 'summer games']
        mask = df['NAME'].str.lower().apply(lambda x: not any(keyword in x for keyword in exclude_keywords))
        filtered_df = df[mask]

        # Ensure 'SOLD' is numeric
        filtered_df['SOLD'] = pd.to_numeric(filtered_df['SOLD'], errors='coerce')
        filtered_df.dropna(subset=['SOLD'], inplace=True)

        # Group by 'NAME' and sum the 'SOLD'
        units_sold_by_product = filtered_df.groupby('NAME')['SOLD'].sum().nlargest(10).reset_index()

        if not units_sold_by_product.empty:
            # Plotting code
            fig, ax = plt.subplots(figsize=(11, 8.5))  # Landscape orientation
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            ax.barh(
                units_sold_by_product['NAME'],
                units_sold_by_product['SOLD'],
                color='#39FF14'
            )

            # Invert y-axis so the highest values are at the top
            ax.invert_yaxis()

            # Set titles and labels with white text
            ax.set_title('Top 10 Products by Units Sold', fontsize=16, color='white')
            ax.set_xlabel('Units Sold', fontsize=14, color='#39FF14')
            ax.set_ylabel('Product Name', fontsize=14, color='#39FF14')

            # Set tick parameters for white text
            ax.tick_params(colors='white')

            # Adjust x-axis ticks color
            plt.xticks(color='white')

            # Add grid lines for readability
            ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            # Wrap product names
            ax.set_yticklabels([\
                '\n'.join([name[i:i+40] for i in range(0, len(name), 40)]) for name in units_sold_by_product['NAME']
            ])

            plt.tight_layout()

            # Save the plot as a PDF page in landscape orientation
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            logging.info("Units Sold by Product chart saved as PDF page.")
            print("Units Sold by Product chart added to PDF.")
        else:
            logging.warning("No data available for 'Top 10 Products by Units Sold' chart.")
            print("No data available for 'Top 10 Products by Units Sold' chart.")
    except Exception as e:
        logging.error(f"Error generating Units Sold by Product chart: {e}")
        print(f"Error generating Units Sold by Product chart: {e}")

def save_conversion_rate_by_product_chart(csv_path, pdf):
    """Generate and save a bar chart for Top 10 Products by Conversion Rate."""
    try:
        # Read data from CSV
        df = pd.read_csv(csv_path)
        
        # Ensure 'NAME' and 'CONVERSION' are present
        if 'NAME' not in df.columns or 'CONVERSION' not in df.columns:
            logging.error("Required columns ('NAME' and 'CONVERSION') are missing from the data.")
            print("Required columns ('NAME' and 'CONVERSION') are missing from the data.")
            return None

        # Exclude products with specific keywords
        exclude_keywords = ['paris', 'olympics', 'summer games']
        mask = df['NAME'].str.lower().apply(lambda x: not any(keyword in x for keyword in exclude_keywords))
        filtered_df = df[mask]

        # Clean 'CONVERSION' column: remove '%' and convert to numeric, fill blanks with zero
        filtered_df['CONVERSION'] = (
            filtered_df['CONVERSION']
            .str.replace('%', '', regex=True)
            .replace('-', '0')
            .replace('', '0')
            .fillna('0')
        )
        filtered_df['CONVERSION'] = pd.to_numeric(filtered_df['CONVERSION'], errors='coerce')

        # Group by 'NAME' and get the top 10 products by 'CONVERSION'
        conversion_rate_by_product = filtered_df.groupby('NAME')['CONVERSION'].max().nlargest(10).reset_index()

        if not conversion_rate_by_product.empty:
            # Plotting code
            fig, ax = plt.subplots(figsize=(11, 8.5))  # Landscape orientation
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            ax.barh(
                conversion_rate_by_product['NAME'],
                conversion_rate_by_product['CONVERSION'],
                color='#39FF14'
            )

            # Invert y-axis so the highest values are at the top
            ax.invert_yaxis()

            # Set titles and labels with white text
            ax.set_title('Top 10 Products by Conversion Rate', fontsize=16, color='white')
            ax.set_xlabel('Conversion Rate (%)', fontsize=14, color='#39FF14')
            ax.set_ylabel('Product Name', fontsize=14, color='#39FF14')

            # Set tick parameters for white text
            ax.tick_params(colors='white')

            # Adjust x-axis ticks color
            plt.xticks(color='white')

            # Add grid lines for readability
            ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            # Wrap product names
            ax.set_yticklabels([
                '\n'.join([name[i:i+40] for i in range(0, len(name), 40)]) for name in conversion_rate_by_product['NAME']
            ])

            plt.tight_layout()

            # Save the plot as a PDF page in landscape orientation
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            logging.info("Conversion Rate by Product chart saved as PDF page.")
            print("Conversion Rate by Product chart added to PDF.")
        else:
            logging.warning("No data available for 'Top 10 Products by Conversion Rate' chart.")
            print("No data available for 'Top 10 Products by Conversion Rate' chart.")
    except Exception as e:
        logging.error(f"Error generating Conversion Rate by Product chart: {e}")
        print(f"Error generating Conversion Rate by Product chart: {e}")

def generate_product_type_pie_chart(df, pdf):
    """Generate and add a pie chart of sold product types to the PDF in landscape mode with color-coded legend, title, and sales breakdown."""
    try:
        logging.info("Generating Product Type Pie Chart")

        # Categorize products based on sales data
        clipart_keywords = ['clipart', 'clip-art', 'clip art']
        coloring_keywords = ['coloring pages', 'coloring', 'coloring page', 'coloring pack']

        # Initialize counts for sold products
        clipart_sales_count = 0
        color_pages_sales_count = 0
        other_sales_count = 0

        # Check if the required columns exist
        if 'NAME' not in df.columns or 'SOLD' not in df.columns:
            logging.error("The required columns ('NAME' and 'SOLD') are missing from the data.")
            print("The required columns ('NAME' and 'SOLD') are missing from the data.")
            return None

        # Ensure 'SOLD' is numeric
        df['SOLD'] = pd.to_numeric(df['SOLD'], errors='coerce').fillna(0)

        # Filter products that have sold more than zero units
        sold_products_df = df[df['SOLD'] > 0]

        # Iterate through sold product titles and categorize
        for product_title in sold_products_df['NAME']:
            product_title = str(product_title).lower()  # Convert to lowercase for case-insensitive matching

            if any(keyword in product_title for keyword in clipart_keywords):
                clipart_sales_count += 1
            elif any(keyword in product_title for keyword in coloring_keywords):
                color_pages_sales_count += 1
            else:
                other_sales_count += 1

        # Total number of products sold
        total_products_sold = clipart_sales_count + color_pages_sales_count + other_sales_count

        # Data for the pie chart
        sizes = [clipart_sales_count, color_pages_sales_count, other_sales_count]
        labels = ['CLIPART', 'COLOR PAGES', 'OTHER']
        colors = ['#39FF14', 'darkgray', 'white']

        # Set up figure
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Function to display both product counts and percentages
        def make_autopct(sizes):
            def my_autopct(pct):
                total = sum(sizes)
                count = int(round(pct * total / 100.0))
                return f'({pct:.1f}%)'
            return my_autopct

        # Plot the pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, startangle=120, radius=0.8,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            autopct=make_autopct(sizes), pctdistance=0.75
        )

        # Set colors for text labels and percentages
        for text in texts:
            text.set_color('white')
            text.set_fontsize(14)
        
        for autotext in autotexts:
            autotext.set_color('black')  # Keep the percentage text black for better visibility on lighter colors
            autotext.set_fontsize(14)

        # Title
        fig.suptitle('Sales by Product Type', fontsize=20, color='white', weight='bold', y=0.85)

        # Adding legend
        ax.legend(
            wedges, labels, loc="center left", bbox_to_anchor=(0.85, 0.5),
            fontsize=12, labelspacing=1.2, frameon=False, facecolor='black', edgecolor='black'
        )

        # Ensure legend text color is white
        for text in ax.get_legend().get_texts():
            text.set_color("white")

        # Informational sentence
        info_text = "The distribution of sales between Clipart, Coloring Pages, and Other categories."
        wrapped_text = fill(info_text, width=60)

        # Display the wrapped text
        fig.text(
            0.5, 0.08, wrapped_text, ha='center', va='center', fontsize=12, color='white', style='italic'
        )

        # Create the sales breakdown text box with the total number of products sold and each category breakdown
        sales_breakdown_text = (
            f"Total Products Sold:\n"
            f"  - Clipart: {clipart_sales_count}\n"
            f"  - Coloring Pages: {color_pages_sales_count}\n"
            f"  - Other: {other_sales_count}"
        )

        # Display the sales breakdown text to the left of the chart
        fig.text(
            0.09, 0.76, sales_breakdown_text, ha='left', va='center', fontsize=8,
            color='#39FF14', style='italic', transform=ax.transAxes,
            bbox=dict(facecolor='black', alpha=0.5)
        )

        
    

        # Save to PDF in landscape mode
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', orientation='landscape')
        plt.close(fig)
        logging.info("Product Type Pie Chart added to PDF in landscape mode with title, legend, and sales breakdown.")
        print("Product Type Pie Chart added to PDF.")

    except Exception as e:
        logging.error(f"Error generating Product Type Pie Chart: {e}")
        print(f"Error generating Product Type Pie Chart: {e}")

def generate_product_distribution_pie_chart(df, pdf):
    """Generate and add a pie chart of product distribution across all categories with total number of products displayed."""
    try:
        logging.info("Generating Product Distribution Pie Chart")

        # Categorize products into Clipart, Coloring Pages, and Other
        clipart_keywords = ['clipart', 'clip-art', 'clip art']
        coloring_keywords = ['coloring pages', 'coloring', 'coloring page', 'coloring pack']

        # Initialize counts
        clipart_count = 0
        color_pages_count = 0
        other_count = 0

        # Check if 'NAME' column exists
        if 'NAME' not in df.columns:
            logging.error("The 'NAME' column is missing from the data.")
            print("The 'NAME' column is missing from the data.")
            return None

        # Iterate through product titles and categorize them into Clipart, Coloring Pages, or Other
        for product_title in df['NAME']:
            product_title = str(product_title).lower()  # Convert to lowercase for case-insensitive matching

            if any(keyword in product_title for keyword in clipart_keywords):
                clipart_count += 1
            elif any(keyword in product_title for keyword in coloring_keywords):
                color_pages_count += 1
            else:
                other_count += 1

        # Data for the pie chart
        sizes = [clipart_count, color_pages_count, other_count]
        labels = [
            f'CLIPART ({clipart_count} Products)',
            f'COLOR PAGES ({color_pages_count} Products)',
            f'OTHER ({other_count} Products)'
        ]
        colors = ['#39FF14', 'darkgray', 'white']

        # Set up figure for plotting
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, startangle=120, radius=0.8,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            autopct=lambda p: f'{p * sum(sizes) / 100:.0f}', pctdistance=0.75
        )

        # Set colors for text labels and percentages
        for text in texts:
            text.set_color('white')
            text.set_fontsize(14)
        
        for autotext in autotexts:
            autotext.set_color('black')  # Keep the percentage text black for better visibility on lighter colors
            autotext.set_fontsize(14)

        # Title
        fig.suptitle('Product Distribution Across Categories', fontsize=20, color='white', weight='bold', y=0.85)

        # Adding legend
        ax.legend(
            wedges, labels, loc="center left", bbox_to_anchor=(0.85, 0.5),
            fontsize=12, labelspacing=1.2, frameon=False, facecolor='black', edgecolor='black'
        )

        # Ensure legend text color is white
        for text in ax.get_legend().get_texts():
            text.set_color("white")

        # Informational sentence
        info_text = (
            f"The distribution of all products in your store between Clipart, Coloring Pages, and Other."
        )
        wrapped_text = fill(info_text, width=60)

        # Display the wrapped text
        fig.text(
            0.5, 0.08, wrapped_text, ha='center', va='center', fontsize=12, color='white', style='italic'
        )

        # Save to PDF in landscape mode
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', orientation='landscape')
        plt.close(fig)
        logging.info("Product Distribution Pie Chart added to PDF in landscape mode with title and legend.")
        print("Product Distribution Pie Chart added to PDF.")
        
    except Exception as e:
        logging.error(f"Error generating Product Distribution Pie Chart: {e}")
        print(f"Error generating Product Distribution Pie Chart: {e}")

# Generate PDF with all charts including the new Product Distribution Pie Chart
with PdfPages(output_pdf_path) as pdf:
    # Generate Sales by Product Chart
    save_sales_by_product_chart(csv_path, pdf)
    
    # Generate Units Sold by Product Chart
    save_units_sold_by_product_chart(csv_path, pdf)
    
    # Generate Conversion Rate by Product Chart
    save_conversion_rate_by_product_chart(csv_path, pdf)
    
    # Generate Product Type Pie Chart
    df = pd.read_csv(csv_path)  # Load the dataframe once
    generate_product_type_pie_chart(df, pdf)  # Use df in the pie chart function

    # Generate Product Distribution Pie Chart (for all products in the store)
    generate_product_distribution_pie_chart(df, pdf)
