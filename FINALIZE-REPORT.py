#!/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/venv/bin/python3

import os
import time
from PyPDF2 import PdfMerger
import logging

# Define paths
pdf_output_folder = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/TPT-REPORT.pdf'
combined_pdf_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/FINALIZE-REPORT.pdf'
log_file_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/combine_pdfs.log'

# Set up logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def wait_for_sales_map(output_folder, timeout=90, interval=10):
    sales_map_path = os.path.join(output_folder, 'sales_map.pdf')
    elapsed_time = 0
    while elapsed_time < timeout:
        if os.path.exists(sales_map_path):
            return True
        logging.info(f"Waiting for 'sales_map.pdf' to appear. Checked at {elapsed_time} seconds.")
        time.sleep(interval)
        elapsed_time += interval
    logging.error("Required file 'sales_map.pdf' did not appear within the given timeout. Aborting PDF combination.")
    print("Error: Required file 'sales_map.pdf' did not appear within the given timeout. Aborting PDF combination.")
    return False

def combine_pdfs(output_folder, output_file):
    try:
        # Wait for 'sales_map.pdf' to exist
        if not wait_for_sales_map(output_folder):
            return

        # Initialize the PdfMerger object
        merger = PdfMerger()

        # Iterate through all files in the output folder
        for item in sorted(os.listdir(output_folder)):
            item_path = os.path.join(output_folder, item)
            # Check if the file is a PDF
            if os.path.isfile(item_path) and item.endswith('.pdf'):
                logging.info(f"Adding {item} to the merger.")
                merger.append(item_path)

        # Write all merged PDFs into a single output file
        with open(output_file, 'wb') as output_pdf:
            merger.write(output_pdf)
            logging.info(f"Combined PDF saved as {output_file}")

    except Exception as e:
        logging.error(f"Error combining PDFs: {e}")
        print(f"Error combining PDFs: {e}")

    finally:
        # Close the PdfMerger object
        merger.close()

def main():
    combine_pdfs(pdf_output_folder, combined_pdf_output_path)

if __name__ == "__main__":
    main()
