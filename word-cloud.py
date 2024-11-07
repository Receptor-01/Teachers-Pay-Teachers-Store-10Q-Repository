
#!/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/venv/bin/python3



import pandas as pd
import logging
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Define paths
data_file_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/product-stats.csv'
log_file_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/word-cloud.log'
output_image_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/word-cloud.png'
virtual_env_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/venv'

# Set up logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_word_cloud_text(df):
    """Generate a single string from product titles for the word cloud."""
    try:
        # Check if 'product_title' exists
        if 'NAME' not in df.columns:
            logging.error("The 'NAME' column is missing from the product data.")
            return ''

        # Extract product titles
        product_titles = df['NAME'].astype(str)
        text = ' '.join(product_titles)

        # Convert text to lowercase
        text = text.lower()

        # Remove punctuation and non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        custom_stopwords = {'worksheet', 'worksheets', 'lesson', 'lessons', 'activities', 'activity', 'math', 'reading', 'english', 'grade', 'grades'}
        stop_words.update(custom_stopwords)

        # Filter stopwords from the text
        text_tokens = text.split()
        text_tokens_filtered = [word for word in text_tokens if word not in stop_words]

        # Rejoin the filtered words
        filtered_text = ' '.join(text_tokens_filtered)

        return filtered_text
    except Exception as e:
        logging.error(f"Error generating text for word cloud: {e}")
        print(f"Error generating text for word cloud: {e}")
        return ''

def save_product_name_word_cloud(df):
    """Generate and save a word cloud image from product titles."""
    try:
        text = generate_word_cloud_text(df)
        
        if not text:
            logging.warning("No text available for word cloud; skipping word cloud generation.")
            return None
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='black',
            colormap='Greens',
            stopwords=None,
            max_words=200,
            max_font_size=100,
            random_state=42
        ).generate(text)
        
        # Set up the plot
        plt.figure(figsize=(11, 8.5), facecolor='k')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save the word cloud image
        plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight', facecolor='k')
        plt.close()
        logging.info("Product Name Word Cloud saved as PNG image.")
        return output_image_path
    except Exception as e:
        logging.error(f"Error generating Word Cloud: {e}")
        print(f"Error generating Word Cloud: {e}")
        return None

def main():
    try:
        # Load the product data
        df = pd.read_csv(data_file_path)
        
        # Generate and save the word cloud
        save_product_name_word_cloud(df)
    except FileNotFoundError:
        logging.error(f"The data file at {data_file_path} was not found.")
        print(f"The data file at {data_file_path} was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
