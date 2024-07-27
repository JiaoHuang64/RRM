import pandas as pd
import os

# Change to the directory where the data washing script is located
os.chdir('/Users/jiaohuangbixia/Downloads/1/M1')
print("Starting Data_Wash.py")

# Update to the local CSV file path
csv_file_path = '/Users/jiaohuangbixia/Downloads/1/M1/filtered_arxiv_dark_matter_articles_with_images.csv'

# Read the CSV file and check the data
df = pd.read_csv(csv_file_path, dtype={'arxiv_id': str})
print(f"Initial data loaded. Number of rows: {len(df)}")

# Fill missing values with "No information available"
df.fillna('No information available', inplace=True)

# Function to safely evaluate the string representation of image paths
def safe_eval_image_paths(image_paths_str):
    try:
        return eval(image_paths_str)
    except:
        return []

# Convert the image_paths column from string to list
df['image_paths'] = df['image_paths'].apply(safe_eval_image_paths)

# Check for rows with no image paths after conversion
rows_with_no_images = df[df['image_paths'].apply(len) == 0]
print(f"Number of rows with no image paths: {len(rows_with_no_images)}")

# Update image paths to absolute paths
df['image_paths'] = df['image_paths'].apply(lambda paths: [os.path.abspath(os.path.join('/Users/jiaohuangbixia/Downloads/1/M1/viewable_images', os.path.basename(path))) for path in paths])
print("Image paths updated to absolute paths.")

# Create the posts directory
posts_dir = '/Users/jiaohuangbixia/Downloads/1/M1/posts'
if not os.path.exists(posts_dir):
    os.makedirs(posts_dir)
print("Posts directory created.")

# Generate an HTML file for each article
generated_posts = 0
for index, row in df.iterrows():
    relevance = str(row['relevance']).rstrip('%')  # Ensure only one percent sign
    arxiv_id = row['arxiv_id'].strip()  # Ensure arxiv_id is correctly formatted
    post_html = f"""
    <html>
    <head>
        <title>{row['title']}</title>
    </head>
    <body>
        <p><strong>Relevance:</strong> {relevance}%</p>
        <p><strong>Title:</strong> {row['title']}</p>
        <p><strong>Authors:</strong> {row['authors']}</p>
        <p><strong>Summary:</strong> {row['summary']}</p>
        <p><strong>Category:</strong> {row['category']}</p>
        <p><strong>ArXiv ID:</strong> {arxiv_id}</p>
        <p><strong>PDF URL:</strong> <a href="{row['pdf_url']}">{row['pdf_url']}</a></p>
        <p><strong>HTML URL:</strong> <a href="{row['html_url']}">{row['html_url']}</a></p>
        <h2>Images</h2>
        {''.join([f'<img src="{img_path}" alt="Image" style="max-width:100%;">' for img_path in row['image_paths']]) if row['image_paths'] != 'No information available' else 'No images available'}
    </body>
    </html>
    """
    post_file_path = os.path.join(posts_dir, f"{arxiv_id}_{index}.html")
    with open(post_file_path, 'w', encoding='utf-8') as f:
        f.write(post_html)
    generated_posts += 1
    print(f"Post generated for {arxiv_id} with index {index}")

print(f"Total number of posts generated: {generated_posts}")

# Save the cleaned data to a new CSV file
cleaned_csv_file_path = '/Users/jiaohuangbixia/Downloads/1/M1/cleaned_arxiv_dark_matter_articles_with_images.csv'
df.to_csv(cleaned_csv_file_path, index=False)
print("Data cleaning completed and HTML posts generated.")
print("Finished Data_Wash.py")