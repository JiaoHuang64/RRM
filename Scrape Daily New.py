import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import ssl
import arxiv
import re
import shutil
import tarfile
from collections import Counter
import random
from datetime import datetime, timedelta
import hashlib
import fitz  # PyMuPDF
from PIL import Image
import cv2

# Scrape_Daily_New.py
print("Starting Scrape_Daily_New.py")
os.chdir('/Users/jiaohuangbixia/Downloads/1/M1')
# Ignore SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

PROCESSED_IDS_FILE = 'processed_arxiv_ids.csv'
PROCESSED_IDS_REPORT = 'processed_arxiv_ids_report.html'


def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_processed_ids():
    """Load processed arxiv IDs from a CSV file."""
    if os.path.exists(PROCESSED_IDS_FILE):
        df = pd.read_csv(PROCESSED_IDS_FILE)
        return set(df['arxiv_id'].tolist())
    return set()


def save_processed_ids(processed_ids):
    """Save processed arxiv IDs to a CSV file."""
    df = pd.DataFrame(list(processed_ids), columns=['arxiv_id'])
    df.to_csv(PROCESSED_IDS_FILE, index=False)
    generate_processed_ids_report(df)


def generate_processed_ids_report(df):
    """Generate an HTML report of processed arxiv IDs."""
    html = df.to_html(index=False)
    with open(PROCESSED_IDS_REPORT, 'w') as f:
        f.write(html)


def fetch_arxiv_new_articles(category):
    """Fetch new articles from arxiv for a given category."""
    url = f'https://arxiv.org/list/{category}/new'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from {url}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    for dt in soup.find_all('dt'):
        a = dt.find('a', title='Abstract')
        if a:
            arxiv_id = a.get_text().strip().replace('\n', '').replace('arXiv:', '').replace(' ', '')
            articles.append(arxiv_id)
    return articles


def fetch_article_details(arxiv_id):
    """Fetch the details of an arxiv article by ID."""
    url = f'https://arxiv.org/abs/{arxiv_id}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch details for {arxiv_id}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1', class_='title mathjax').get_text(strip=True).replace('Title:', '').strip()
    abstract = soup.find('blockquote', class_='abstract mathjax').get_text(strip=True).replace('Abstract:', '').strip()
    authors = [a.get_text(strip=True) for a in soup.find_all('a', class_='author')]
    return {
        'title': title,
        'abstract': abstract,
        'authors': ', '.join(authors),
        'arxiv_id': arxiv_id,
        'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
        'html_url': f'https://arxiv.org/abs/{arxiv_id}'
    }


def fetch_author_info(arxiv_id):
    """Fetch the authors of an arxiv article by ID."""
    search = arxiv.Search(id_list=[arxiv_id])
    client = arxiv.Client()
    result = next(client.results(search), None)
    if result:
        return [author.name for author in result.authors]
    return []


def download_tex_from_arxiv(arxiv_id, download_dir='downloads'):
    """Download the LaTeX source file of an arxiv article by ID."""
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(source_url)
    if response.status_code == 200:
        content_type = response.headers.get('content-type')
        print(f"Downloaded content type for {arxiv_id}: {content_type}")
        os.makedirs(download_dir, exist_ok=True)
        tar_path = os.path.join(download_dir, f"{arxiv_id}.tar.gz")
        with open(tar_path, 'wb') as f:
            f.write(response.content)
        return tar_path
    else:
        print(f"Failed to download {arxiv_id}")
        return None


def extract_image_paths_from_figure_env(tex_file, extract_dir):
    """Extract image paths from LaTeX figure environments in the source file."""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    image_paths = []
    encodings = ['utf-8', 'latin1', 'ascii']

    try:
        with tarfile.open(tex_file) as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError:
        print("The downloaded file is not a valid tar.gz archive.")
        return image_paths

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".tex"):
                tex_file_path = os.path.join(root, file)
                for encoding in encodings:
                    try:
                        with open(tex_file_path, 'r', encoding=encoding) as f:
                            tex_content = f.read()
                            figures = re.findall(r'\\begin{figure}.*?\\end{figure}', tex_content, re.DOTALL)
                            for figure in figures:
                                image_paths += re.findall(r'\\includegraphics\[.*?\]{(.*?)}', figure)
                        break
                    except UnicodeDecodeError:
                        print(f"Error decoding {tex_file_path} with {encoding} encoding. Trying next encoding.")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {tex_file_path}: {e}")
                        break

    return image_paths


def download_random_top_image(image_paths, article_id, extract_dir, download_dir):
    """Download a random top image from the list of image paths."""
    article_download_dir = os.path.join(download_dir, article_id)
    if not os.path.exists(article_download_dir):
        os.makedirs(article_download_dir)

    if not image_paths:
        print(f"No images found in the article {article_id}.")
        # No images found, try extracting from PDF
        pdf_path = download_pdf(article_id, extract_dir)
        if pdf_path:
            extracted_images = extract_images_from_pdf(pdf_path, extract_dir)
            if extracted_images:
                selected_image = random.choice(extracted_images)
                shutil.copy(selected_image, article_download_dir)
                print(f"Saved {os.path.basename(selected_image)} from PDF extraction.")
        return

    # Count the frequency of each image
    image_counter = Counter(image_paths)
    # Get the top 5 most referenced images that are PDF or PNG
    top_images = [img for img, count in image_counter.most_common() if img.endswith('.pdf') or img.endswith('.png')][:5]

    if not top_images:
        print(f"No PDF or PNG images found in the top references for article {article_id}.")
        # No PDF or PNG images found, try extracting from PDF
        pdf_path = download_pdf(article_id, extract_dir)
        if pdf_path:
            extracted_images = extract_images_from_pdf(pdf_path, extract_dir)
            if extracted_images:
                selected_image = random.choice(extracted_images)
                shutil.copy(selected_image, article_download_dir)
                print(f"Saved {os.path.basename(selected_image)} from PDF extraction.")
        return

    # Randomly select one image from the top 5
    selected_image = random.choice(top_images)

    # Check if the selected image file exists
    full_img_path = os.path.abspath(os.path.join(extract_dir, selected_image))
    if os.path.exists(full_img_path):
        img_name = os.path.basename(selected_image)
        dest_img_path = os.path.join(article_download_dir, img_name)
        with open(full_img_path, 'rb') as src, open(dest_img_path, 'wb') as dest:
            dest.write(src.read())
        print(f"Saved {img_name} of {len(image_paths)}.")
        # Clean up: remove all other files in the article's directory
        for item in os.listdir(article_download_dir):
            item_path = os.path.join(article_download_dir, item)
            if item_path != dest_img_path:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    else:
        print(f"Selected image not found: {full_img_path}")


def download_pdf(arxiv_id, output_dir):
    """Download the PDF file of an arxiv article by ID."""
    url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download PDF for {arxiv_id}")
        return None

    pdf_path = os.path.join(output_dir, f'{arxiv_id}.pdf')
    with open(pdf_path, 'wb') as file:
        file.write(response.content)
    return pdf_path


def image_hash(image_path):
    """Generate a hash for an image file."""
    with open(image_path, 'rb') as f:
        img_hash = hashlib.md5(f.read()).hexdigest()
    return img_hash


def contains_text_or_graph(img_path):
    """Check if an image contains text or graphs."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    return True


def extract_images_from_pdf(pdf_path, output_dir, min_image_area=10000):
    """Extract images from a PDF file."""
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    unique_hashes = set()

    for page_num in range(len(pdf_document)):
        try:
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_path = os.path.join(output_dir,
                                        f'{os.path.basename(pdf_path).split(".")[0]}_page{page_num + 1}_img{img_index + 1}.{image_ext}')

                with open(img_path, 'wb') as img_file:
                    img_file.write(image_bytes)

                # Check for duplicate images
                img_hash = image_hash(img_path)
                if img_hash in unique_hashes:
                    os.remove(img_path)
                    continue
                unique_hashes.add(img_hash)

                # Check if the image has information (filter by area)
                with Image.open(img_path) as img:
                    if img.size[0] * img.size[1] < min_image_area:
                        os.remove(img_path)
                        continue

                # Check if the image contains text or graphs
                if not contains_text_or_graph(img_path):
                    os.remove(img_path)
                    continue

                image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing page {page_num} of {pdf_path}: {e}")
            continue

    return image_paths


if __name__ == '__main__':
    # 定义需要处理的Arxiv分类
    categories = ['astro-ph', 'gr-qc', 'hep-ex', 'hep-ph', 'hep-th']
    start = datetime.now()  # 记录开始时间

    all_articles = []  # 存储所有获取到的文章
    filtered_articles = []  # 存储筛选后的文章

    processed_ids = load_processed_ids()  # 加载已处理过的Arxiv ID

    # 定义关键词和评分
    keywords = {
        'dark matter': 100,
        'axion': 90,
        'axion-like particle': 90,
        'dark photon': 90,
        'sterile neutrino': 90,
        'primordial black hole': 90,
        'Millicharged Particle': 80,
        'Cosmic neutrino': 80,
        'Cosmic String': 70,
        'Domain Wall': 70,
        'N-body': 70
    }

    # 过滤文章并添加相关性评分
    def calculate_relevance(details):
        relevance = 0
        for keyword, score in keywords.items():
            if keyword.lower() in details['title'].lower() or keyword.lower() in details['abstract'].lower():
                relevance = max(relevance, score)
        return relevance

    for category in categories:  # 遍历每个分类
        arxiv_ids = fetch_arxiv_new_articles(category)  # 获取该分类的新文章ID
        # 过滤掉已处理过的文章ID
        new_arxiv_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in processed_ids]
        for arxiv_id in new_arxiv_ids:  # 处理每个新文章ID
            print(f"Processing arxiv_id: {arxiv_id}")
            details = fetch_article_details(arxiv_id)  # 获取文章详细信息
            if not details:  # 如果没有详细信息，则跳过
                continue

            authors = fetch_author_info(arxiv_id)  # 获取文章作者信息

            relevance = calculate_relevance(details)  # 计算文章的相关性评分

            # 创建文章字典，包含各种信息
            article = {
                'relevance': relevance,  # 添加相关性评分
                'category': category,
                'title': details['title'],
                'authors': ', '.join(authors),
                'summary': details['abstract'],
                'arxiv_id': details['arxiv_id'],
                'pdf_url': details['pdf_url'],
                'html_url': details['html_url'],
                'source_url': details['pdf_url'],
                'image_paths': '[]'
            }
            all_articles.append(article)  # 将文章添加到所有文章列表中

            # 如果相关性评分大于0，则添加到筛选后的文章列表中
            if relevance > 0:
                filtered_articles.append(article)
        print(f'{len(new_arxiv_ids)} new titles saved from Arxiv {category}.')

    end = datetime.now()  # 记录结束时间
    print(f'Time elapsed: {timedelta(seconds=round((end - start).total_seconds()))}')

    # Fetch images and update image paths
    if filtered_articles:
        for article in filtered_articles:
            arxiv_id = article['arxiv_id']
            print(f"Fetching images for article: {arxiv_id}")
            article_output_dir = os.path.join('./viewable_images', arxiv_id)
            create_directory(article_output_dir)

            # Download tar file to a temporary directory
            temp_dir = os.path.join(article_output_dir, 'temp')
            create_directory(temp_dir)
            tex_tar_path = download_tex_from_arxiv(arxiv_id, temp_dir)

            if tex_tar_path:
                tex_images = extract_image_paths_from_figure_env(tex_tar_path, temp_dir)
                if not tex_images:  # If no images found in .tex files, try extracting from PDF
                    pdf_path = download_pdf(arxiv_id, temp_dir)
                    if pdf_path:
                        extracted_images = extract_images_from_pdf(pdf_path, article_output_dir)
                        if extracted_images:
                            article['image_paths'] = extracted_images[0]  # Save only one image
                else:
                    download_random_top_image(tex_images, arxiv_id, temp_dir, article_output_dir)
                    article['image_paths'] = os.listdir(article_output_dir)  # Update image_paths

            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

        # Save filtered articles to a new CSV file
        df = pd.DataFrame(filtered_articles)
        df.to_csv('filtered_arxiv_dark_matter_articles_with_images.csv', index=False, encoding='utf-8')
        print(f"Filtered and saved {len(filtered_articles)} articles to 'filtered_arxiv_dark_matter_articles_with_images.csv'.")
    else:
        print("No articles with relevant keywords found.")

    # Save all articles to a CSV file
    df_all = pd.DataFrame(all_articles)
    df_all.to_csv('arxiv_dark_matter_articles_with_images.csv', index=False, encoding='utf-8')
    print(f"Saved {len(all_articles)} articles to 'arxiv_dark_matter_articles_with_images.csv'.")

    # Update the list of processed article IDs
    processed_ids.update(new_arxiv_ids)
    save_processed_ids(processed_ids)
print("Finished Scrape_Daily_New.py")