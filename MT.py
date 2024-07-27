import os
import re
import time
import random
import pandas as pd
import tweepy
from PIL import Image
import signal
import imagehash
from pdf2image import convert_from_path
import logging
import json

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 设置Twitter API密钥和访问令牌
api_key = 'F1SIZA6VzxrJJlhGet0hTKMss'
api_secret_key = 'FfUxAGCPcrFH2xw47yd9Y09UqoPgXOm1Xtiu0yaiH2bFKThZGH'
access_token = '1816090652402409472-jWMOgebWyFZDHZ2NytLhLECUCmF51c'
access_token_secret = 'tIes1tCzk5fQ8WPjvm4GLHYXl7XRoAYUOApM1fggTDm7j'

# 认证
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# 设置工作目录
os.chdir('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN')
print("Starting Twitter_Post.py")

def compress_image(image_path, output_path, quality=70):
    """Compress an image to reduce file size."""
    try:
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # Convert to RGB mode
        image.save(output_path, "JPEG", quality=quality)
        return output_path
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

def clean_text(text):
    """Clean up the text by removing extra spaces and special characters."""
    if isinstance(text, float):
        text = ""
    text = re.sub(r'\s+', ' ', text)
    text.replace('$', '').replace('_', '')
    return text.strip()

def timeout_handler(signum, frame):
    raise TimeoutError

def post_update_with_images(message, image_paths):
    """Post an update with images to Twitter."""
    media_ids = []
    upload_delay = 10

    for image_path in image_paths:  # Only post one image
        if os.path.exists(image_path):
            compressed_image_path = compress_image(image_path, os.path.splitext(image_path)[0] + '_compressed.jpg')
            if compressed_image_path:
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(120)
                    print(f"Uploading image: {compressed_image_path}")
                    media = api.media_upload(compressed_image_path)
                    media_ids.append(media.media_id)
                    signal.alarm(0)
                    print(f"Uploaded image: {compressed_image_path}")
                except TimeoutError:
                    print(f"Uploading image timed out: {compressed_image_path}")
                    signal.alarm(0)
                except AttributeError as e:
                    print(f"Error uploading media (AttributeError): {e}")
                    signal.alarm(0)
                except Exception as e:
                    print(f"Error uploading media: {e}")
                    signal.alarm(0)

                time.sleep(upload_delay)
            else:
                print(f"Image compression failed: {image_path}")
        else:
            print(f"Image not found: {image_path}")

        break  # Ensure only one image is posted

    try:
        if len(message) > 280:
            message = message[:277] + '...'

        if media_ids:
            print(f"Posting tweet with message: {message} and images")
            response = api.update_status(status=message, media_ids=media_ids)
            print(f"Posted tweet with message: {message} and images: {image_paths}")
            print(f"Response: {response}")
        else:
            print(f"No images to post for message: {message}. Skipping post.")
    except tweepy.Forbidden as e:
        print(f"Error posting to Twitter (Forbidden): {e}")
    except tweepy.TweepyException as e:
        print(f"Error posting to Twitter: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def convert_pdf_to_images(pdf_path, output_dir):
    """Convert a PDF file to images."""
    poppler_path = '/opt/homebrew/bin'  # Set your poppler path
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    image_paths = []
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f'page_{i+1}.jpg')
        image.save(output_path, 'JPEG')
        image_paths.append(output_path)
    return image_paths

def main():
    viewable_images_dir = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/viewable_images'

    # 已注释掉的已发布文章检查相关代码
    # published_titles = load_published_titles()
    # print(f"Loaded published titles: {published_titles}")

    # 读取CSV文件
    csv_file_path = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/filtered.csv'
    df = pd.read_csv(csv_file_path, dtype={'arxiv_id': str})

    for index, row in df.iterrows():
        title = clean_text(row['title'])
        relevance = float(row['relevance'])
        if relevance < 0:
            relevance = 0
        elif relevance > 100:
            relevance = 100
        relevance = f"{relevance:.2f}"  # 格式化为两位小数

        authors = clean_text(row['authors'])
        summary = clean_text(row['summary'])
        arxiv_id = row['arxiv_id'].strip()

        print(f"Processing article: {title}")

        # 已注释掉的已发布文章检查相关代码
        # if title in published_titles:
        #     print(f"Article titled '{title}' has already been published, skipping.")
        #     continue

        # Find images in the specified directory for the article
        image_dir = os.path.join(viewable_images_dir, arxiv_id)
        image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
                elif file.endswith('.pdf'):
                    pdf_images = convert_pdf_to_images(os.path.join(root, file), root)
                    image_paths.extend(pdf_images)

        if not image_paths:
            print(f"No images found for article titled '{title}', skipping post.")
            continue

        # If more than one image, randomly select one
        if len(image_paths) > 1:
            image_paths = [random.choice(image_paths)]

        valid_image_paths = []
        seen_hashes = set()
        for img_path in image_paths:
            if os.path.exists(img_path):
                img_hash = imagehash.average_hash(Image.open(img_path))
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    valid_image_paths.append(img_path)
                else:
                    print(f"Duplicate image detected: {img_path}")
            else:
                print(f"Image path does not exist: {img_path}")

        message = f"Relevance: {relevance}%\nTitle: {title}\nAuthors: {authors}\nSummary: {summary}"

        print(f"Posting article: {title}")
        print(f"Message: {message}")
        print(f"Image paths: {valid_image_paths}")
        post_update_with_images(message, valid_image_paths)
        print(f"Finished posting article: {title}")

        # 已注释掉的已发布文章检查相关代码
        # save_published_title(title)
        # published_titles.add(title)
        # print(f"Saved published title: {title}")

        time.sleep(10)

if __name__ == "__main__":
    main()
print("Finished Twitter_Post.py")