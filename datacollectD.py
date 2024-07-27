import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
import pandas as pd
import ssl

# 设置工作目录
os.chdir('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN')

# 忽略SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_arxiv_articles(category, start_date, end_date):
    start_date_str = start_date.strftime('%Y%m%d%H%M%S')
    end_date_str = end_date.strftime('%Y%m%d%H%M%S')
    url = f'https://export.arxiv.org/api/query?search_query=cat:{category}+AND+submittedDate:[{start_date_str}+TO+{end_date_str}]&start=0&max_results=1000&sortBy=submittedDate&sortOrder=descending'
    print(f"Fetching URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from {url}")
        return []

    print(f"Response status: {response.status_code}")
    print(f"Response content: {response.content[:500]}...")

    try:
        soup = BeautifulSoup(response.content, 'lxml')
    except Exception as e:
        print(f"lxml解析失败，使用html.parser。错误信息：{e}")
        soup = BeautifulSoup(response.content, 'html.parser')

    entries = soup.find_all('entry')
    print(f"Found {len(entries)} entries")
    articles = []
    for entry in entries:
        published = entry.published.get_text().strip()
        published_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
        if start_date <= published_date <= end_date:
            arxiv_id = entry.id.get_text().split('/')[-1]
            title = entry.title.get_text().strip()
            abstract = entry.summary.get_text().strip()
            articles.append({
                'arxiv_id': arxiv_id,
                'title': title,
                'summary': abstract,
                'published_date': published_date
            })
    return articles

if __name__ == '__main__':
    categories = ['astro-ph', 'gr-qc', 'hep-ex', 'hep-ph', 'hep-th']
    start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')  # 使用datetime对象进行比较
    end_date = datetime.strptime('2024-06-30', '%Y-%m-%d')    # 使用datetime对象进行比较
    start = datetime.now()

    dark_matter_articles = []
    non_dark_matter_articles = []

    for category in categories:
        arxiv_articles = fetch_arxiv_articles(category, start_date, end_date)
        for article in arxiv_articles:
            print(f"Processing arxiv_id: {article['arxiv_id']} published on {article['published_date']}")
            if 'dark matter' in article['title'].lower() or 'dark matter' in article['summary'].lower():
                article['relevance'] = 100
                dark_matter_articles.append(article)
            else:
                article['relevance'] = 5
                non_dark_matter_articles.append(article)
        print(f'{len(arxiv_articles)} articles processed from Arxiv {category}.')

    end = datetime.now()
    print(f'Time elapsed: {timedelta(seconds=round((end - start).total_seconds()))}')

    # 保存包含 "dark matter" 的文章到CSV文件F1.csv
    if dark_matter_articles:
        df_dark_matter = pd.DataFrame(dark_matter_articles, columns=['arxiv_id', 'title', 'summary', 'relevance'])
        df_dark_matter.to_csv('traindataB.csv', index=False, encoding='utf-8')
        print(f"Filtered and saved {len(dark_matter_articles)} articles to 'traindataB.csv'.")
    else:
        print("No articles with 'dark matter' found.")

    # 保存不包含 "dark matter" 的文章到CSV文件F2.csv
    if non_dark_matter_articles:
        df_non_dark_matter = pd.DataFrame(non_dark_matter_articles, columns=['arxiv_id', 'title', 'summary', 'relevance'])
        df_non_dark_matter.to_csv('traindataD.csv', index=False, encoding='utf-8')
        print(f"Filtered and saved {len(non_dark_matter_articles)} articles to 'traindataD.csv'.")

print("Finished Scrape_Daily_New.py")