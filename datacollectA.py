import csv
import requests
from xml.etree import ElementTree


# Define the function to get articles from arXiv API
def get_arxiv_articles(category, max_results=30):
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results={max_results}"
    response = requests.get(url)
    articles = []

    if response.status_code == 200:
        print(f"Response for {category} received successfully")
        xml = ElementTree.fromstring(response.content)
        for entry in xml.findall('{http://www.w3.org/2005/Atom}entry'):
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            articles.append((arxiv_id, title, summary))
    else:
        print(f"Failed to retrieve articles for category {category}: HTTP {response.status_code}")

    return articles


# Get articles for each category
economics_articles = get_arxiv_articles('econ.GN', max_results=30)  # Economics General
statistics_articles = get_arxiv_articles('stat.ML', max_results=20)  # Statistics Machine Learning
mathematics_articles = get_arxiv_articles('math.PR', max_results=20)  # Mathematics Probability

# Check if articles were retrieved
print(f"Economics articles retrieved: {len(economics_articles)}")
print(f"Statistics articles retrieved: {len(statistics_articles)}")
print(f"Mathematics articles retrieved: {len(mathematics_articles)}")

# Define relevance scores
economics_relevance = 0
statistics_relevance = 10
mathematics_relevance = 5

# Prepare data for CSV
data = []

for article in economics_articles:
    data.append((*article, economics_relevance))

for article in statistics_articles:
    data.append((*article, statistics_relevance))

for article in mathematics_articles:
    data.append((*article, mathematics_relevance))

# Write data to CSV
output_file = 'traindataA.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['arxiv_id', 'title', 'summary', 'relevance'])
    writer.writerows(data)

print(f"CSV file '{output_file}' has been created successfully.")