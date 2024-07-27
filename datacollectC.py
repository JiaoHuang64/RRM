import csv
import requests


# Define the function to get the title and abstract from arXiv API
def get_arxiv_info(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        xml = response.text
        title_start = xml.find('<title>') + len('<title>')
        title_end = xml.find('</title>', title_start)
        title = xml[title_start:title_end].strip()

        abstract_start = xml.find('<summary>') + len('<summary>')
        abstract_end = xml.find('</summary>', abstract_start)
        abstract = xml[abstract_start:abstract_end].strip()

        return title, abstract
    else:
        return None, None


# Read the input CSV file
input_file = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/final_updated_arxiv_relevance.csv'
output_file = '/traindataC.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header for the output file
    writer.writerow(['arxiv_id', 'title', 'summary', 'relevance'])

    # Skip the header of the input file
    next(reader, None)

    for row in reader:
        arxiv_id = row[0]
        relevance = row[1]  # Assuming relevance is in the second column

        print(f"Processing arxiv_id: {arxiv_id}")  # Debug information

        title, abstract = get_arxiv_info(arxiv_id)

        if title and abstract:
            writer.writerow([arxiv_id, title, abstract, relevance])
        else:
            writer.writerow([arxiv_id, 'Title not found', 'Abstract not found', relevance])
            print(f"Failed to retrieve information for arxiv_id: {arxiv_id}")  # Debug information