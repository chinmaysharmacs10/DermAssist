def get_doc_urls():
    with open('doc_urls.txt', 'r') as file:
        urls = file.readlines()

    doc_urls = [str(url).strip() for url in urls]
    return doc_urls
