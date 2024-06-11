def get_doc_urls():
    with open('doc_urls.txt', 'r') as file:
        urls = file.readlines()

    doc_urls = [str(url).strip() for url in urls]
    return doc_urls


def get_skin_disease_labels():
    with open('skin_disease_labels.txt', 'r') as file:
        lines = file.readlines()

    skin_disease_labels = []
    for line in lines:
        labels = [label.strip() for label in line.strip().split(",")]
        skin_disease_labels.extend(labels)

    return skin_disease_labels
