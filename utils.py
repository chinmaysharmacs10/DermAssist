def get_doc_urls():
    with open('doc_urls.txt', 'r') as file:
        urls = file.readlines()

    doc_urls = [str(url).strip() for url in urls]
    return doc_urls


def get_skin_disease_labels():
    all_labels = ['Abrasion, scrape, or scab', 'Abscess', 'Acne',
                  'Acute and chronic dermatitis', 'Acute dermatitis, NOS',
                  'Allergic Contact Dermatitis', 'CD - Contact dermatitis',
                  'Cellulitis', 'Chronic dermatitis, NOS', 'Cutaneous lupus',
                  'Cutaneous sarcoidosis', 'Drug Rash', 'Eczema',
                  'Erythema multiforme', 'Folliculitis', 'Granuloma annulare',
                  'Herpes Simplex', 'Herpes Zoster', 'Hypersensitivity', 'Impetigo',
                  'Inflicted skin lesions', 'Insect Bite', 'Intertrigo',
                  'Irritant Contact Dermatitis', 'Keratosis pilaris',
                  'Leukocytoclastic Vasculitis', 'Lichen Simplex Chronicus',
                  'Lichen nitidus', 'Lichen planus/lichenoid eruption', 'Miliaria',
                  'Molluscum Contagiosum', 'O/E - ecchymoses present',
                  'Perioral Dermatitis', 'Photodermatitis',
                  'Pigmented purpuric eruption', 'Pityriasis lichenoides',
                  'Pityriasis rosea', 'Post-Inflammatory hyperpigmentation',
                  'Prurigo nodularis', 'Psoriasis', 'Purpura', 'Rosacea',
                  'SCC/SCCIS', 'Scabies', 'Scar Condition', 'Seborrheic Dermatitis',
                  'Skin and soft tissue atypical mycobacterial infection',
                  'Stasis Dermatitis', 'Syphilis', 'Tinea', 'Tinea Versicolor',
                  'Urticaria', 'Verruca vulgaris', 'Viral Exanthem', 'Xerosis']
    return all_labels
