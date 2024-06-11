from PIL import Image
import torch
import torchvision.transforms as transforms

THRESHOLD = 0.4
ALL_LABELS = ['Abrasion, scrape, or scab', 'Abscess', 'Acne',
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

# Load image and apply transformation to match ResNet50 input parameters
def load_image(image_path):
    transform = transforms.Compose([
                transforms.Resize( (224, 224) , interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
    img = transform(Image.open(image_path))
    return img

# Load the model and make inference
def perform_inference(img):
    model = torch.load("model.bin")
    model.eval()
    model_output = torch.sigmoid(model(img))

    binary_output = (model_output > THRESHOLD).astype(int)

    # Convert binary to labels
    labels = []
    labels = list(filter(lambda x: binary_output[ALL_LABELS.index(x)] == 1, ALL_LABELS))
    return labels




if __name__=="__main__":
    image_path = "/path/to/image"
    img = load_image(image_path)
    labels = perform_inference(img)
    print(labels)





