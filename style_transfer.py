import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# STYLE OPTIONS
# -------------------------
styles = {
    "1": "starry.jpg",
    "2": "watercolor.jpg",
    "3": "oil.jpg",
    "4": "abstract.jpg",
    "5": "swirl.jpg",
    "6": "colorpencil.jpg"
}

print("\nChoose Style:")
print("1 - Starry Night")
print("2 - Watercolor Painting")
print("3 - Oil Painting")
print("4 - Abstract Art")
print("5 - Psychedelic Swirl Art")
print("6 - Color Pencil Art")

choice = input("Enter choice (1-6): ")

style_path = styles.get(choice, "starry.jpg")

content_path = "content.jpg"

# -------------------------
# IMAGE LOADER
# -------------------------
image_size = 512

loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content = load_image(content_path)
style = load_image(style_path)

# -------------------------
# LOAD VGG19
# -------------------------
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# freeze parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# -------------------------
# FEATURE EXTRACTION
# -------------------------
def get_features(image, model):

    layers = {
        '0': 'conv1',
        '5': 'conv2',
        '10': 'conv3',
        '19': 'conv4'
    }

    features = {}

    x = image

    for name, layer in model._modules.items():

        x = layer(x)

        if name in layers:
            features[layers[name]] = x

    return features

# -------------------------
# GRAM MATRIX
# -------------------------
def gram_matrix(tensor):

    b, c, h, w = tensor.size()

    tensor = tensor.view(c, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram

# -------------------------
# GET FEATURES
# -------------------------
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# -------------------------
# OUTPUT IMAGE (clone content)
# -------------------------
output = content.clone().to(device)
output.requires_grad_(True)

# -------------------------
# OPTIMIZER
# -------------------------
optimizer = optim.Adam([output], lr=0.003)

# -------------------------
# TRAINING
# -------------------------
steps = 800

content_weight = 15
style_weight = 100

print("\nApplying style...\n")

for step in range(steps):

    optimizer.zero_grad()

    output_features = get_features(output, vgg)

    # content loss
    content_loss = torch.mean(
        (output_features['conv4'] - content_features['conv4']) ** 2
    )

    # style loss
    style_loss = 0

    for layer in style_grams:

        output_gram = gram_matrix(output_features[layer])
        style_gram = style_grams[layer]

        style_loss += 0.2 * torch.mean((output_gram - style_gram) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    total_loss.backward()

    optimizer.step()

    # clamp to valid pixel range
    output.data.clamp_(0, 1)

    if step % 50 == 0:
        print("Step:", step)

# -------------------------
# SAVE OUTPUT
# -------------------------
# convert tensors to PIL
output_image = unloader(output.squeeze().cpu())
content_image = Image.open(content_path).convert("RGB").resize(output_image.size)

# convert both to YCbCr
output_ycbcr = output_image.convert("YCbCr")
content_ycbcr = content_image.convert("YCbCr")

# keep luminance from output, colors from content
y, _, _ = output_ycbcr.split()
_, cb, cr = content_ycbcr.split()

final_image = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

final_image.save("output.jpg")

print("\nDONE. output.jpg saved.")