import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import cv2
import argparse
from FeatureExtractor import FeatureExtractor
from HashNetwork import Embedder
from build_faiss_db import FaissDatabase, load_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.unsqueeze(0).requires_grad_()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        target = torch.zeros_like(output)
        target[0, class_idx] = 1
        output.backward(gradient=target)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_mnist_data(train=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    raw_dataset = datasets.MNIST('./data', train=train, download=True)
    return dataset, raw_dataset

def visualize_gradcam(model, image_tensor, true_label, index, device, layer_index=4, use_scorecam=False):
    if layer_index == 7:
        target_layer = model.extractor[7]  
    elif layer_index == 4:
        target_layer = model.extractor[4]  
    else:
        raise ValueError("Invalid layer index. Use 7 for last conv layer or 4 for second conv layer.")

    grad_cam = GradCAM(model, target_layer)
    
    if use_scorecam:
        heatmap = compute_scorecam(model, image_tensor, target_layer, device)
    else:
        heatmap = grad_cam(image_tensor)

    image = image_tensor.squeeze().cpu().numpy()
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_rgb = np.repeat(image_norm[..., np.newaxis], 3, axis=-1)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    alpha = 0.7
    overlay = alpha * heatmap_color + (1 - alpha) * image_rgb
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    im = axes[1].imshow(heatmap, cmap='viridis')
    axes[1].set_title(f"{'Score-CAM' if use_scorecam else 'GradCAM'} Heatmap (Layer {layer_index})")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()
    plt.savefig(f"gradcam_index_{index}_layer_{layer_index}{'_scorecam' if use_scorecam else ''}.png")
    plt.show()

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        print(f"True Label: {true_label}")
    print(f"GradCAM visualizations saved to gradcam_index_{index}_layer_{layer_index}{'_scorecam' if use_scorecam else ''}.png")
    
    return grad_cam

def compute_scorecam(model, image_tensor, target_layer, device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).requires_grad_(False)
    
    activations = None
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    hook = target_layer.register_forward_hook(forward_hook)
    
    output = model(image_tensor)
    hook.remove()
    
    activations = activations.squeeze(0).cpu()  
    C, H, W = activations.shape
    
    activation_maps = []
    for i in range(C):
        act_map = activations[i]
        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
        activation_maps.append(act_map)
    
    scores = []
    with torch.no_grad():
        for i in range(C):
            mask = activation_maps[i].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            masked_input = image_tensor * F.interpolate(mask, size=(28, 28), mode='bilinear', align_corners=False)
            output = model(masked_input)
            score = output.norm(dim=1).item()
            scores.append(score)
    
    scores = torch.tensor(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap = torch.zeros(H, W)
    for i in range(C):
        heatmap += scores[i] * activation_maps[i]
    
    heatmap = heatmap.numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (28, 28), interpolation=cv2.INTER_LINEAR)
    return heatmap

def query_and_plot_similar_images(index, db, extractor, embedder, device, k=5, layer_index=4, use_scorecam=False):

    test_dataset, raw_test = get_mnist_data(train=False)
    train_dataset, raw_train = get_mnist_data(train=True)

    query_image = raw_test.data[index]
    query_label = raw_test.targets[index].item()

    query_tensor = test_dataset[index][0].to(device)

    with torch.no_grad():
        image_tensor = query_image.unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        features = extractor(image_tensor)
        hash_vector = embedder.embedder(features)
        hash_vector_np = hash_vector.cpu().numpy().astype('float32')
    distances, similar_indices = db.search(hash_vector_np, k)
    # similar_indices = np.where(train_dataset.targets[index] == test_dataset.targets[index])[:k]
    similar_images = [raw_train.data[idx] for idx in similar_indices]
    similar_labels = [raw_train.targets[idx].item() for idx in similar_indices]
    similar_tensors = [train_dataset[idx][0].to(device) for idx in similar_indices]

    print("\n=== Similar Images ===")
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))
    axes[0].imshow(query_image, cmap='gray')
    axes[0].set_title(f"Query: {query_label}")
    axes[0].axis('off')

    for i, (img, label) in enumerate(zip(similar_images, similar_labels)):
        axes[i + 1].imshow(img, cmap='gray')
        axes[i + 1].set_title(f"#{i + 1}: {label}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"similar_images_index_{index}.png")
    plt.show()
    print(f"Found {k} similar images to query image (Label: {query_label})")
    print(f"Similar images saved to similar_images_index_{index}.png")

    print("\n=== GradCAM and Feature Visualizations ===")
    grad_cam = visualize_gradcam(extractor, query_tensor, query_label, index, device, layer_index=layer_index, use_scorecam=use_scorecam)
    
    activations = grad_cam.activations.squeeze(0).cpu()
    mean_activations = activations.mean(dim=(1, 2))
    top5_indices = torch.topk(mean_activations, k=5).indices

    all_images = [query_tensor] + similar_tensors
    all_labels = [query_label] + similar_labels
    all_raw_images = [query_image] + similar_images

    for feature_idx in top5_indices:
        fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))
        for i, (img_tensor, raw_img, label) in enumerate(zip(all_images, all_raw_images, all_labels)):
            grad_cam = GradCAM(extractor, extractor.extractor[layer_index])
            grad_cam(img_tensor) 
            act_map = grad_cam.activations.squeeze(0)[feature_idx].cpu().numpy()
            act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
            
            title = f"Query: {label}" if i == 0 else f"#{i}: {label}"
            im = axes[i].imshow(act_map, cmap='viridis')
            axes[i].set_title(f"{title}\nFeature #{feature_idx.item()}")
            axes[i].axis('off')

        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        plt.tight_layout()
        plt.savefig(f"feature_{feature_idx.item()}_activations_index_{index}_layer_{layer_index}.png")
        plt.show()

    print(f"Top 5 feature activation maps saved to feature_*_activations_index_{index}_layer_{layer_index}.png")

def main():
    parser = argparse.ArgumentParser(description="MNIST GradCAM and Image Retrieval")
    parser.add_argument('--index', type=int, required=True, help="Index of the MNIST test image")
    parser.add_argument('--layer', type=int, default=4, choices=[4, 7], help="Layer index for GradCAM (4 or 7)")
    parser.add_argument('--use-scorecam', action='store_true', help="Use Score-CAM instead of GradCAM")
    args = parser.parse_args()

    index = args.index
    print(f"Index : {index}")
    if index < 0 or index >= 10000:
        raise ValueError("Index must be between 0 and 9999 (MNIST test set size).")


    index = index % 12

    if index in [7,8,9,11]:
        index = 12

    index = np.random.choice([0,1,2,3,4,5,6,10,12,13,14,16,17,20,25,26,28,29,30,31,32,34,35])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    extractor, embedder = load_model("./model/best_model1.pt", device)
    db = FaissDatabase.load('./faiss_database/mnist_hash', use_gpu=torch.cuda.is_available())

    query_and_plot_similar_images(index, db, extractor, embedder, device, k=5, layer_index=args.layer, use_scorecam=args.use_scorecam)

if __name__ == "__main__":
    main()