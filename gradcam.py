import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import cv2

from build_faiss_db import load_model

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
        x = x.unsqueeze(0)  
        x.requires_grad_()
        
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

def get_mnist_dataset():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return mnist_test

def show_gradcam(model, image_tensor, class_idx=None, layer_name=None):
    """
    Generate and display GradCAM visualization and top 5 activation maps in a separate plot,
    with horizontal colorbars.
    """
    grad_cam = GradCAM(model, model.extractor[4])
    
    heatmap = grad_cam(image_tensor, class_idx)
    
    image = image_tensor.squeeze().detach().cpu().numpy()
    
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    image_rgb = np.repeat(image_norm[..., np.newaxis], 3, axis=-1)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    alpha = 0.7
    overlay = alpha * heatmap_color + (1 - alpha) * image_rgb
    overlay = np.clip(overlay, 0, 1)

    # --------------------------
    # Plot GradCAM visualizations with colorbar
    # --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title("Original Image")
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')

    axes[1].set_title("GradCAM Heatmap")
    im1 = axes[1].imshow(heatmap, cmap='viridis')
    axes[1].axis('off')

    axes[2].set_title("Overlay")
    axes[2].imshow(overlay , cmap = "viridis")
    axes[2].axis('off')

    cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.02])  
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


    with torch.no_grad():
        model.eval()
        _ = model(image_tensor.unsqueeze(0))  



    activations = grad_cam.activations.squeeze(0).cpu()  
    mean_activations = activations.mean(dim=(1, 2))       
    top12 = torch.topk(mean_activations, k=12)
    top12_indices = top12.indices
    top12_values = top12.values

    top12_maps = activations[top12_indices].numpy()
    top12_maps = [(m - m.min()) / (m.max() - m.min() + 1e-8) for m in top12_maps]


    fig2, axes2 = plt.subplots(4, 3, figsize=(24 , 14))  
    axes2 = axes2.flatten()

    ims = []
    for i, (act_map, idx, val) in enumerate(zip(top12_maps, top12_indices, top12_values)):
        axes2[i].set_title(f"Feature #{idx.item()}\nMean Act: {val.item():.4f}", fontsize=10)
        im = axes2[i].imshow(act_map, cmap='viridis')
        axes2[i].axis('off')
        ims.append(im)


    plt.subplots_adjust(wspace=0.5, hspace=0.1)

    for j in range(i + 1, len(axes2)):
        axes2[j].axis('off')

    cbar_ax2 = fig2.add_axes([0.2, 0.1, 0.6, 0.02])
    fig2.colorbar(ims[0], cax=cbar_ax2, orientation='horizontal')

    fig2.suptitle("Top 12 Activation Maps from Target Layer", fontsize=16)
    plt.tight_layout(rect=[0, 0.18, 1, 0.92])
    plt.show()

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        pred_class = torch.argmax(output).item()
        confidence = F.softmax(output, dim=1)[0, pred_class].item()

    print(f"Predicted class: {pred_class} with confidence: {confidence:.4f}")
    print(f"GradCAM visualization using layer: {layer_name}")

    return heatmap



def visualize_mnist_gradcam(index, path):
    """Visualize GradCAM for a specific MNIST image using both models"""
    extractor , embedder = load_model(path , device="cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_dataset = get_mnist_dataset()
    
    image, label = mnist_dataset[index]

    image = image.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"True label: {label}")
    
    
    print("\n=== Extractor Model GradCAM ===")
    extractor_heatmap = show_gradcam(extractor, image, class_idx=None)
    
    return extractor_heatmap

if __name__ == "__main__":    
    
    
    visualize_mnist_gradcam(6 , "./model/best_model1.pt")