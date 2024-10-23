import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import nibabel as nib
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import random

# Установка сида для воспроизводимости
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Вы можете выбрать любое число в качестве сида

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(nii_file, slice_index):
    nii_image = nib.load(nii_file)
    image_data = nii_image.get_fdata()

    # Загрузка среднего среза
    image_slice = image_data[:, :, slice_index]

    # Предобработка
    image_pil = Image.fromarray(image_slice.astype(np.uint8))
    transformations = Compose([
        Resize((256, 256)),
        ToTensor()
    ])
    processed_image = transformations(image_pil)

    # Нормализация изображения
    processed_image = Normalize(mean=[0.485], std=[0.229])(processed_image)
    processed_image = processed_image.unsqueeze(0)  # Добавление размерности batch
    return processed_image, image_pil

# Функция для визуализации изображения
def plot_image(ax, image):
    ax.imshow(image.squeeze().numpy(), cmap='gray')
    ax.set_title("Original Image")
    ax.axis("off")

# Функция для визуализации маски
def plot_predicted_mask(ax, predicted_mask):
    ax.imshow(predicted_mask, cmap='gray')
    ax.set_title("Predicted Mask")
    ax.axis("off")

# Функция для визуализации наложения маски на изображение
def plot_overlay(ax, image, predicted_mask):
    ax.imshow(image.squeeze().numpy(), cmap='gray')
    ax.imshow(predicted_mask, alpha=0.5, cmap='jet')
    ax.set_title("Overlay")
    ax.axis("off")

# Функция для предсказания маски
def predict_mask(model, image, device):
    with torch.no_grad():
        output = model(image.to(device))
        pred_mask = torch.argmax(output, dim=1)
    return pred_mask

# Функция для визуализации результатов
def visualize_inference_results(images, gt_masks, pred_masks, n_ims=15):
    cols = 3
    rows = n_ims // cols

    plt.figure(figsize=(25, 20))
    count = 1

    for idx in range(len(images)):
        if count > n_ims * 3:  # Проверяем, если уже отображены все необходимые изображения
            break

        # Первый график - оригинальное изображение
        plt.subplot(rows, cols, count)
        plt.imshow(images[idx].squeeze(0), cmap='gray')
        plt.axis('off')
        plt.title("Original Image")
        count += 1

        # Второй график - целевая маска (Ground Truth)
        plt.subplot(rows, cols, count)
        plt.imshow(gt_masks[idx].squeeze(0), cmap='gray')
        plt.axis('off')
        plt.title("Ground Truth")
        count += 1

        # Третий график - предсказанная моделью маска
        plt.subplot(rows, cols, count)
        plt.imshow(pred_masks[idx].squeeze(0), cmap='gray')
        plt.axis('off')
        plt.title("Predicted Mask")
        count += 1

    plt.tight_layout()
    plt.show()

# Функция для выполнения вывода модели на входном DataLoader
def inference(dl, model, device, n_ims=15):
    images, gt_masks, pred_masks = [], [], []

    for idx, data in enumerate(dl):
        if idx == n_ims:
            break

        image, gt_mask = data
        pred_mask = predict_mask(model, image, device)

        images.append(image.cpu())
        gt_masks.append(gt_mask.cpu())
        pred_masks.append(pred_mask.cpu())

    visualize_inference_results(images, gt_masks, pred_masks, n_ims)

# Путь к файлу модели и изображению
nii_file = 'testmodel/images/liver_1.nii'
model_path = 'testmodel/model2.pth'

# Загрузка и предобработка изображения
slice_index = 32  # Средний срез
processed_image, image_pil = load_and_preprocess_image(nii_file, slice_index)

# Создание пользовательской модели U-Net
class CustomUnet(smp.Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )
        self.segmentation_head = torch.nn.Conv2d(16, 2, kernel_size=(1, 1), stride=(1, 1))

# Инициализация модели с правильным декодером
model = CustomUnet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,  # Один канал для одноканальных медицинских изображений
    classes=2,
    activation=None
)

# Загрузка state_dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)  # Используйте strict=False для игнорирования несовпадений
model.eval()

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Пример использования с тестовым DataLoader (test_dl)
# Предположим, что test_dl уже определен и загружен с правильными данными
# inference(test_dl, model=model, device=device)

# Для отображения одного изображения
with torch.no_grad():
    prediction = model(processed_image.to(device))

# Обработка предсказания модели, получение бинарной маски
predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu()

# Проверка уникальных значений в маске
print(f"Unique values in the predicted mask: {torch.unique(predicted_mask)}")

# Отображение результата
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Оригинальное изображение
plot_image(axes[0], processed_image)

# Предсказанная маска
plot_predicted_mask(axes[1], predicted_mask)

# Наложение маски на оригинальное изображение
plot_overlay(axes[2], processed_image, predicted_mask)

plt.tight_layout()
plt.show()
