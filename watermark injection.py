import json
import os
import random

import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from steganogan import SteganoGAN
import skimage.io
import torch
import torch.nn.functional as F
from torchvision import transforms
import shutil
import image_transforms

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

steganogan = SteganoGAN.load(architecture='25epoch-FHHQ')


def encode_messages_in_folder(input_folder, output_folder, text, encode_ratio=0.3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    encoded_images_folder = os.path.join(output_folder, 'encoded_images')
    if not os.path.exists(encoded_images_folder):
        os.makedirs(encoded_images_folder)

    selected_images_folder = os.path.join(output_folder, 'selected_images')
    if not os.path.exists(selected_images_folder):
        os.makedirs(selected_images_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    num_files_to_encode = int(len(image_files) * encode_ratio)


    files_to_encode = random.sample(image_files, num_files_to_encode)


    for filename in files_to_encode:
        input_path = os.path.join(input_folder, filename)

        encoded_output_path = os.path.join(encoded_images_folder, filename)

        selected_image_path = os.path.join(selected_images_folder, filename)
        shutil.copy(input_path, selected_image_path)

        steganogan.encode(input_path, encoded_output_path, text)


def decode_messages_in_folder(folder,benign=None,mse_output_path =None ):

    steganogan.total_attempts = 0
    steganogan.failed_attempts = 0
    steganogan.random_values_sum = 0
    steganogan.random_values_count = 0


    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder, filename)
            Flag = benign
            message = steganogan.decode(file_path ,Flag,mse_output_path)
            print(f'Decoded message from {file_path}: {message}')

    average_random_value = steganogan.average_random_value()
    print(f'BRA = : {average_random_value:.4f}')
    decode_rate = steganogan.get_decode_rate()
    print(f'Decode rate: {decode_rate:.2%}')
    print(f'Total attempts: {steganogan.total_attempts}')
    print(f'Failed attempts: {steganogan.failed_attempts}')

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def load_images(image_filepaths, img_size=256):
    image_batch_np = []
    for file_path in image_filepaths:
        image_from_file = skimage.io.imread(file_path)/255.0
        image_from_file = image_from_file[:, :, :3]
        image_batch_np.append(image_from_file)
    image_batch_np = np.stack(image_batch_np, axis=0)
    image_batch = torch.from_numpy(image_batch_np).float()
    image_batch = image_batch.permute(0, 3, 1, 2)

    h, w = image_batch.shape[2:]
    if h > w:
        image_batch = image_batch[:, :, int((h-w)/2):int((h+w)/2), :]
    elif w > h:
        image_batch = image_batch[:, :, :, int((w-h)/2):int((w+h)/2)]
    image_batch = F.interpolate(image_batch, size=(img_size, img_size), mode='bilinear', align_corners=True)

    return image_batch

def save_images(image_batch, out_dir, prefix=""):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image_paths = []
    for img_idx in range(image_batch.shape[0]):
        image_np = image_batch[img_idx].permute(1, 2, 0).cpu().numpy()
        image_np = np.uint8(image_np*255.)
        file_path = os.path.join(out_dir, "{}_{}.png".format(prefix, img_idx))
        PIL.Image.fromarray(image_np).save(file_path)
        image_paths.append(file_path)

    return image_paths

def find_image_paths(image_dir):
    image_paths = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            image_paths.append(os.path.join(image_dir, img_file))
    image_paths.sort()
    return image_paths

def showarray(a):
    """
    takes a numpy array (0 to 1) of size h, w, 3
    """
    a = np.uint8(a * 255)
    plt.imshow(a)
    plt.axis('off')  # Hide axes for better visual
    plt.show()

def load_image_as_tensor(image_path):

    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    return image_tensor

def visualize_residuals(images_dir, signed_image_dir, mse_output_path):
    def save_residual_image(residual_scaled, save_path):

        residual_scaled_uint8 = (residual_scaled * 255).astype(np.uint8)
        residual_image = Image.fromarray(residual_scaled_uint8)

        residual_image.save(save_path)
        print(f"Residual scaled image saved to: {save_path}")

    if not os.path.exists(mse_output_path):
        with open(mse_output_path, 'w') as json_file:
            json.dump({}, json_file)
    original_image_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir)) if
                            f.endswith(('.png', '.jpg', '.jpeg'))]
    encoded_image_paths = [os.path.join(signed_image_dir, f) for f in sorted(os.listdir(signed_image_dir)) if
                             f.endswith('.png')]

    save_directory = os.path.join(os.path.dirname(images_dir), "residual")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(mse_output_path, 'r') as json_file:
        mse_results = json.load(json_file)

    for sidx in range(len(encoded_image_paths)):
        original_image_path = original_image_paths[sidx]
        encoded_image_path = encoded_image_paths[sidx]


        original_image_tensor = load_image_as_tensor(original_image_path)
        encoded_image_tensor = load_image_as_tensor(encoded_image_path)

        original_image_numpy = original_image_tensor.permute(1, 2, 0).cpu().numpy()
        encoded_image_numpy = encoded_image_tensor.permute(1, 2, 0).cpu().numpy()
        residual = (encoded_image_numpy - original_image_numpy)
        rmin, rmax = np.min(residual), np.max(residual)
        residual_scaled = (residual - rmin) / (rmax - rmin)

        mse = float(np.mean((residual_scaled - original_image_numpy) ** 2))
        residual_image_name = f"residual_scaled_{sidx}.png"
        mse_results[f"image_{sidx}"] = {
            "mse": mse,
            "residual_image": residual_image_name
        }
        residual_image_path = os.path.join(save_directory, residual_image_name)
        save_residual_image(residual_scaled, residual_image_path)

        original_encoded_image = np.concatenate((original_image_numpy, encoded_image_numpy, residual_scaled), axis=1)
        print("Original Image,", "Signed Image,", "Perturbation (Scaled for Visualization)")
        showarray(original_encoded_image)

    with open(mse_output_path, 'w') as json_file:
        json.dump(mse_results, json_file, indent=4)
    print(f"MSE results saved to: {mse_output_path}")

    def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
        n = int(bits, 2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

def benign_transformations(signed_image_dir):

    encoded_image_paths = [os.path.join(signed_image_dir, f) for f in sorted(os.listdir(signed_image_dir)) if
                           f.endswith('.png')]
    benign_dir = os.path.join(out_dir, "benign_transformed_images")
    image_transforms.apply_benign_transforms(encoded_image_paths, benign_dir)



def apply_malicious_transforms(signed_image_dir, target_image_dir, out_dir):
    mal_dir = os.path.join(out_dir, "mal_transformed_images")
    target_image_paths = find_image_paths(target_image_dir)



def calculate_and_save_averages(json_file, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    mse_sum = 0
    acc_sum = 0
    mse_count = 0
    acc_count = 0

    for key, value in data.items():
        if 'mse' in value:
            mse_sum += value['mse']
            mse_count += 1
        if 'bit_acc' in value:
            acc_sum += value['bit_acc']
            acc_count += 1

    mse_avg = mse_sum / mse_count if mse_count != 0 else 0
    acc_avg = acc_sum / acc_count if acc_count != 0 else 0

    weighted_avg = 0.5 *  (1-mse_avg)  + 0.5 * acc_avg


    with open(output_file, 'w') as output:
        output.write(f"MSE Average: {mse_avg}\n")
        output.write(f"ACC Average: {acc_avg}\n")
        output.write(f"weighted_avg: { weighted_avg}\n")


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

secret_text = 'secret message!'
secret_size = 128
secret_bits = text_to_bits(secret_text)
secrete_num_bits = len(secret_bits)

assert secrete_num_bits <= secret_size
message = secret_text  +"".join(["0"]*(secret_size-secrete_num_bits))

encode_messages_in_folder(input_folder,signed_image_dir, message,encode_ratio=1)
