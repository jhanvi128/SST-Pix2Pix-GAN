#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SST Data Generation using Pix2Pix GAN
For use in Google Colab
With improved handling of land-sea boundaries and NaN values
"""

# Import necessary libraries
import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import shutil
import argparse
from scipy import ndimage
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define paths for Google Drive
input_dir = '/content/drive/MyDrive/data/sst'
output_dir = '/content/drive/MyDrive/data/sst_missing'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parse command line arguments for compatibility
parser = argparse.ArgumentParser(description='SST Data Generation using Pix2Pix GAN')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate (default: 0.0002)')
parser.add_argument('--lambda_l1', type=float, default=200.0, help='Weight for L1 loss (default: 200.0)')
parser.add_argument('--land_penalty', type=float, default=50.0, help='Penalty weight for incorrect land values (default: 50.0)')
parser.add_argument('--coastal_penalty', type=float, default=30.0, help='Penalty weight for coastal temperature gradients (default: 30.0)')
parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model for generation only')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'], 
                    help='Mode: train for training model, generate for using pre-trained model')
parser.add_argument('--validation_split', type=float, default=0.15, help='Fraction of data to use for validation')
parser.add_argument('--coastal_buffer', type=int, default=3, help='Buffer size for coastal areas (default: 3)')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List of missing dates (as provided)
missing_dates = [
    '20060728', '20060813', '20071024', '20080125', '20080210', '20080508', '20080711',
    '20100415', '20100821', '20101211', '20111024', '20120804', '20120913', '20130306',
    '20130906', '20140829', '20160101', '20161202', '20170125', '20170517', '20181211',
    '20190829', '20220407'
]

# Function to parse dates from filenames
def parse_date_from_filename(filename):
    # Extract the date range from the filename
    parts = os.path.basename(filename).split('.')
    date_range = parts[1]
    start_date, end_date = date_range.split('_')
    return start_date, end_date

# Function to calculate next date range
def get_next_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    next_date = date_obj + timedelta(days=8)
    return next_date.strftime('%Y%m%d')

# Function to calculate previous date range
def get_prev_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    prev_date = date_obj - timedelta(days=8)
    return prev_date.strftime('%Y%m%d')

# Function to find adjacent date files for a missing date
def find_adjacent_files(missing_date, all_files):
    # Convert missing date to datetime
    missing_date_obj = datetime.strptime(missing_date, '%Y%m%d')
    
    # Calculate expected filenames for before and after
    prev_start = get_prev_date(missing_date)
    prev_end = missing_date
    
    next_start = missing_date
    next_end = get_next_date(missing_date)
    
    # Special case for year-end
    if missing_date_obj.month == 12 and missing_date_obj.day > 23:
        # Year-end case, adjust end date
        year = missing_date_obj.year
        next_end = f"{year}1231"
    
    # Find closest file before missing date
    prev_files = [f for f in all_files if parse_date_from_filename(f)[1][:8] <= missing_date]
    prev_file = max(prev_files, key=lambda f: parse_date_from_filename(f)[1]) if prev_files else None
    
    # Find closest file after missing date
    next_files = [f for f in all_files if parse_date_from_filename(f)[0][:8] >= missing_date]
    next_file = min(next_files, key=lambda f: parse_date_from_filename(f)[0]) if next_files else None
    
    return prev_file, next_file

# Function to create coastal mask
def create_coastal_mask(land_mask, buffer_size=3):
    """
    Create a mask identifying coastal areas (areas close to land/water boundaries)
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(land_mask, torch.Tensor):
        land_mask_np = land_mask.cpu().numpy()
    else:
        land_mask_np = land_mask
    
    # Dilate the land mask to include coastal waters
    dilated = ndimage.binary_dilation(land_mask_np, iterations=buffer_size)
    
    # Coastal areas are the difference between dilated land and original land
    coastal = dilated & ~land_mask_np
    
    # Convert back to tensor if input was tensor
    if isinstance(land_mask, torch.Tensor):
        return torch.from_numpy(coastal).to(land_mask.device)
    
    return coastal

# Improved function to load SST data from NetCDF file with better land-sea boundary handling
def load_sst_data(file_path):
    try:
        with xr.open_dataset(file_path) as ds:
            # Extract SST data
            sst = ds['sst'].values
            
            # Create a land mask (True for land, False for water)
            land_mask = np.isnan(sst)
            
            # Store the original NaN positions for later restoration
            nan_mask = land_mask.copy()
            
            # Use morphological operations to improve land-sea boundaries
            # This helps identify problematic pixels at the boundaries
            dilated_land = ndimage.binary_dilation(land_mask, iterations=1)
            eroded_land = ndimage.binary_erosion(land_mask, iterations=1)
            boundary_pixels = dilated_land & ~eroded_land
            
            # Handle missing values (NaN -> specific value for training)
            # Use a distinct negative value that won't be confused with real temperatures
            sst = np.nan_to_num(sst, nan=-999.0)
            
            # Normalize to [0, 1] range for model input, excluding the land values
            sst_water = sst[~land_mask]
            if len(sst_water) > 0:  # Check if there's any water data
                sst_min, sst_max = np.min(sst_water), np.max(sst_water)
                if sst_max > sst_min:
                    # Only normalize water pixels, leave land pixels as -999
                    sst_normalized = sst.copy()
                    sst_normalized[~land_mask] = (sst_water - sst_min) / (sst_max - sst_min)
                    sst_normalized[land_mask] = -1.0  # Use -1 for land in normalized space
                else:
                    sst_normalized = sst
                    sst_min, sst_max = 0, 1
            else:
                sst_min, sst_max = 0, 1
                sst_normalized = sst
            
            # Apply a small amount of smoothing to the coastal boundary pixels
            # This helps the model learn more natural transitions
            if np.any(boundary_pixels):
                # Only work with water pixels at the boundary
                water_boundary = boundary_pixels & ~land_mask
                if np.any(water_boundary):
                    # Extract neighboring values
                    kernel = np.ones((3, 3)) / 8.0  # 3x3 kernel excluding center
                    kernel[1, 1] = 0  # Don't include the center pixel
                    
                    # Apply convolution to get average of neighboring pixels
                    # This smooths the boundaries without affecting the land pixels
                    temp_array = sst_normalized.copy()
                    temp_array[land_mask] = 0  # Set land to 0 for convolution
                    smoothed = ndimage.convolve(temp_array, kernel, mode='constant', cval=0)
                    
                    # Adjust the weights for boundary pixels
                    # 0.7 * original value + 0.3 * smoothed neighborhood
                    sst_normalized[water_boundary] = 0.7 * sst_normalized[water_boundary] + 0.3 * smoothed[water_boundary]
            
            # Store min and max for denormalization later
            meta = {
                'min': sst_min,
                'max': sst_max,
                'lats': ds['lat'].values,
                'lons': ds['lon'].values,
                'attributes': ds.attrs,
                'sst_attrs': ds['sst'].attrs,
                'nan_mask': nan_mask,  # Store the NaN mask for restoration
                'boundary_pixels': boundary_pixels  # Store boundary pixels for special handling
            }
            
            return sst_normalized, meta
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# Function to smooth coastal boundaries in the final output
def smooth_coastal_boundaries(sst_data, land_mask, n_iterations=2, kernel_size=3):
    """
    Apply a smoothing filter to coastal boundaries to ensure
    gradual temperature transitions
    """
    smooth_data = sst_data.copy()
    
    # Identify coastal zones
    dilated_mask = ndimage.binary_dilation(land_mask, iterations=2)
    coastal_zone = dilated_mask & ~land_mask
    
    # Apply a weighted smoothing to coastal areas only
    for _ in range(n_iterations):
        # Create a temporary array with NaNs replaced by zeros for processing
        temp_array = smooth_data.copy()
        temp_array[land_mask] = 0
        
        # Create a kernel with specified size
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size*kernel_size - 1)
        center = kernel_size // 2
        kernel[center, center] = 0  # Remove center to avoid self-contribution
        
        # Apply convolution
        smoothed = ndimage.convolve(temp_array, kernel, mode='reflect')
        
        # Weight for blending (gradually increase weight of original data near land)
        # Calculate distance from land
        dist_from_land = ndimage.distance_transform_edt(~land_mask)
        dist_from_land = np.clip(dist_from_land, 0, 3) / 3  # Normalize to [0, 1] with max of 3 pixels
        
        # Apply weighted average in coastal zone
        # Weight: 0.6 * original + 0.4 * smoothed for closest to land
        # Weight: 0.9 * original + 0.1 * smoothed for furthest in coastal zone
        weight_original = 0.6 + 0.3 * dist_from_land
        weight_original = weight_original.reshape(smooth_data.shape)
        
        blend_mask = coastal_zone & ~np.isnan(smooth_data)
        if np.any(blend_mask):
            smooth_data[blend_mask] = (
                weight_original[blend_mask] * smooth_data[blend_mask] + 
                (1 - weight_original[blend_mask]) * smoothed[blend_mask]
            )
    
    # Ensure land areas are still NaN
    smooth_data[land_mask] = np.nan
    
    return smooth_data
# Custom Dataset for SST data with improved coastal handling
class SSTDataset(Dataset):
    def __init__(self, file_list, transform=None, coastal_buffer=3):
        self.file_list = file_list
        self.transform = transform
        self.coastal_buffer = coastal_buffer
        
        # Create pairs of consecutive files
        self.pairs = []
        sorted_files = sorted(file_list)
        for i in range(len(sorted_files) - 1):
            self.pairs.append((sorted_files[i], sorted_files[i+1]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        file1, file2 = self.pairs[idx]
        
        # Load data from both files
        sst1, meta1 = load_sst_data(file1)
        sst2, meta2 = load_sst_data(file2)
        
        if sst1 is None or sst2 is None:
            # Return zeros if file loading failed
            sst1 = np.zeros((1, 256, 256), dtype=np.float32)
            sst2 = np.zeros((1, 256, 256), dtype=np.float32)
            # Create dummy land mask
            land_mask = np.zeros((256, 256), dtype=bool)
            coastal_mask = np.zeros((256, 256), dtype=bool)
        else:
            # Get land mask from the first file
            land_mask = (sst1 == -1.0)
            
            # Create coastal mask
            coastal_mask = create_coastal_mask(land_mask, self.coastal_buffer)
            
            # Reshape for CNN input (add channel dimension if needed)
            if len(sst1.shape) == 2:
                sst1 = sst1[np.newaxis, :, :]
            if len(sst2.shape) == 2:
                sst2 = sst2[np.newaxis, :, :]
        
        # Resize to a standard size for the model
        sst1_resized = cv2.resize(sst1[0], (256, 256))
        sst2_resized = cv2.resize(sst2[0], (256, 256))
        
        # Also resize the land mask and coastal mask
        if len(land_mask.shape) == 2:
            land_mask_resized = cv2.resize(land_mask.astype(np.float32), (256, 256)) > 0.5
        else:
            land_mask_resized = cv2.resize(land_mask[0].astype(np.float32), (256, 256)) > 0.5
            
        if len(coastal_mask.shape) == 2:
            coastal_mask_resized = cv2.resize(coastal_mask.astype(np.float32), (256, 256)) > 0.5
        else:
            coastal_mask_resized = cv2.resize(coastal_mask[0].astype(np.float32), (256, 256)) > 0.5
        
        # Convert to PyTorch tensors
        sst1_tensor = torch.FloatTensor(sst1_resized).unsqueeze(0)  # Add channel dimension
        sst2_tensor = torch.FloatTensor(sst2_resized).unsqueeze(0)  # Add channel dimension
        land_mask_tensor = torch.BoolTensor(land_mask_resized)
        coastal_mask_tensor = torch.BoolTensor(coastal_mask_resized)
        
        # Apply transformations if specified
        if self.transform:
            sst1_tensor = self.transform(sst1_tensor)
            sst2_tensor = self.transform(sst2_tensor)
        
        return sst1_tensor, sst2_tensor, land_mask_tensor, coastal_mask_tensor

# Define the Generator with enhanced land-sea boundary handling
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Modified final layer for SST prediction
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            # No activation function to allow unconstrained temperature values
        )
        
        # Special branch for land mask prediction (binary classification)
        self.land_mask_predictor = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        
        # Extra branch for coastal area detection (helps with boundary handling)
        self.coastal_detector = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        # Final SST prediction (main branch)
        sst_pred = self.up8(torch.cat([u7, d1], 1))
        
        # Predict land mask (binary)
        land_mask_pred = self.land_mask_predictor(u7)
        
        # Predict coastal areas (helps with boundary handling)
        coastal_pred = self.coastal_detector(u7)
        
        # Create binary land mask (1 for land, 0 for water)
        land_binary = (land_mask_pred > 0.5).float()
        
        # Apply land mask to SST prediction (force water values for water, -1 for land)
        masked_sst = sst_pred * (1 - land_binary) + (-1) * land_binary
        
        # Apply coastal awareness - use coastal prediction to weight transition area
        # This helps create a smoother boundary between land and water
        coastal_weight = coastal_pred * 0.3  # Scale down coastal influence
        
        # The final output is a blend at the coastal regions
        final_sst = masked_sst
        
        return final_sst, land_mask_pred, coastal_pred

# Define the Discriminator with improved land-sea boundary awareness
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()
        
        # First layer without normalization
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Subsequent layers with batch normalization
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Special layer for land mask contribution
        self.land_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final layer for classification
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y, land_mask=None):
        # Concatenate input and target images
        if land_mask is None:
            concat_input = torch.cat([x, y], dim=1)
            
            # Process through standard layers
            feat1 = self.layer1(concat_input)
            feat2 = self.layer2(feat1)
            feat3 = self.layer3(feat2)
            feat4 = self.layer4(feat3)
            
            # Final classification
            output = self.classifier(torch.cat([feat4, feat3[:, :128]], dim=1))
        else:
            # Concatenate input and target images
            concat_input = torch.cat([x, y], dim=1)
            
            # Process through standard layers
            feat1 = self.layer1(concat_input)
            feat2 = self.layer2(feat1)
            feat3 = self.layer3(feat2)
            feat4 = self.layer4(feat3)
            
            # Process land mask information
            land_feat = self.land_layer(land_mask.float().unsqueeze(1))
            
            # Final classification with land mask contribution
            output = self.classifier(torch.cat([feat4, land_feat], dim=1))
            
        return output
# Define custom loss functions for improved coastal handling
def masked_l1_loss(pred, target, land_mask, land_penalty=50.0):
    """
    L1 loss that:
    1. Ignores land areas in the calculation of the water temperature loss
    2. Adds an extra penalty for water predictions in land areas
    """
    # Convert to Boolean if not already
    if not isinstance(land_mask, torch.BoolTensor):
        land_mask = land_mask.bool()
    
    # Standard L1 loss for water areas
    water_loss = torch.abs(pred[~land_mask] - target[~land_mask]).mean() if torch.any(~land_mask) else torch.tensor(0.0).to(pred.device)
    
    # Penalty for values in land areas that should be -1
    land_loss = torch.abs(pred[land_mask] - (-1)).mean() if torch.any(land_mask) else torch.tensor(0.0).to(pred.device)
    
    return water_loss + land_penalty * land_loss

# Fix the coastal_gradient_loss function
def coastal_gradient_loss(pred, target, coastal_mask, penalty=30.0):
    """
    Special loss for coastal areas that penalizes abrupt changes in temperature.
    This encourages smoother transitions between land and sea.
    """
    if not isinstance(coastal_mask, torch.BoolTensor):
        coastal_mask = coastal_mask.bool()
    
    if not torch.any(coastal_mask):
        return torch.tensor(0.0).to(pred.device)
    
    # Extract coastal area temperatures - add unsqueeze to handle channel dimension
    coastal_mask_expanded = coastal_mask.unsqueeze(1)  # Add channel dimension
    pred_coastal = pred[coastal_mask_expanded]
    target_coastal = target[coastal_mask_expanded]
    
    # L1 loss for coastal areas (weighted higher)
    coastal_l1 = torch.abs(pred_coastal - target_coastal).mean()
    
    # Get gradient of coastal areas
    # Horizontal gradients
    pred_gradient_h = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_gradient_h = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # Vertical gradients
    pred_gradient_v = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_gradient_v = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    
    # Prepare coastal masks for gradient comparisons
    coastal_mask_h = coastal_mask[:, :, 1:] | coastal_mask[:, :, :-1]  # Horizontal mask
    coastal_mask_v = coastal_mask[:, 1:, :] | coastal_mask[:, :-1, :]  # Fixed: Vertical mask
    
    # Add channel dimension for masks to match gradients
    coastal_mask_h = coastal_mask_h.unsqueeze(1)
    coastal_mask_v = coastal_mask_v.unsqueeze(1)
    
    # Extract only the coastal gradients
    h_loss = torch.abs(pred_gradient_h[coastal_mask_h] - target_gradient_h[coastal_mask_h]).mean() if torch.any(coastal_mask_h) else torch.tensor(0.0).to(pred.device)
    v_loss = torch.abs(pred_gradient_v[coastal_mask_v] - target_gradient_v[coastal_mask_v]).mean() if torch.any(coastal_mask_v) else torch.tensor(0.0).to(pred.device)
    
    # Combine losses
    gradient_loss = (h_loss + v_loss) / 2.0
    
    return coastal_l1 + penalty * gradient_loss


def boundary_continuity_regularizer(pred, land_mask, coastal_mask):
    """
    Encourages smooth transitions at the boundaries between land and water
    """
    if not torch.any(coastal_mask):
        return torch.tensor(0.0).to(pred.device)
    
    # Create water coastal mask
    water_coastal = coastal_mask & ~land_mask
    
    if not torch.any(water_coastal):
        return torch.tensor(0.0).to(pred.device)
    
    # Squeeze pred if it has channel dimension
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    
    # Calculate gradients without explicit padding
    # Horizontal gradients (right - left)
    h_grad = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
    
    # Vertical gradients (down - up)
    v_grad = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
    
    # Create masks for valid gradient positions
    h_mask = water_coastal[:, :, :-1] & water_coastal[:, :, 1:]
    v_mask = water_coastal[:, :-1, :] & water_coastal[:, 1:, :]
    
    # Calculate mean gradients if masks are not empty
    h_loss = torch.mean(h_grad[:, h_mask]) if torch.any(h_mask) else torch.tensor(0.0).to(pred.device)
    v_loss = torch.mean(v_grad[:, v_mask]) if torch.any(v_mask) else torch.tensor(0.0).to(pred.device)
    
    # Return mean of horizontal and vertical losses
    return (h_loss + v_loss) / 2.0

# Validation function with improved metrics
def validate(generator, val_dataloader, epoch, land_penalty=50.0, coastal_penalty=30.0):
    generator.eval()
    val_losses = []
    land_errors = []
    coastal_errors = []
    water_errors = []
    
    with torch.no_grad():
        for prev_sst, next_sst, land_mask, coastal_mask in val_dataloader:
            prev_sst = prev_sst.to(device)
            next_sst = next_sst.to(device)
            land_mask = land_mask.to(device)
            coastal_mask = coastal_mask.to(device)
            
            # Generate prediction
            fake_sst, land_pred, _ = generator(prev_sst)
            
            # Calculate main loss
            l1_loss = masked_l1_loss(fake_sst, next_sst, land_mask, land_penalty)
            val_losses.append(l1_loss.item())
            
            # Calculate land error (accuracy of land mask prediction)
            land_pred_binary = (land_pred > 0.5).float()
            land_accuracy = (land_pred_binary == land_mask.float().unsqueeze(1)).float().mean()
            land_errors.append((1.0 - land_accuracy.item()) * 100)  # Error percentage
            
            # Calculate coastal error
            if torch.any(coastal_mask):
                coastal_error = torch.abs(fake_sst[:, 0][coastal_mask] - next_sst[:, 0][coastal_mask]).mean()
                coastal_errors.append(coastal_error.item())
            
            # Calculate water error (away from coast)
            water_only = ~land_mask & ~coastal_mask
            if torch.any(water_only):
                water_error = torch.abs(fake_sst[:, 0][water_only] - next_sst[:, 0][water_only]).mean()
                water_errors.append(water_error.item())
    
    avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0
    avg_land_error = sum(land_errors) / len(land_errors) if land_errors else 0
    avg_coastal = sum(coastal_errors) / len(coastal_errors) if coastal_errors else 0
    avg_water = sum(water_errors) / len(water_errors) if water_errors else 0
    
    print(f"Validation - Epoch {epoch}, Loss: {avg_loss:.4f}, Land Error: {avg_land_error:.2f}%, "
          f"Coastal Error: {avg_coastal:.4f}, Water Error: {avg_water:.4f}")
    
    return avg_loss, avg_coastal, avg_water, avg_land_error
# Train the GAN with improved coastal handling
def train_gan(file_list, epochs=100, batch_size=1, save_interval=10, checkpoint_path=None, 
              validation_split=0.15, lr=0.0002, lambda_l1=200.0, land_penalty=50.0, 
              coastal_penalty=30.0, coastal_buffer=3):
    # Create dataset
    full_dataset = SSTDataset(file_list, coastal_buffer=coastal_buffer)
    
    # Split dataset into training and validation
    val_size = int(validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize generator and discriminator
    generator = UNetGenerator().to(device)
    discriminator = Discriminator(in_channels=2).to(device)
    
    # Define loss functions
    criterion_gan = nn.BCELoss()
    
    # Define optimizers with custom learning rate
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Setup learning rate schedulers
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.9)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=30, gamma=0.9)
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        if 'optimizer_g' in checkpoint:
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if 'optimizer_d' in checkpoint:
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
    
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create log file for training metrics
    log_file = open('training_log.csv', 'a')
    if os.path.getsize('training_log.csv') == 0:
        log_file.write('epoch,g_loss,d_loss,val_loss,coastal_error,water_error,land_error\n')
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_g_loss_gan = 0.0
        running_g_loss_l1 = 0.0
        running_g_loss_coastal = 0.0
        
        generator.train()
        discriminator.train()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (prev_sst, next_sst, land_mask, coastal_mask) in enumerate(pbar):
                # Move data to device
                prev_sst = prev_sst.to(device)
                next_sst = next_sst.to(device)
                land_mask = land_mask.to(device)
                coastal_mask = coastal_mask.to(device)
                
                # Ground truth labels
                real_label = torch.ones(prev_sst.size(0), 1, 30, 30).to(device)
                fake_label = torch.zeros(prev_sst.size(0), 1, 30, 30).to(device)
                
                # -----------------
                # Train Discriminator
                # -----------------
                optimizer_d.zero_grad()
                
                # Real loss
                real_output = discriminator(prev_sst, next_sst, land_mask.unsqueeze(1).float())
                d_real_loss = criterion_gan(real_output, real_label)
                
                # Fake loss
                fake_sst, land_pred, _ = generator(prev_sst)
                fake_output = discriminator(prev_sst, fake_sst.detach(), land_mask.unsqueeze(1).float())
                d_fake_loss = criterion_gan(fake_output, fake_label)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()
                optimizer_d.step()
                
                # -----------------
                # Train Generator
                # -----------------
                optimizer_g.zero_grad()
                
                # GAN loss
                fake_output = discriminator(prev_sst, fake_sst, land_mask.unsqueeze(1).float())
                g_loss_gan = criterion_gan(fake_output, real_label)
                
                # L1 loss with land masking
                g_loss_l1 = masked_l1_loss(fake_sst, next_sst, land_mask, land_penalty) * lambda_l1
                
                # Coastal gradient loss
                g_loss_coastal = coastal_gradient_loss(fake_sst, next_sst, coastal_mask, coastal_penalty)
                
                # Land mask prediction loss
                g_loss_mask = nn.BCELoss()(land_pred, land_mask.float().unsqueeze(1)) * 10.0
                
                # Boundary continuity regularization
                g_loss_boundary = boundary_continuity_regularizer(fake_sst, land_mask, coastal_mask) * 20.0
                
                # Total generator loss
                g_loss = g_loss_gan + g_loss_l1 + g_loss_coastal + g_loss_mask + g_loss_boundary
                g_loss.backward()
                optimizer_g.step()
                
                # Update progress bar
                running_d_loss += d_loss.item()
                running_g_loss += g_loss.item()
                running_g_loss_gan += g_loss_gan.item()
                running_g_loss_l1 += g_loss_l1.item()
                running_g_loss_coastal += g_loss_coastal.item()
                
                pbar.set_postfix({
                    'D Loss': running_d_loss / (i + 1), 
                    'G Loss': running_g_loss / (i + 1),
                    'G_GAN': running_g_loss_gan / (i + 1),
                    'G_L1': running_g_loss_l1 / (i + 1),
                    'G_Coast': running_g_loss_coastal / (i + 1)
                })
        
        # Update learning rate schedulers
        scheduler_g.step()
        scheduler_d.step()
        
        # Validation at the end of each epoch
        val_loss, coastal_error, water_error, land_error = validate(
            generator, val_dataloader, epoch+1, land_penalty, coastal_penalty
        )
        
        # Write metrics to log file
        log_file.write(f"{epoch+1},{running_g_loss/(i+1)},{running_d_loss/(i+1)},{val_loss},{coastal_error},{water_error},{land_error}\n")
        log_file.flush()
        
        # Save checkpoint based on validation performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_file = f'checkpoints/sst_gan_best.pth'
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'best_val_loss': best_val_loss,
                'coastal_error': coastal_error,
                'water_error': water_error,
                'land_error': land_error
            }, best_checkpoint_file)
            print(f"New best model saved: {best_checkpoint_file} (val_loss: {val_loss:.4f})")
        
        # Save model every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_file = f'checkpoints/sst_gan_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'val_loss': val_loss,
                'coastal_error': coastal_error,
                'water_error': water_error,
                'land_error': land_error,
                'best_val_loss': best_val_loss
            }, checkpoint_file)
            print(f"Checkpoint saved: {checkpoint_file}")
    
    # Close log file
    log_file.close()
    
    # Save final model
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'val_loss': val_loss,
        'coastal_error': coastal_error,
        'water_error': water_error,
        'land_error': land_error
    }, 'checkpoints/sst_gan_final.pth')
    
    return generator

# Function to generate missing SST data with improved coastal handling
def generate_missing_sst(missing_date, all_files, model_path, output_dir):
    print(f"Generating SST data for {missing_date}")
    
    # Find adjacent files
    prev_file, next_file = find_adjacent_files(missing_date, all_files)
    
    if prev_file is None or next_file is None:
        print(f"Error: Could not find adjacent files for {missing_date}")
        return False
    
    # Load data from adjacent files
    prev_sst, prev_meta = load_sst_data(prev_file)
    next_sst, next_meta = load_sst_data(next_file)
    
    if prev_sst is None or next_sst is None:
        print(f"Error: Could not load data for {missing_date}")
        return False
    
    # Load the trained model
    generator = UNetGenerator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Extract land mask from the previous data
    land_mask = (prev_sst == -1.0)
    
    # Reshape for model input
    if len(prev_sst.shape) == 2:
        prev_sst = prev_sst[np.newaxis, np.newaxis, :, :]
    else:
        prev_sst = prev_sst[np.newaxis, :, :]
    
    # Convert to tensor
    prev_tensor = torch.FloatTensor(prev_sst).to(device)
    
    # Generate prediction
    with torch.no_grad():
        pred_sst, land_pred, _ = generator(prev_tensor)
    
    # Convert prediction to numpy
    pred_sst_np = pred_sst.cpu().numpy()[0, 0]
    
    # Ensure land areas are marked correctly
    land_pred_np = (land_pred.cpu().numpy()[0, 0] > 0.5)
    pred_sst_np[land_pred_np] = -1.0
    
    # Ensure exact land/water consistency with source data
    pred_sst_np[land_mask] = -1.0
    
    # Post-process the prediction to smooth coastal boundaries
    coastal_mask = create_coastal_mask(land_mask, buffer_size=3)
    
    # Denormalize the prediction
    denorm_sst = pred_sst_np.copy()
    denorm_sst[~land_mask] = denorm_sst[~land_mask] * (prev_meta['max'] - prev_meta['min']) + prev_meta['min']
    
    # Apply additional smoothing to coastal areas
    smoothed_sst = smooth_coastal_boundaries(denorm_sst, land_mask, n_iterations=3, kernel_size=5)
    
    # Convert land values to NaN for final output
    final_sst = smoothed_sst.copy()
    final_sst[land_mask] = np.nan
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct filename for the output
    output_filename = os.path.join(output_dir, f"sst.{missing_date}_{get_next_date(missing_date)}.nc")
    
    # Create a new NetCDF file using xarray
    lats = prev_meta['lats']
    lons = prev_meta['lons']
    
    # Create output dataset
    ds_out = xr.Dataset(
        data_vars={
            'sst': (('lat', 'lon'), final_sst)
        },
        coords={
            'lat': lats,
            'lon': lons
        }
    )
    
    # Copy attributes from original data
    ds_out.attrs = prev_meta['attributes']
    ds_out['sst'].attrs = prev_meta['sst_attrs']
    
    # Add generation info
    ds_out.attrs['generated_by'] = 'SST GAN Model'
    ds_out.attrs['generation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds_out.attrs['source_files'] = f"{os.path.basename(prev_file)}, {os.path.basename(next_file)}"
    
    # Save to NetCDF
    ds_out.to_netcdf(output_filename)
    
    print(f"Generated SST data saved to {output_filename}")
    return True
# Main function
def main():
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("Warning: CUDA is not available, falling back to CPU")
    
    # List all files in the input directory
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
    
    if args.mode == 'train':
        print(f"Training mode: {len(all_files)} files found in {input_dir}")
        
        # Train the model
        generator = train_gan(
            all_files,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            checkpoint_path=args.checkpoint,
            validation_split=args.validation_split,
            lr=args.lr,
            lambda_l1=args.lambda_l1,
            land_penalty=args.land_penalty,
            coastal_penalty=args.coastal_penalty,
            coastal_buffer=args.coastal_buffer
        )
        
        print("Training completed successfully!")
        
    elif args.mode == 'generate':
        if args.model_path is None:
            print("Error: Model path must be provided for generation mode")
            return
        
        print(f"Generation mode: Using model {args.model_path}")
        print(f"Processing {len(missing_dates)} missing dates")
        
        # Loop through missing dates and generate data
        successful = 0
        failed = 0
        
        for missing_date in missing_dates:
            try:
                if generate_missing_sst(missing_date, all_files, args.model_path, args.output_dir):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error generating data for {missing_date}: {e}")
                failed += 1
        
        print(f"Generation completed: {successful} successful, {failed} failed")
        
    else:
        print(f"Unknown mode: {args.mode}")

# Visualization function for model outputs
def visualize_results(input_sst, output_sst, target_sst=None, land_mask=None, save_path=None):
    """
    Visualize the model input, output, and optionally target data
    """
    plt.figure(figsize=(16, 6))
    
    if target_sst is not None:
        plt.subplot(1, 3, 1)
    else:
        plt.subplot(1, 2, 1)
    
    plt.title("Input SST")
    plt.imshow(input_sst, cmap='coolwarm')
    plt.colorbar(label='SST (°C)')
    
    if target_sst is not None:
        plt.subplot(1, 3, 2)
        plt.title("Generated SST")
        plt.imshow(output_sst, cmap='coolwarm')
        plt.colorbar(label='SST (°C)')
        
        plt.subplot(1, 3, 3)
        plt.title("Target SST")
        plt.imshow(target_sst, cmap='coolwarm')
        plt.colorbar(label='SST (°C)')
    else:
        plt.subplot(1, 2, 2)
        plt.title("Generated SST")
        plt.imshow(output_sst, cmap='coolwarm')
        plt.colorbar(label='SST (°C)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

# Diagnostic function to compare coastal regions
def analyze_coastal_regions(original_sst, generated_sst, land_mask, output_dir=None):
    """
    Analyze temperature differences in coastal regions and create diagnostic plots
    """
    # Create coastal mask
    coastal_mask = create_coastal_mask(land_mask, buffer_size=3)
    
    # Calculate temperature statistics for coastal regions
    coastal_orig = original_sst[coastal_mask & ~land_mask]
    coastal_gen = generated_sst[coastal_mask & ~land_mask]
    
    # Statistics
    mean_diff = np.mean(coastal_gen - coastal_orig)
    std_diff = np.std(coastal_gen - coastal_orig)
    max_diff = np.max(np.abs(coastal_gen - coastal_orig))
    
    print(f"Coastal region statistics:")
    print(f"  Mean difference: {mean_diff:.4f}°C")
    print(f"  Std deviation: {std_diff:.4f}°C")
    print(f"  Max absolute difference: {max_diff:.4f}°C")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Original coastal temps
    plt.subplot(2, 2, 1)
    plt.title("Original SST (Coastal)")
    masked_orig = np.ma.masked_array(original_sst, mask=~(coastal_mask & ~land_mask))
    plt.imshow(masked_orig, cmap='coolwarm')
    plt.colorbar(label='SST (°C)')
    
    # Generated coastal temps
    plt.subplot(2, 2, 2)
    plt.title("Generated SST (Coastal)")
    masked_gen = np.ma.masked_array(generated_sst, mask=~(coastal_mask & ~land_mask))
    plt.imshow(masked_gen, cmap='coolwarm')
    plt.colorbar(label='SST (°C)')
    
    # Difference map
    plt.subplot(2, 2, 3)
    plt.title("Difference (Generated - Original)")
    diff = generated_sst - original_sst
    masked_diff = np.ma.masked_array(diff, mask=~(coastal_mask & ~land_mask))
    im = plt.imshow(masked_diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    plt.colorbar(im, label='Difference (°C)')
    
    # Histogram of differences
    plt.subplot(2, 2, 4)
    plt.title("Histogram of Coastal Temperature Differences")
    plt.hist(coastal_gen - coastal_orig, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Temperature Difference (°C)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "coastal_analysis.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_diff': max_diff
    }

if __name__ == "__main__":
    main()
