from PIL import Image
import numpy as np

# Load the image
img = Image.open(r'logo.png').convert('RGBA')
img_array = np.array(img)

# Make white background transparent
# Check if pixels are close to white (RGB > 250)
white_pixels = np.all(img_array[:,:,:3] > 250, axis=2)
img_array[:,:,3] = np.where(white_pixels, 0, 255)

# Save the transparent version
Image.fromarray(img_array).save('logo_transparent.png')
print("Transparent logo created successfully!")
