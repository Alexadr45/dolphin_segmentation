class InferenceConfig:
    def __init__(self):
        # paths
	directory_or_image_path = '../test_image.jpg' # Путь до файла или директории
	output_directory = '../output_images'
	model_path = '../unet_dolphin_seg.pt'
        
        # model params
        self.model = 'UnetPlusPlus'
        self.encoder_name="efficientnet-b4"
	self.alpha = 0.25
	self.thresshold = 0.5
	self.directory = False # Если указан путь для директории, установить True, если до изображения - False.
​
        # for resize images and masks
        self.shape = [256, 256]
​
        # albumentation.Normallize
        self.normallize_mean = [0.485, 0.456, 0.406]
        self.normallize_std  = [0.229, 0.224, 0.225]
