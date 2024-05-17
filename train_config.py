class ModelConfig:
    def __init__(self):
        # paths
        self.image_path = '../data/images' # Путь к каталогу с изображениями
        self.train_directory = '../data/train_dir' # Путь к каталогу с тренировочной выборкой масок
        self.val_directory = '../data/val_dir' # Путь к каталогу с тестовой выборкой масок
        self.tb_log_path = 'tb_log' # Куда сохранять Tensorboard logs
        self.model_save_path = 'best_model.pt' # Куда сохранять модель
        
        # model params
        self.model = 'UnetPlusPlus'
        self.encoder_name='mobilenet_v2'
        self.num_epochs = 10
        self.batch_size = 16
        self.learning_rate = 0.001
​
        # for crop image and mask
        self.pad_if_needed_min_height = 1024
        self.pad_if_needed_min_width = 1024
        self.random_crop_height = 1024
        self.random_crop_width = 1024
​
        # for resize images and masks
        self.shape = [256, 256]
​
        # albumentation.Normallize
        self.normallize_mean = [0.485, 0.456, 0.406]
        self.normallize_std  = [0.229, 0.224, 0.225]
        
        # albumentations.ShiftScaleRotate
        self.shift_scale_rotate_shift_limit = 0.2
        self.shift_scale_rotate_scale_limit = 0.2
        self.shift_scale_rotate_rotate_limit = 30
        self.shift_scale_rotate_p = 0.4
​
        # albumentations.RGBShift
        self.rgb_shift_r_shift_limit = 25
        self.rgb_shift_g_shift_limit = 25
        self.rgb_shift_b_shift_limit = 25
        self.rgb_shift_p = 0.4
​
        # albumentations.RandomBrightnessContrast
        self.random_brightness_contrast_brightness_limit = 0.3
        self.random_brightness_contrast_contrast_limit = 0.3
        self.random_brightness_contrast_p = 0.4
        
        # albumentations.Blur
        self.blur_limit_min = 3
        self.blur_limit_max = 7
        self.blur_p = 0.3
        
        # albumentations.ToGray
        self.togray_p = 0.1
        
        # albumentations.Spatter
        self.spatter_mean_min = 0.61
        self.spatter_mean_max = 0.63
        self.spatter_std_min = 0.29
        self.spatter_std_max = 0.31
        self.spatter_gauss_sigma_min = 1.5
        self.spatter_gauss_sigma_max = 2
        self.spatter_cutout_threshold_min = 0.68
        self.spatter_cutout_threshold_max = 0.68
        self.spatter_intensity_min = 0.63
        self.spatter_intensity_max = 0.63
        self.spatter_mode = 'rain'
        self.spatter_p = 0.2
        
        # albumentations.RandomSunFlare
        self.flare_roi_x_min = 0
        self.flare_roi_y_min = 0
        self.flare_roi_x_max = 1
        self.flare_roi_y_max = 0.1
        self.angle_lower = 0
        self.angle_upper = 1
        self.num_flare_circles_lower = 20
        self.num_flare_circles_upper = 30
        self.src_radius = 180
        self.random_sun_flare_p = 0.1
​
        # train_test_split
        self.train_part = 0.8
        self.valid_part = 0.2
        
    def __str__(self):
        return (
            f"Model: {self.model}\n"
            f"Encoder: {self.encoder_name}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Batch size: {self.batch_size}\n"
            f"Number of epochs: {self.num_epochs}\n"
        )
