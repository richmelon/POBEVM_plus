"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""


DATA_PATHS = {
    
    'videomatte': {
        'train': '/home/xjm/code/RVM/matting-data/VideoMatte240K_JPEG_SD/train',
        'valid': '/home/xjm/code/RVM/matting-data/VideoMatte240K_JPEG_SD/valid',
    },
    'imagematte': {
        'train': '/home/xjm/code/RVM/matting-data/ImageMatte/train',
        'valid': '/home/xjm/code/RVM/matting-data/ImageMatte/valid',
    },
    'background_images': {
        'train': '/home/xjm/code/RVM/matting-data/background_image/train',
        'valid': '/home/xjm/code/RVM/matting-data/background_image/valid',
    },
    'background_videos': {
        'train': '/home/xjm/code/RVM/matting-data/background_video/train',
        'valid': '/home/xjm/code/RVM/matting-data/background_video/valid',
    },
    
    
    'coco_panoptic': {
        'imgdir': '/home/xjm/code/RVM/matting-data/coco/train2017/',
        'anndir': '/home/xjm/code/RVM/matting-data/coco/pano_img/panoptic_train2017/',
        'annfile': '/home/xjm/code/RVM/matting-data/coco/pano_img/panoptic_train2017.json',
    },
    'spd': {
        'imgdir': '/home/xjm/code/RVM/matting-data/SuperviselyPersonDataset/img',
        'segdir': '/home/xjm/code/RVM/matting-data/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir': '/home/xjm/code/RVM/matting-data/youtubuvis/JPEGImages',
        'annfile': '/home/xjm/code/RVM/matting-data/youtubuvis/instances.json',
    }
    
}
