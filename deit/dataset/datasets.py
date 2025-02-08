# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import datasets_seeds
import aircraft_seeds
import aircraft
import aircraft_seeds_lt
import aircraft_lt
import inat_seeds
import inat
import birds_seeds
import birds
import breeds_seeds
import breeds_seeds_lt
import breeds
import breeds_lt
import inat21_mini
import inat21_mini_seeds

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets_seeds.ImageFolder(
            root,
            transform=transform,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = 1000
    elif args.data_set == 'AIR':
        if is_train:
            split = 'trainval'
        else:
            split = 'test'

        dataset = aircraft.FGVCAircraft_Hier(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=False,
            category=args.category,
        )
        if args.category == 'name':
            annotation_level = 'variant'
            nb_classes = 100
        elif args.category == 'family':
            annotation_level = 'family'
            nb_classes = 70
        elif args.category == 'order':
            annotation_level = 'manufacturer'
            nb_classes = 30
        else:
            raise ValueError('Invalid category')
        
        # dataset = datasets.FGVCAircraft(root=args.data_path, split=split, 
        #                                 annotation_level=annotation_level,
        #                                 transform=transform)
        
    elif args.data_set == 'AIR-HIER':
        dataset = aircraft.FGVCAircraft_Hier(
            args.data_path,
            is_train=is_train,
            transform=transform,
        )
        nb_classes = [100, 70, 30]

    elif args.data_set == 'AIR-HIER-LT':
        if args.img_max is None:
                args.img_max = 67
        if is_train == True:
            dataset = aircraft_lt.ImbalancedFGVCAircraft(
                args.data_path,
                is_train=is_train,
                transform=transform,
                is_hier=True,
                imb_type=args.imb_type, imb_factor=0.1, rand_number=0,
                img_max=args.img_max,
            )
        else:
            dataset = aircraft.FGVCAircraft_Hier(
                args.data_path,
                is_train=is_train,
                transform=transform,
            )
        nb_classes = [100, 70, 30]

    elif args.data_set == 'AIR-HIER-SUPERPIXEL-LT':
        if args.img_max is None:
                args.img_max = 67
        if is_train == True:
            dataset = aircraft_seeds_lt.ImbalancedFGVCAircraft(
                args.data_path,
                is_train=is_train,
                transform=transform,
                is_hier=True,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                n_segments=args.num_superpixels,
                compactness=10.0,
                blur_ops=None,
                scale_factor=1.0,
                imb_type=args.imb_type, imb_factor=0.1, rand_number=0,
                img_max=args.img_max,
            )

        else:
            dataset = aircraft_seeds.FGVCAircraft(
                args.data_path,
                is_train=is_train,
                transform=transform,
                is_hier=True,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                n_segments=args.num_superpixels,
                compactness=10.0,
                blur_ops=None,
                scale_factor=1.0
            )
        nb_classes = [100, 70, 30]


    elif args.data_set == 'AIR-HIER-SUPERPIXEL':
        dataset = aircraft_seeds.FGVCAircraft(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [100, 70, 30]

    elif args.data_set == 'AIR-SUPERPIXEL':
        dataset = aircraft_seeds.FGVCAircraft(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=False,
            category = args.category,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        if args.category == 'name':
            nb_classes = 100
        elif args.category == 'family':
            nb_classes = 70
        elif args.category == 'order':
            nb_classes = 30
    elif args.data_set == 'BIRD-HIER':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            random_seed=args.random_seed,
            train=is_train,
        )
        nb_classes = [200, 38, 13]

    elif args.data_set == 'BIRD':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds.ImageFolder(
            root,
            transform=transform,
            is_hier=False,
            category = args.category,
        )
        if args.category == 'name':
            nb_classes = 200
        elif args.category == 'family':
            nb_classes = 38
        elif args.category == 'order':
            nb_classes = 13

    elif args.data_set == 'BIRD-HIER-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds_seeds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
            random_seed=args.random_seed,
            train=is_train,
        )
        nb_classes = [200, 38, 13]

    elif args.data_set == 'BIRD-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds_seeds.ImageFolder(
            root,
            transform=transform,
            is_hier=False,
            category = args.category,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        if args.category == 'name':
            nb_classes = 200
        elif args.category == 'family':
            nb_classes = 38
        elif args.category == 'order':
            nb_classes = 13

    elif args.data_set == 'INAT21-MINI-HIER':
        dataset = inat21_mini.iNat21MiniDataset(
            args.data_path,
            transform=transform,
            is_hier=True,
            is_train=is_train,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'INAT21-MINI-HIER-SUPERPIXEL':
        dataset = inat21_mini_seeds.iNat21MiniDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [10000, 1103, 273]


    elif args.data_set == 'INAT18-HIER-SUPERPIXEL':
        dataset = inat_seeds.iNatHierDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [8142, 274, 14]

    elif args.data_set == 'INAT18-SUPERPIXEL':
        dataset = inat_seeds.iNatHierDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=False,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = 8142
    
    elif args.data_set == 'INAT18-HIER':
        dataset = inat.iNatHierDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
        )
        nb_classes = [8142, 274, 14] # To-do: fix -> 272

    elif args.data_set == 'INAT18':
        dataset = inat.iNatHierDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=False,
            category=args.category
        )
        if args.category == 'name':
            nb_classes = 8142
        elif args.category == 'family':
            nb_classes = 274
        elif args.category == 'order':
            nb_classes = 14
    
    elif args.data_set == 'BREEDS-HIER-SUPERPIXEL':
        dataset = breeds_seeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort = args.breeds_sort,
            is_source = args.issource,
            path_yn=args.path_yn,
            sourcefile=args.sourcefile,
        )
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]

    elif args.data_set == 'BREEDS-HIER-SUPERPIXEL-LT':
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
            if args.img_max is None:
                args.img_max = 1053
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
            if args.img_max is None:
                args.img_max = 1300 
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
            if args.img_max is None:
                args.img_max = 1069
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]
            if args.img_max is None:
                args.img_max = 1300

        if is_train == True:
            dataset = breeds_seeds_lt.ImbalancedBreedsDataset(
                args.data_path,
                is_train=is_train,
                transform=transform,
                is_hier=True,
                sort = args.breeds_sort,
                is_source = args.issource,
                path_yn=args.path_yn,
                imb_type=args.imb_type, imb_factor=0.1, rand_number=0,
                cls_num=nb_classes[0], img_max=args.img_max,
            )
        else:
            dataset = breeds_seeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort = args.breeds_sort,
            is_source = args.issource,
            path_yn=args.path_yn,
            sourcefile=args.sourcefile,
        )
        


    elif args.data_set == 'BREEDS-SUPERPIXEL':
        dataset = breeds_seeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=False,
            sort = args.breeds_sort,
            category=args.category
        )
        if args.breeds_sort == 'entity13':
            if args.category == 'name':
                nb_classes = 130
            else:
                nb_classes = 13
        elif args.breeds_sort == 'living17':
            if args.category == 'name':
                nb_classes = 34
            else:
                nb_classes = 17
        elif args.breeds_sort == 'nonliving26':
            if args.category == 'name':
                nb_classes = 52
            else:
                nb_classes = 26
        elif args.breeds_sort == 'entity30':
            if args.category == 'name':
                nb_classes = 120
            else:
                nb_classes = 30


    elif args.data_set == 'BREEDS-HIER':
        dataset = breeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort = args.breeds_sort,
            is_source = args.issource,
            path_yn=args.path_yn,
            sourcefile=args.sourcefile,
        )
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]

    elif args.data_set == 'BREEDS-HIER-LT':
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
            if args.img_max is None:
                args.img_max = 1053
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
            if args.img_max is None:
                args.img_max = 1300 
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
            if args.img_max is None:
                args.img_max = 1069
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]
            if args.img_max is None:
                args.img_max = 1300

        if is_train == True:
            dataset = breeds_lt.ImbalancedBreedsDataset(
                args.data_path,
                is_train=is_train,
                transform=transform,
                is_hier=True,
                sort = args.breeds_sort,
                is_source = args.issource,
                path_yn=args.path_yn,
                imb_type=args.imb_type, imb_factor=0.1, rand_number=0,
                cls_num=nb_classes[0], img_max=args.img_max,
            )
        else:
            dataset = breeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort = args.breeds_sort,
            is_source = args.issource,
            path_yn=args.path_yn,
        )
            
    elif args.data_set == 'BREEDS':
        dataset = breeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=False,
            sort = args.breeds_sort,
            category=args.category
        )
        if args.breeds_sort == 'entity13':
            if args.category == 'name':
                nb_classes = 130
            else:
                nb_classes = 13
        elif args.breeds_sort == 'living17':
            if args.category == 'name':
                nb_classes = 34
            else:
                nb_classes = 17
        elif args.breeds_sort == 'nonliving26':
            if args.category == 'name':
                nb_classes = 52
            else:
                nb_classes = 26
        elif args.breeds_sort == 'entity30':
            if args.category == 'name':
                nb_classes = 120
            else:
                nb_classes = 30

    # elif args.data_set == 'INAT':
    #     dataset = INatDataset(args.data_path, train=is_train, year=2018,
    #                           category=args.inat_category, transform=transform)
    #     nb_classes = dataset.nb_classes

    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if 'INAT' in args.data_set:
        t.append(transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
