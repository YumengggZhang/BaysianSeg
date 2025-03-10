from .dataset import build_Prostate, build_ImageCAS, build_ImageCAS_3d


def build_dataset(image_set, args):
    if args.dataset == "Prostate":
        return build_Prostate(image_set, args)
    elif args.dataset == "ImageCAS":
        return build_ImageCAS(image_set, args)
    elif args.dataset == 'ImageCAS3d':
        return build_ImageCAS_3d(image_set, args)

    raise ValueError(f"dataset {args.dataset} not supported")

