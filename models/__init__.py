from .BayeSeg import build as build_BayeSeg
from .BayeSeg3D import build as build_BayeSeg_3d
from .BayeSeg3D_norm import build as build_BayeSeg_3d_norm



def build_model(args):
    if args.model == "BayeSeg":
        return build_BayeSeg(args)
    elif args.model == 'BayeSeg3d':
        return build_BayeSeg_3d(args)
    elif args.model == 'BayeSeg3d_norm':
        return build_BayeSeg_3d_norm(args)
    else:
        raise ValueError("invalid model:{}".format(args.model))
