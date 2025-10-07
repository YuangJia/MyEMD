import torch
from scene import Scene, EUVSScene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, vis_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, opt, background,x_offset=None):
    if x_offset is not None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_r{x_offset}")
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    if "scene" in model_path:
        last = ".png"
    else:
        last = ".jpg"
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        renders = render(opt, view, gaussians, background, return_dx=True, iter=iteration, is_train=False)
        rendering = renders["render"]
        img_name = view.image_name
        torchvision.utils.save_image(rendering, os.path.join(render_path, img_name + last))


def render_sets(dataset: ModelParams, opt: OptimizationParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, opt)
        scene = EUVSScene(dataset, gaussians, load_iteration=iteration, shuffle=False, offset=True, x_offset=1.0)

        # gaussians._scaling[:, 0] = 0.001
        # gaussians._scaling[:, 1] = 0.0005
        # gaussians._scaling[:, 2] = -10000.0
        # gaussians._rotation[:, 0] = 1
        # gaussians._rotation[:, 1:] = 0
        scales = gaussians.get_scaling

        # min_scale, _ = torch.min(scales, dim=1)
        # max_scale, _ = torch.max(scales, dim=1)
        # median_scale, _ = torch.median(scales, dim=1)
        # print(min_scale)
        # print(max_scale)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, opt,
                       background,x_offset=1.0)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, opt,
                       background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), op.extract(args) ,args.iteration, pipeline.extract(args), skip_train=False, skip_test=True)

    # render_sets(model.extract(args), op.extract(args) ,25000, pipeline.extract(args), skip_train=False, skip_test=True)
    # render_sets(model.extract(args), op.extract(args) ,35000, pipeline.extract(args), skip_train=False, skip_test=True)
    # render_sets(model.extract(args), op.extract(args) ,47000, pipeline.extract(args), skip_train=False, skip_test=True)
    # render_sets(model.extract(args), op.extract(args) ,60000, pipeline.extract(args), skip_train=False, skip_test=True)
    # render_sets(model.extract(args), op.extract(args) ,75000, pipeline.extract(args), skip_train=False, skip_test=True)
    render_sets(model.extract(args), op.extract(args), 35000, pipeline.extract(args), skip_train=False, skip_test=False)