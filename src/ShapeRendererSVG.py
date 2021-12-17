import torch
from torch import nn
from utils import unnormalize_functional
import svgwrite
from ShapeRenderer import ShapeRenderer
import tempfile
from PIL import Image
from cairosvg import svg2png
import torchvision.transforms as TF


class ShapeRendererSVG():
    def __init__(self, canvas_size, mean, std):
        self.canvas_size = canvas_size
        self.mean = mean
        self.std = std
        self.internal_renderer = ShapeRenderer(device="cpu").eval()

    def svg_color(self, r, g, b):
        rgb = torch.tensor([r, g, b]).view(-1, 1, 1)
        rgb = 255*unnormalize_functional(rgb, self.mean, self.std)
        return svgwrite.rgb(*rgb)

    def get_string(self, shapes_args):
        shapes_args = self.internal_renderer.process_shape_arguments(shapes_args)
        dwg = svgwrite.Drawing(profile='tiny', size=(self.canvas_size, self.canvas_size))

        # Add background
        background_color = self.svg_color(0, 0, 0)   # Black in domain colors
        dwg.add(dwg.rect(insert=(0, 0), size=(self.canvas_size, self.canvas_size), fill=background_color))

        # Add shapes
        for shape_args in shapes_args:
            _, _, pos_y, pos_x, height, width, angle, squareness, r, g, b = shape_args.numpy()
            # Move/rescale shapes according to canvas size
            # Uses a single canvas_size (square) since width/height is ambiguous for roateted shapes.
            # Non-square canvas sizes could maybe be implemented with other transforms (translate, rescale).
            pos_x *= self.canvas_size
            pos_y *= self.canvas_size
            height *= self.canvas_size
            width *= self.canvas_size
            angle = (180/torch.pi)*angle
            color = self.svg_color(r, g, b)
            if squareness > 0.5:
                shape = dwg.rect(insert=(pos_x-width, pos_y-height), size=(2*width, 2*height), fill=color)
            else:
                shape = dwg.ellipse(center=(pos_x, pos_y), r=(width, height), fill=color)
            shape.rotate(angle=angle, center=(pos_x, pos_y))
            dwg.add(shape)
        return dwg.tostring()

    def save_svg(self, shapes_args, file_path):
        svg_string = self.get_string(shapes_args)
        with open(file_path, "w") as svg_file:
            svg_file.write(svg_string)

    def save_png(self, shapes_args, png_path):
        with tempfile.NamedTemporaryFile(suffix=".svg") as svg_path:
            self.save_svg(shapes_args, svg_path.name)
            with open(svg_path.name, mode="rb") as svg_file:
                svg2png(bytestring=svg_file.read(), write_to=str(png_path))
            svg_path.unlink()

    def to_pil_image(self, shape_args):
        with tempfile.NamedTemporaryFile(suffix=".png") as png_path:
            self.save_png(shape_args, png_path.name)
            image = Image.open(png_path)
        return image