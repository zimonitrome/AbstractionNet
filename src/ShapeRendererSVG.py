import tempfile
from PIL import Image
from einops.einops import rearrange
import torch
from src.utils import unnormalize_functional
from cairosvg import svg2png
import svgwrite
from pathlib import Path


class ShapeRendererSVG():
    def __init__(self, internal_renderer, canvas_size, mean, std):
        self.canvas_size = canvas_size
        self.mean = mean
        self.std = std
        self.internal_renderer = internal_renderer

    def svg_color(self, r, g, b):
        rgb = torch.tensor([r, g, b]).view(-1, 1, 1)
        rgb = 255*unnormalize_functional(rgb, self.mean, self.std)
        return svgwrite.rgb(*rgb)

    def get_string(self, shapes_args):
        assert len(shapes_args.shape) == 2, "Shape args should be a single sample without batch."
        processed_shapes_args = self.internal_renderer.process_shape_arguments(shapes_args.unsqueeze(0)).squeeze()
        processed_shapes_args = processed_shapes_args.cpu().detach().numpy()

        dwg = svgwrite.Drawing(profile='tiny', size=(self.canvas_size, self.canvas_size))

        # Add background
        background_color = self.svg_color(0, 0, 0)   # Black in domain colors
        dwg.add(dwg.rect(insert=(0, 0), size=(self.canvas_size, self.canvas_size), fill=background_color))

        # Add shapes
        for shape_args in processed_shapes_args:
            _, _, pos_y, pos_x, height, width, angle, squareness, r, g, b = shape_args
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
        svg_file = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        svg_path = Path(svg_file.name)
        svg_file.close()

        self.save_svg(shapes_args, svg_path)
        with open(svg_path, mode="rb") as svg_file:
            svg2png(bytestring=svg_file.read(), write_to=str(png_path))
        svg_path.unlink()

    def to_pil_image(self, shape_args):
        with tempfile.NamedTemporaryFile(suffix=".png") as png_path:
            self.save_png(shape_args, png_path.name)
            image = Image.open(png_path)
        return image