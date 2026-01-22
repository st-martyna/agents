"""
Programmatic test image generation for the deliberative experiment.

Generates simple scene images that match the scenarios for VLM testing.
These are stylized/schematic images, not realistic renders.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


def create_scene_image(
    scenario_name: str,
    objects: Dict[str, Dict[str, Any]],
    size: Tuple[int, int] = (512, 512),
    background_color: Tuple[int, int, int] = (240, 240, 240)
) -> np.ndarray:
    """
    Create a simple schematic image for a scenario.

    Args:
        scenario_name: Name of the scenario
        objects: Objects in the scenario
        size: Image dimensions
        background_color: Background RGB color

    Returns:
        RGB image as numpy array
    """
    img = Image.new('RGB', size, background_color)
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()
        small_font = font

    # Draw title
    draw.text((10, 10), f"Scenario: {scenario_name}", fill=(0, 0, 0), font=font)

    # Draw grid
    for i in range(0, size[0], 50):
        draw.line([(i, 0), (i, size[1])], fill=(220, 220, 220), width=1)
        draw.line([(0, i), (size[0], i)], fill=(220, 220, 220), width=1)

    # Position objects
    obj_positions = {}
    y_offset = 80

    for obj_id, props in objects.items():
        obj_type = props.get("type", "unknown")

        # Determine position based on object type and scene position
        if "position" in props:
            pos = props["position"]
            # Map from scene coordinates to image coordinates
            x = int((pos[0] + 1) / 2 * (size[0] - 100) + 50)
            y = int((1 - pos[1]) / 2 * (size[1] - 150) + 100)
        else:
            x = size[0] // 2
            y = y_offset
            y_offset += 100

        obj_positions[obj_id] = (x, y)

        # Draw object based on type
        if obj_type == "light_source":
            draw_lamp(draw, x, y, props, font, small_font)
        elif obj_type == "pill":
            draw_pill(draw, x, y, props, font, small_font)
        elif obj_type == "bin":
            draw_bin(draw, x, y, props, font, small_font)
        elif obj_type == "surface":
            draw_surface(draw, x, y, props, font, small_font)
        else:
            draw_generic(draw, x, y, obj_id, props, font, small_font)

    return np.array(img)


def draw_lamp(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    props: Dict[str, Any],
    font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont
):
    """Draw a lamp icon."""
    confidence = props.get("confidence", 0.5)
    brightness = props.get("brightness", 0.5)
    reachable = props.get("reachable", True)

    # Color based on reachability
    fill_color = (255, 255, 100) if reachable else (200, 200, 100)
    outline_color = (200, 150, 0) if reachable else (150, 150, 150)

    # Draw lamp base
    draw.rectangle([x-15, y+10, x+15, y+30], fill=(100, 100, 100), outline=(50, 50, 50))

    # Draw lamp shade (triangle)
    draw.polygon([(x, y-30), (x-25, y+10), (x+25, y+10)], fill=fill_color, outline=outline_color)

    # Draw light rays if bright
    if brightness > 0.5:
        for angle in [-30, 0, 30]:
            import math
            rad = math.radians(angle - 90)
            x2 = x + int(40 * math.cos(rad))
            y2 = y - 30 + int(40 * math.sin(rad))
            draw.line([(x, y-30), (x2, y2)], fill=(255, 255, 200), width=2)

    # Label
    label = f"lamp ({confidence:.0%})"
    if not reachable:
        label += " [unreachable]"
    draw.text((x-30, y+35), label, fill=(0, 0, 0), font=small_font)


def draw_pill(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    props: Dict[str, Any],
    font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont
):
    """Draw a pill icon."""
    color_dist = props.get("color_distribution", {"pink": 0.5, "red": 0.5})
    confidence = props.get("confidence", 0.5)
    grasped = props.get("grasped", False)

    # Determine color based on distribution
    pink_ratio = color_dist.get("pink", 0.5)
    if pink_ratio > 0.5:
        fill_color = (255, 182, 193)  # Pink
    else:
        fill_color = (220, 80, 80)  # Red

    # Draw pill (ellipse)
    draw.ellipse([x-20, y-10, x+20, y+10], fill=fill_color, outline=(100, 50, 50), width=2)

    # Add "?" to indicate uncertainty
    draw.text((x-5, y-8), "?", fill=(100, 50, 50), font=font)

    # Grasped indicator
    if grasped:
        draw.rectangle([x-25, y+12, x+25, y+20], fill=(150, 150, 150))
        draw.text((x-20, y+12), "HELD", fill=(50, 50, 50), font=small_font)

    # Label
    label = f"pill ({confidence:.0%})"
    draw.text((x-25, y+25), label, fill=(0, 0, 0), font=small_font)


def draw_bin(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    props: Dict[str, Any],
    font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont
):
    """Draw a bin icon."""
    label_text = props.get("label", "?")
    confidence = props.get("confidence", 0.5)

    # Color based on label
    if label_text == "pink":
        fill_color = (255, 200, 210)
    elif label_text == "red":
        fill_color = (255, 180, 180)
    else:
        fill_color = (200, 200, 200)

    # Draw bin (rectangle with open top)
    draw.rectangle([x-25, y-15, x+25, y+25], fill=fill_color, outline=(100, 100, 100), width=2)

    # Draw label
    draw.text((x-10, y-5), label_text.upper(), fill=(50, 50, 50), font=font)

    # Confidence label
    draw.text((x-20, y+30), f"bin ({confidence:.0%})", fill=(0, 0, 0), font=small_font)


def draw_surface(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    props: Dict[str, Any],
    font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont
):
    """Draw a surface (table) icon."""
    confidence = props.get("confidence", 0.5)

    # Draw table top
    draw.rectangle([x-50, y-5, x+50, y+5], fill=(139, 90, 43), outline=(100, 60, 30), width=2)

    # Draw legs
    draw.rectangle([x-45, y+5, x-35, y+30], fill=(120, 70, 35))
    draw.rectangle([x+35, y+5, x+45, y+30], fill=(120, 70, 35))

    # Label
    draw.text((x-25, y+35), f"table ({confidence:.0%})", fill=(0, 0, 0), font=small_font)


def draw_generic(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    obj_id: str,
    props: Dict[str, Any],
    font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont
):
    """Draw a generic object icon."""
    confidence = props.get("confidence", 0.5)

    # Draw generic box
    draw.rectangle([x-20, y-20, x+20, y+20], fill=(180, 180, 180), outline=(100, 100, 100), width=2)

    # Draw object type
    obj_type = props.get("type", "?")[:4]
    draw.text((x-10, y-8), obj_type, fill=(50, 50, 50), font=small_font)

    # Label
    draw.text((x-25, y+25), f"{obj_id} ({confidence:.0%})", fill=(0, 0, 0), font=small_font)


def generate_scenario_images(
    scenarios: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Generate images for all scenarios.

    Args:
        scenarios: Dictionary of scenario definitions
        output_dir: Optional directory to save images

    Returns:
        Dictionary mapping scenario names to images
    """
    import os

    images = {}

    for name, scenario in scenarios.items():
        if hasattr(scenario, 'objects'):
            objects = scenario.objects
        else:
            objects = scenario.get('objects', {})

        img = create_scene_image(name, objects)
        images[name] = img

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(img).save(os.path.join(output_dir, f"{name}.png"))

    return images


def create_dummy_image(size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create a simple dummy image for testing."""
    img = Image.new('RGB', size, (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((size[0]//2 - 50, size[1]//2), "Test Image", fill=(100, 100, 100))
    return np.array(img)
