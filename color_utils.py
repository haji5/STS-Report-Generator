"""
Color palette utilities for chart generation.
This module provides functions for validating, parsing, and managing color palettes.
"""

import re


def validate_hex_color(color_string):
    """
    Validate a single hex color code.
    Returns True if valid, False otherwise.
    """
    color_string = color_string.strip()
    # Check if it matches the pattern #RRGGBB
    pattern = r'^#[0-9A-Fa-f]{6}$'
    return bool(re.match(pattern, color_string))


def parse_color_palette(palette_string):
    """
    Parse a comma-separated string of hex color codes.
    Returns a list of valid hex colors or None if any are invalid.
    """
    if not palette_string or not palette_string.strip():
        return None

    colors = [color.strip() for color in palette_string.split(',')]
    valid_colors = []

    for color in colors:
        if color:  # Skip empty strings
            if validate_hex_color(color):
                valid_colors.append(color)
            else:
                return None  # Return None if any color is invalid

    return valid_colors if valid_colors else None


def get_color_palette(custom_palette=None, num_colors=None):
    """
    Get a color palette, either custom or default.
    If custom_palette is provided and valid, use it (cycling if needed).
    Otherwise, return default colors.

    Args:
        custom_palette: List of hex color codes or None
        num_colors: Number of colors needed (optional)

    Returns:
        List of hex color codes
    """
    # Default color palette - warm, earthy theme
    default_colors = [
        '#264653',  # Dark teal
        '#2a9d8f',  # Teal
        '#e9c46a',  # Golden yellow
        '#f4a261',  # Orange
        '#e76f51',  # Coral red
        '#8b5a3c',  # Brown
        '#6b8e23',  # Olive green
        '#cd853f',  # Peru
        '#b8860b',  # Dark goldenrod
        '#d2691e'   # Chocolate
    ]

    if custom_palette and len(custom_palette) > 0:
        colors = custom_palette
    else:
        colors = default_colors

    # If num_colors is specified, cycle the palette to get that many colors
    if num_colors:
        if len(colors) >= num_colors:
            return colors[:num_colors]
        else:
            # Cycle the palette to get enough colors
            cycled_colors = []
            for i in range(num_colors):
                cycled_colors.append(colors[i % len(colors)])
            return cycled_colors

    return colors
