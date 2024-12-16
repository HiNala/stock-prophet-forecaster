"""
UI styles and theme configuration
"""

from ..utils.config import get_config

config = get_config()

# Color schemes
DARK_THEME = {
    'bg_color': '#2b2b2b',
    'fg_color': '#ffffff',
    'button_color': '#3d3d3d',
    'button_hover_color': '#4d4d4d',
    'entry_color': '#3d3d3d',
    'border_color': '#1a1a1a',
    'success_color': '#28a745',
    'warning_color': '#ffc107',
    'error_color': '#dc3545',
    'primary_color': '#007bff',
    'secondary_color': '#6c757d',
    'tooltip_bg': '#4d4d4d',
    'tooltip_fg': '#ffffff',
    'debug_bg': '#1e1e1e',
    'debug_fg': '#d4d4d4'
}

LIGHT_THEME = {
    'bg_color': '#f8f9fa',
    'fg_color': '#212529',
    'button_color': '#e9ecef',
    'button_hover_color': '#dee2e6',
    'entry_color': '#ffffff',
    'border_color': '#dee2e6',
    'success_color': '#28a745',
    'warning_color': '#ffc107',
    'error_color': '#dc3545',
    'primary_color': '#007bff',
    'secondary_color': '#6c757d',
    'tooltip_bg': '#f8f9fa',
    'tooltip_fg': '#212529',
    'debug_bg': '#ffffff',
    'debug_fg': '#212529'
}

# Get active theme
ACTIVE_THEME = DARK_THEME if config.get('ui', 'theme') == 'dark' else LIGHT_THEME

# Font configurations
FONTS = {
    'default': ('Segoe UI', 10),
    'header': ('Segoe UI', 12, 'bold'),
    'title': ('Segoe UI', 14, 'bold'),
    'small': ('Segoe UI', 8),
    'monospace': ('Consolas', 10)
}

# Widget styles
BUTTON_STYLE = {
    'fg_color': ACTIVE_THEME['button_color'],
    'hover_color': ACTIVE_THEME['button_hover_color'],
    'text_color': ACTIVE_THEME['fg_color'],
    'border_width': 0,
    'corner_radius': 6
}

ENTRY_STYLE = {
    'fg_color': ACTIVE_THEME['entry_color'],
    'text_color': ACTIVE_THEME['fg_color'],
    'border_width': 1,
    'border_color': ACTIVE_THEME['border_color'],
    'corner_radius': 6
}

FRAME_STYLE = {
    'fg_color': ACTIVE_THEME['bg_color'],
    'border_width': 1,
    'border_color': ACTIVE_THEME['border_color'],
    'corner_radius': 6
}

# Chart styles
CHART_STYLE = {
    'template': 'plotly_dark' if config.get('ui', 'theme') == 'dark' else 'plotly_white',
    'paper_bgcolor': ACTIVE_THEME['bg_color'],
    'plot_bgcolor': ACTIVE_THEME['bg_color'],
    'font_color': ACTIVE_THEME['fg_color']
}

# Layout configurations
PADDINGS = {
    'small': 5,
    'medium': 10,
    'large': 20
}

MARGINS = {
    'small': 5,
    'medium': 10,
    'large': 20
}

# Status colors
STATUS_COLORS = {
    'success': ACTIVE_THEME['success_color'],
    'warning': ACTIVE_THEME['warning_color'],
    'error': ACTIVE_THEME['error_color']
}

def apply_theme_to_widget(widget, widget_type='frame'):
    """Apply theme styles to a widget"""
    if widget_type == 'button':
        for key, value in BUTTON_STYLE.items():
            try:
                widget.configure(**{key: value})
            except:
                pass
    elif widget_type == 'entry':
        for key, value in ENTRY_STYLE.items():
            try:
                widget.configure(**{key: value})
            except:
                pass
    elif widget_type == 'frame':
        for key, value in FRAME_STYLE.items():
            try:
                widget.configure(**{key: value})
            except:
                pass

def get_chart_style():
    """Get the current chart style configuration"""
    return CHART_STYLE.copy()

def get_status_color(status):
    """Get color for status type"""
    return STATUS_COLORS.get(status, ACTIVE_THEME['fg_color']) 