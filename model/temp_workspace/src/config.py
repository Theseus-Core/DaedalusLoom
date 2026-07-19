# -*- coding: utf-8 -*-
# Configuration and metadata definitions shared between training and inference

LABEL_MAP = {
    'draw_o': 0,


    'wave_hand': 1,
    
    'stand_up': 2,
    'draw_x': 3,
    
}

# English label inverse mapping (for training scripts and dataset loading)
INV_LABEL_MAP_EN = {v: k for k, v in LABEL_MAP.items()}

# Chinese label inverse mapping (for GUI tools and user-facing displays)
INV_LABEL_MAP_CN = {
    0: '画o',
    
    1: '挥手',
    2: '起身',
    3: '画x',
  
  
 
   
}
    


# A list of 10 premium colors for UI displays
COLOR_PALETTE = [
    "#2ecc71",  # Emerald Green
    "#3498db",  # Peter River Blue
    "#9b59b6",  # Amethyst Purple
    "#e67e22",  # Carrot Orange
    "#1abc9c",  # Turquoise
    "#e74c3c",  # Alizarin Red
    "#f1c40f",  # Sun Flower Yellow
    "#e84393",  # Pink / Rose
    "#00cec9",  # Teal
    "#6c5ce7"   # Lavender
]

def get_color_by_index(index):
    """
    Get a hex color from the palette based on class index (supports wrap-around modulo 10).
    """
    try:
        idx = int(index)
        return COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    except (ValueError, TypeError):
        return "#ffffff"
