#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if glxinfo is available
if ! command_exists glxinfo; then
    echo "Error: glxinfo is not installed. Please install mesa-utils package."
    exit 1
fi

# Check if vglrun is available
if ! command_exists vglrun; then
    echo "Error: vglrun is not installed. Please install VirtualGL package."
    exit 1
fi

# Function to get the OpenGL renderer
get_renderer() {
    $1 glxinfo | grep "OpenGL renderer string" | cut -d':' -f2 | xargs
}

# Check current renderer
current_renderer=$(get_renderer)
echo "Current renderer: $current_renderer"

# Check renderer with VirtualGL
vgl_renderer=$(get_renderer vglrun)
echo "Renderer with VirtualGL: $vgl_renderer"

# Function to check if renderer is software
is_software_renderer() {
    [[ $1 == *"llvmpipe"* ]] || [[ $1 == *"softpipe"* ]] || [[ $1 == *"swrast"* ]]
}

if is_software_renderer "$current_renderer"; then
    echo "Current rendering is software-based."
    
    if ! is_software_renderer "$vgl_renderer"; then
        echo "VirtualGL successfully enabled hardware acceleration!"
        echo "You should use 'vglrun' to launch your 3D applications for hardware acceleration."
    else
        echo "WARNING: VirtualGL could not enable hardware acceleration."
        echo "The application may run slowly."
    fi
else
    if [[ "$current_renderer" != "$vgl_renderer" ]]; then
        echo "Hardware acceleration is already enabled, but VirtualGL is connecting to a different GPU."
        echo "You may want to use 'vglrun' to ensure consistent GPU usage across applications."
    else
        echo "Hardware acceleration is already enabled."
        echo "VirtualGL is not necessary in this case, but using it won't hurt."
    fi
fi