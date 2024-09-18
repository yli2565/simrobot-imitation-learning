curDir="$PWD"
cd BadgerRLSystem/Util/TypedShmem
python -m build -w --verbose
pip install dist/typedshmem-0.0.1-cp39-cp39-linux_x86_64.whl --no-clean --force-reinstall
cd "$curDir"

