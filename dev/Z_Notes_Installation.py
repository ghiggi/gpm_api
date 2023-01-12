# Create distribution and upload to testpypi
micromamba activate gpm_api 
python setup.py sdist bdist_wheel
twine upload --repository testpypi dist/*
micromamba deactivate

# Test installation
micromamba create --name gpm_api_dev python=3.9
micromamba activate gpm_api_dev
micromamba install curl cartopy>=0.20.0
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ gpm-api==0.0.1 

# Remove environment 
micromamba deactivate
micromamba env remove --name gpm_api_dev

# Final push to pip 
twine check dist/* 
twine upload dist/*

# Retry instrallation
micromamba create --name gpm_api_dev python=3.9
micromamba activate gpm_api_dev
micromamba install cartopy
pip install gpm_api
