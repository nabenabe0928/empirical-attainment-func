rm -r build/ dist/ eaf.egg-info/

pip install wheel twine
python setup.py bdist_wheel
twine upload --repository pypi dist/*