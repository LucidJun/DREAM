
from setuptools import setup, find_packages
import os

# Function to read the list of dependencies from requirements.txt
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return file.read().splitlines()



 
setup(
    name='dreams-mc',
    version='0.0.3',
    author="LucidMoon",
    packages=find_packages(),
    description="Library for DL models reporting. dreams_mc is a Python package designed to simplify the process of creating comprehensive model cards for deep learning models. Model cards(mc) are crucial for documenting the performance, usage, and limitations of AI models, promoting transparency and ethical use of AI.",
    url="https://github.com/LucidJun/DREAM",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    package_data={'dreams_mc': ['assets/css/*.css', 'assets/js/*.js', 
                            'assets/vendor/bootstrap/css/bootstrap.min.css',
                            'assets/vendor/font-awesome/css/all.min.css',
                            'assets/vendor/magnific-popup/magnific-popup.min.css',
                            'assets/vendor/highlight.js/styles/github.css',
                            'assets/vendor/jquery/jquery.min.js',
                            'assets/vendor/bootstrap/js/bootstrap.bundle.min.js'
                            'assets/vendor/highlight.js/highlight.min.js',
                            'assets/vendor/jquery.easing/jquery.easing.min.js',
                            'assets/vendor/magnific-popup/jquery.magnific-popup.min.js',

                            ] },
    include_package_data=True,
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'dreams-mc=dreams_mc.make_model_card:generate_modelcard',  # Creates a command-line script
        ],
    },
)
