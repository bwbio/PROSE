import setuptools

setuptools.setup(
    name="pyprose",
    version="1.0.0",
    url="https://github.com/bwbio/PROSE",
    author="Bertrand Wong",
    author_email="bwjh98@gmail.com",
    description="PROSE uses a simple list of high-/low-confidence proteins in a proteomic screen to generate enrichment scores for individual proteins in the proteome.",
    long_description='View the project details at: https://github.com/bwbio/PROSE/',
    packages=setuptools.find_packages(),
    
    include_package_data=True,
    package_data={'':['vignette/*']},
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'sklearn',
                      'matplotlib',
                      'seaborn',
                      ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)