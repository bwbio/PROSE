import setuptools

setuptools.setup(
    name="pyprose",
    version="0.1.0",
    url="https://github.com/bwbio/PROSE",
    author="Bertrand Wong",
    author_email="bwjh98@gmail.com",
    description="PROSE takes a list of observed and unobserved proteins, and generates an enrichment score for individual elements in the entire proteome.",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    
    include_package_data=True,
    package_data={'vignette':['vignette/HeLa_DDA_sample.pkl',
                              'klijn_panel_spearman.csv.gz']}  ,
    install_requires=[],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)