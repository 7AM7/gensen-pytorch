from setuptools import setup, find_packages

setup(
    name='gensen',
    version='1.1.0',
    description="Learning General Purpose Distributed Sentence Representations",
    author_email='ibrahimsharafelden@gmail.com, ahmed.moorsy798@gmail.com',
    license='Proprietary: MIT',

    classifiers=[
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],
    keywords='Arabic sentence embeddings',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'numpy==1.18.2',
        'torch==1.4.0',
        'scikit-learn==0.22.1',
        'h5py==2.10.0',
        'nltk==3.5',
        'pyarabic==0.6.8',
        'sentencepiece==0.1.94',
        'langdetect==1.0.8',
        'tqdm==4.51.0'
    ],
    include_package_data=True,
)
