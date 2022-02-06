from setuptools import setup

setup(name='padisi_modules',
      version='0.1',
      description='This library is implemented to handle the PADISI-USC dataset',
      url='',
      author='Leonidas Spinoulas',
      author_email='lspinoulas@isi.edu',
      include_package_data=True,
      packages=['padisi_modules',
                'padisi_modules.dataIO',
                'padisi_modules.utils',
                'padisi_modules.dataset_utils',
                'padisi_modules.dataset_utils.finger_datasets',
                'padisi_modules.dataset_utils.face_datasets'],
      zip_safe=False)
