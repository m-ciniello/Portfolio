from setuptools import setup

entry_points = {'console_scripts':['init_model=run_models.init_model:main']}

setup(name='run_models',
      version='0.1.dev',
      description='Package for running models',
      keywords='dope as model dev tool',
      url='http://github.com/m-ciniello/run_models',
      author='Mike Ciniello',
      author_email='michael.ciniello@gmail.com',
      #include_package_data=True, # TODO: Dont need this for now
      license='MIT',
      packages=['run_models'],
      zip_safe=False,
      install_requires=['keras','tensorflow','numpy'], # For command line functionality
      entry_points = entry_points,
      test_suite='nose.collector', # For testing
      tests_require=['nose'])