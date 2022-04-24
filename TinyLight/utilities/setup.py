from distutils.core import setup, Extension
import os


def get_agent():
    module_list = []
    for file_name in os.listdir():
        if file_name.endswith(".cpp"):
            agent = file_name[:-4]
            module = Extension(agent,
                               sources=[file_name],
                               language='c++')
            module_list.append(module)
    return module_list


setup(name='mcu_agent',
      version='1.0',
      description='mcu agent',
      ext_modules=get_agent())
