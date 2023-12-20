from setuptools import find_packages,setup
from typing import List

minus_e_dot='-e .'

def get_req(file_path:str)->List[str]:
    req=[]

    with open(file_path) as file_object:
        req=file_object.readlines()
        req=[r.replace("\n","") for r in req]
        
        if(minus_e_dot in req):
            req.remove(minus_e_dot)

    return req


setup(
    name='Ml_project',
    version='0.0.1',
    author="arkaprabha banerjee",
    author_email="arkaofficial13@gmail.com",
    packages=find_packages(),
    install_requires=get_req('requirements.txt')

)