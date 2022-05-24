import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open('environment.yml'))

requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        print(f'Dep {dep}, Len {len(dep.split("="))}, Val {dep.split("=")}')
        if len(dep.split("=")) > 2:
            package, package_version, python_version = dep.split('=')
        else:
            package, package_version = dep.split('=')
            python_version = ""
        if python_version == '0':
            continue
        requirements.append(package + '==' + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get('pip', []):
            requirements.append(preq)

with open('requirements.txt', 'w') as fp:
    for requirement in requirements:
       print(requirement, file=fp)