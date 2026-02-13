assert(get_option('python').enabled(),
       'The meson.build file for the Python wrapper should only be run if ' +
       'the "python" feature is enabled')

python3 = import('python').find_installation('python3')
python3_dep = python3.dependency(embed : true)

python3.extension_module(
  'butterfly',
  'butterfly.pyx',
  dependencies : [python3_dep, butterfly_dep, numpy_dep],
  install : true
)

python3.extension_module(
  'butterfly_hier',
  'butterfly_hier.pyx',
  dependencies : [python3_dep, butterfly_dep, numpy_dep],
  install : true
)

python3.extension_module(
  'butterfly_dev',
  'butterfly_dev.pyx',
  dependencies : [python3_dep, butterfly_dep, numpy_dep],
  install : true
)

python3.extension_module(
  'thermal_bf',
  sources : [
    'thermal_bf.pyx',
    pcc_src,
  ],
  include_directories : [
    butterfly_inc,
    pcc_inc,
  ],
  dependencies : [
    butterfly_dep,
    python3_dep,
    numpy_dep,
  ],
  install : true,
)
