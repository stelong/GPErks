* add drocstrings to all modules, functions, classes, and methods
* usage documentation (with step-by-step) + concise API
* handle nested object (e.g. Product Kernel, Sum Kernel) while (de)serializing (Christian suggested some code modification for this already)
* extend Plottable whenever possible and implement default plot()
* pip-tools-multi for requirements
* update notebooks to the latest codebase as done for the examples (update: nearly done, check all comments and remains of old code)
* create abstract class for test functions with shared attribute (d=input dimension) and variable analytic form
* think of logic to pull from remote location (dropbox?) dataset json files from available collection of datasets (currently pushed to the repo as local resources)
* implement sensible unit tests to be run as actions
* consider changing installation to use toml file (update: pip works just fine, just need to take care of torch correctly)
* consider using Ruff
* use logging more extensively by replacing all occurrences of 'print' in the source code
* fix github actions and workflows to only perform basic checks