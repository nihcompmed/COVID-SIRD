# Simulating SARS-CoV-2 Spread
## Interactive notebook
Use Binder to run our code online.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nihcompmed/SIRD/master)

## If you'd like to keep a copy of this on your local machine:
### Linux with Singularity container:
Clone the repository
```bash
    USER$ git clone https://github.com/nihcompmed/SIRD
    USER$ cd SIRD
```
Use Singularity to pull and generate container which contains necessary enviornment to run the exmaple code
```bash
USER$ sudo singularity pull docker://evancresswell/sird:#TAG#
USER$ sudo singularity build sird.simg sird_#TAG#.sif
```
Run the example notebook through Singularity Container
```bash
USER$ singularity exec -B </path/to/SIRD/> </path/to/SIRD>/sird.simg jupyter notebook SIRD_example.ipynb
```
* ``singularity exec`` - Ask singularity to execute a command in container.
* `` -B </path/to/SIRD/>`` - Mount your user specific directory. 
* ``</path/to/SIRD>/sird.simg`` - Image to launch.

If you're using remote computing resources you will need to start ssh tunneling to display notebook
