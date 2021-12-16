# TSJulia
Tensor Space Functionality for Julia Programming Platform.


| Stratification | Diagonal Blocks | Step Blocks |
|---------------------|------------------|----------------|
|<img src="docs/images/colossus-X-recon.png" alt="Curve" style="width:75%"> | <img src="docs/images/Blocks.png" alt="Diagonal Blocks" style="width:75%">  | <img src="docs/images/Adj-decomp-recon.png" alt="Steps" style="width:75%"> |







[TheTensor.Space](https://TheTensor.Space/) is an open-source research project dedicated to studying the data types and algorithms for extracting information form high valence tables of numbers, what the sciences call a **tensor**.  The data comes in looking like this:
<img src="docs/images/colossus-X-random.png" alt="Curve" style="width:50%">

Then using the algebra of operators on tensors we recover structures such as those displayed above.  These are not throwing out information but simply modifying the frame of references to reveal internally held structure within the data.

Our algorithms are provided in a number of platforms. 
  * The bleeding edge algorithms are developed and tested for [Magma Computer Algebra System](http://magma.maths.usyd.edu.au/magma/).  Core tensor algorithms are distributed with that system and you can follow [TheTensor.Space](https://TheTensor.Space/) for details on extensions and experimental additions.
  * Python access is available to core algorithms through [SageTensorSpace](https://github.com/thetensor-space/SageTensorSpace) for the [Sage Math](https://www.sagemath.org/) (in Python).
  * [Julia](https://julialang.org/) language port is being developed as [TSJulia](https://github.com/thetensor-space/TSJulia).

The algorithms presented in this tutorial are for instructional purposes.  For detailed treatments and improved performance follow the attached references.

---

### Installing TSJulia.

 - Make sure to have a recent installation of the Julia Language, v. > 1.7.0 seems to compatabile with the features required for TSJulia.  If you do not have an installation of Julia follow the installation instructions for the Julia system available [here](https://julialang.org/).
 - Clone or Download the TSJulia release from github [here](https://github.com/thetensor-space/TSJulia).  Make sure `julia` can be run from whatever folder contains your `TSJulia` download, typically by ensuring that `julia` is in the path of your operating system shell.
 - From the command line start julia and load the `TSJulia` package by using `include("$path$/TSJulia/TSJulia.jl")`

---

### Experiment 1: Stratification.

A tensor supported on a surface.
![](docs/images/colossus-X-orig.png)

A random change of basis to the above tensor.
![](docs/images/colossus-X-random.png)

The result of reconstruction 
![](docs/images/colossus-X-recon.png)
---

### Experiment 2: Simples Blocks

---


## Our Team

We invite you explore the repository and join our team.  We welcome and encourage any contributions to the repository. If you need help getting started, please feel free to @-mention any of the contributors below or you can read the repository's [Projects](https://github.com/thetensor-space/TensorSpace/projects) tab.

|                                                                              | Name                | Username                         | Affiliation                |
-------------------------------------------------------------------------------|---------------------|----------------------------------|----------------------------|
<img src="https://avatars.githubusercontent.com/galois60" height="50px"/>      | Prof. Peter A. Brooksbank | [`@galois60`](https://github.com/galois60)                | Bucknell University |
<img src="https://avatars.githubusercontent.com/kassabov" height="50px"/>  | Prof. Martin Kassabov     | [`@kassabov`](https://github.com/kassabov)        | Cornell University  |
<img src="https://avatars.githubusercontent.com/amaury-minino" height="50px"/>       | Amaury V. Miniño    | [`@amaury-minino`](https://github.com/amaury-minino)                  | Colorado State University |
<img src="https://avatars.githubusercontent.com/algeboy" height="50px"/>       | Prof. James B. Wilson     | [`@algeboy`](https://github.com/algeboy)                  | Colorado State University |


## Installation and Usage

For details on installation and usage visit the package website at [https://TheTensor.Space/TSJulia](https://TheTensor.Space/TSJulia).

